"""This script tests that the output of the model with various decoding strategies does not depend on the batch size."""

from pathlib import Path
import pytest

from model import VanillaEncoderDecoderTransformerLightning
from data_handling.tokenizer_smiles import ChemSMILESTokenizer

import torch
from torch.nn.utils.rnn import pad_sequence

from decoding.standard_decoding import (
    TranslationInferenceGreedy,
    TranslationInferenceBeamSearch,
)
from decoding.speculative_decoding import (
    TranslationInferenceGreedySpeculative,
    TranslationInferenceBeamSearchSpeculativeUnbatched,
)


THIS_SCRIPT_PATH = Path(__file__).resolve()
CKPT_DIR = Path("checkpoints/reaction_prediction")
CKPT_PATH = CKPT_DIR / "last.ckpt"
VOCAB_PATH = CKPT_DIR / "vocab.json"


def get_tokenizer() -> ChemSMILESTokenizer:
    tokenizer = ChemSMILESTokenizer()
    tokenizer.load_vocab(VOCAB_PATH)
    return tokenizer


def get_model(tokenizer: ChemSMILESTokenizer) -> torch.nn.Module:
    model = VanillaEncoderDecoderTransformerLightning(
        embedding_dim=256,
        feedforward_dim=2048,
        num_encoder_layers=4,
        num_decoder_layers=4,
        num_heads=8,
        dropout_rate=0.1,
        activation="relu",
        src_tokenizer=tokenizer,
        tgt_tokenizer=tokenizer,
    )

    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(CKPT_PATH, weights_only=True, map_location=device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model.model


def get_sampler(name):
    if name == "greedy":
        return TranslationInferenceGreedy(
            model=MODEL,
            max_len=200,
            pad_token=TOKENIZER.pad_token_idx,
            bos_token=TOKENIZER.bos_token_idx,
            eos_token=TOKENIZER.eos_token_idx,
        )
    elif name == "beam_search":
        return TranslationInferenceBeamSearch(
            model=MODEL,
            beam_size=5,
            max_len=200,
            pad_token=TOKENIZER.pad_token_idx,
            bos_token=TOKENIZER.bos_token_idx,
            eos_token=TOKENIZER.eos_token_idx,
        )
    elif name == "greedy_speculative":
        return TranslationInferenceGreedySpeculative(
            model=MODEL,
            max_len=200,
            n_speculative_tokens=10,
            max_drafts_num=5,
            pad_token=TOKENIZER.pad_token_idx,
            bos_token=TOKENIZER.bos_token_idx,
            eos_token=TOKENIZER.eos_token_idx,
        )
    elif name == "beam_search_speculative":
        return TranslationInferenceBeamSearchSpeculativeUnbatched(
            model=MODEL,
            max_len=200,
            n_speculative_tokens=10,
            n_best=5,
            pad_token=TOKENIZER.pad_token_idx,
            bos_token=TOKENIZER.bos_token_idx,
            eos_token=TOKENIZER.eos_token_idx,
        )
    else:
        raise ValueError(f"Sampler {name} not found")


def product_prediction_sample(
    tokenizer: ChemSMILESTokenizer,
) -> tuple[torch.LongTensor, torch.LongTensor]:
    with open(THIS_SCRIPT_PATH.parent / "product_prediction_src_test.txt") as f:
        src = [torch.tensor(tokenizer.encode(i.strip())).long() for i in f.readlines()]
        src_batch = pad_sequence(
            src, batch_first=True, padding_value=tokenizer.pad_token_idx
        )
    with open(THIS_SCRIPT_PATH.parent / "product_prediction_tgt_test.txt") as f:
        tgt = [torch.tensor(tokenizer.encode(i.strip())).long() for i in f.readlines()]
        tgt_batch = pad_sequence(
            tgt, batch_first=True, padding_value=tokenizer.pad_token_idx
        )
    return src_batch, tgt_batch


TOKENIZER = get_tokenizer()
MODEL = get_model(TOKENIZER)
SRC_BATCH, TGT_BATCH = product_prediction_sample(TOKENIZER)


@pytest.fixture
def get_predictions() -> tuple[torch.LongTensor, list[torch.LongTensor]]:
    generator = get_sampler("greedy_speculative")
    pred_batch_full = generator.generate(SRC_BATCH)
    pred_batch_one = []
    for entry in SRC_BATCH:
        pred_batch_one.append(generator.generate(entry.unsqueeze(0)).squeeze(0))
    return pred_batch_full, pred_batch_one


@pytest.mark.parametrize("idx", range(SRC_BATCH.size(0)))
def test_product_prediction(get_predictions, idx):
    t = TGT_BATCH[idx]
    pred_batch_full, pred_batch_one = get_predictions
    pred_b_full = pred_batch_full[idx]
    pred_b_one = pred_batch_one[idx]

    t_string = TOKENIZER.decode(t.cpu().numpy())
    b_one_strings = TOKENIZER.decode_batch(pred_b_one.cpu().numpy())
    b_full_strings = TOKENIZER.decode_batch(pred_b_full.cpu().numpy())

    print(f"Test case {idx}:")
    print("TARGET:", [t_string])
    print("B_ONE:", b_one_strings)
    print("B_ALL:", b_full_strings)

    for s1, s2 in zip(b_one_strings, b_full_strings):
        assert s1 == s2, f"Mismatch between batch and single predictions"
