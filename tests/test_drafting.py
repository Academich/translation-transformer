"""This script test the draft generator at various parameters"""

import pytest

from pathlib import Path
from itertools import product

from torch import tensor, LongTensor
from torch.nn.utils.rnn import pad_sequence

from utils.drafting import make_drafts
from data_handling.tokenizer_smiles import ChemSMILESTokenizer

THIS_SCRIPT_PATH = Path(__file__).resolve()
CKPT_DIR = THIS_SCRIPT_PATH.parent.parent / "checkpoints" / "reaction_prediction"
VOCAB_PATH = CKPT_DIR / "vocab.json"
print(VOCAB_PATH)

DRAFT_LENGHTS = [1, 2, 3, 4, 5, 8, 10, 15, 25, 35, 50, 80, 100, 200]
DRAFT_AMOUNTS = [1, 2, 3, 5, 10, 15, 25, 35, 50, 80, 100, 200]
BATCH_SIZES = [1, 2, 3, 4, 5, 10]


def get_tokenizer() -> ChemSMILESTokenizer:
    tokenizer = ChemSMILESTokenizer()
    tokenizer.load_vocab(VOCAB_PATH)
    return tokenizer


def source_sample(
        tokenizer: ChemSMILESTokenizer
) -> LongTensor:
    with open(THIS_SCRIPT_PATH.parent / "product_prediction_src_test.txt") as f:
        src = [tensor(tokenizer.encode(i.strip())).long() for i in f.readlines()]
        src_batch = pad_sequence(
            src, batch_first=True, padding_value=tokenizer.pad_token_idx
        ).long()
    return src_batch


@pytest.mark.parametrize("bld", product(BATCH_SIZES, DRAFT_LENGHTS, DRAFT_AMOUNTS))
def test_drafting(bld):
    """
    Testing that the draft tensors will always have the requested shape
    """
    BATCH_SIZE, N_DRAFTS, DRAFT_LEN = bld
    tkz = get_tokenizer()
    src = source_sample(tkz)[:BATCH_SIZE, :]
    drafts = make_drafts(
        src,
        draft_len=DRAFT_LEN,
        n_drafts=N_DRAFTS,
        min_draft_len=1,
        max_draft_len=200,
        eos_token_idx=tkz.eos_token_idx,
        pad_token_idx=tkz.pad_token_idx,
        replace_token_idx=tkz.encoder_dict["c"]
    )
    for i in range(BATCH_SIZE):
        drafts_for_sequence = drafts[i]
        assert drafts_for_sequence.size() == (N_DRAFTS, DRAFT_LEN)

