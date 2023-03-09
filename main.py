from argparse import ArgumentParser
import json

import torch

import pytorch_lightning as pl
from pytorch_lightning import callbacks

from src import Seq2SeqDM, TokenVocabulary, TextTranslationTransformer, available_models
from src import TranslationInferenceGreedy, TranslationInferenceBeamSearch

# === Config ===
parser = ArgumentParser()

# PROGRAM level args
parser.add_argument("--seed", type=int, default=123456,
                    help="Random seed for reproducibility.")
parser.add_argument("--src_vocab_path", type=str,
                    help="Path to the vocabulary json (token index -> token) of the source sequence.")
parser.add_argument("--tgt_vocab_path", type=str,
                    help="Path to the vocabulary json (token index -> token) of the target sequence.")
parser.add_argument("--save_dir_path", type=str, default="saved_models",
                    help="Path to the folder with the saved models.")
parser.add_argument("--mode", type=str, choices=["train", "test", "predict"], required=True,
                    help="Work mode: training or testing. Possible values: 'train', 'test', 'predict'")
parser.add_argument("--model", type=str, choices=list(available_models), required=True,
                    help=f"Model architecture. Possible values: {list(available_models)}")
parser.add_argument("--checkpoint_name", type=str, default=None,
                    help="The name of the run. Used as a model checkpoint name and as a run name in W&B.")
parser.add_argument("--load_checkpoint", type=str, help="Path to the checkpoint to load. Necessary in the testing mode",
                    default=None)
parser.add_argument("--use_wandb", action="store_true", default=False,
                    help="A flag whether to use W&B for logging or not.")
parser.add_argument("--wandb_project_name", type=str,
                    help="The project name for W&B.")
parser.add_argument("--wandb_run_name", type=str, default=None,
                    help="The run name for the W&B experiment")

# TRAINER level args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(progress_bar_refresh_rate=10)

parser = TextTranslationTransformer.add_model_specific_args(parser)

# DATA MODULE level args
parser = Seq2SeqDM.add_argparse_args(parser)

args = parser.parse_args()

# === Ensure reproducibility ===
pl.seed_everything(args.seed)

# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.determinstic = True
torch.backends.cudnn.benchmark = False

# === Logger clients (W&B or Tensorboard)
logger, run_name = None, args.wandb_run_name
if args.use_wandb:
    import logging_clients

    logger = logging_clients.make_wandb_logger(args.wandb_project_name, run_name)
    # TODO wandb.config.update(args) (add to logging_clients)

# === Tokenizer vocabulary
with open(args.src_vocab_path) as f, open(args.tgt_vocab_path) as g:
    src_vocab = TokenVocabulary(json.load(f))
    tgt_vocab = TokenVocabulary(json.load(g))
assert src_vocab.pad_token_idx == tgt_vocab.pad_token_idx
assert src_vocab.eos_token_idx == tgt_vocab.eos_token_idx
assert src_vocab.bos_token_idx == tgt_vocab.bos_token_idx

# === Callbacks ===
ckpt_name = args.checkpoint_name + '_{epoch}-{step}' if args.checkpoint_name is not None else '{epoch}-{step}'
checkpoint_callback = callbacks.ModelCheckpoint(dirpath=args.save_dir_path,
                                                filename=ckpt_name,
                                                verbose=True,
                                                save_last=True)
progress_bar_callback = callbacks.progress.TQDMProgressBar(refresh_rate=args.progress_bar_refresh_rate)
del args.progress_bar_refresh_rate
cb_list = [checkpoint_callback, progress_bar_callback]

# === Instantiate classes ===
trainer = pl.Trainer.from_argparse_args(args,
                                        logger=logger,
                                        profiler=None,
                                        callbacks=cb_list)

if __name__ == '__main__':
    Architecture = available_models[args.model]
    if args.load_checkpoint is not None:
        # Test mode, train mode
        model = Architecture.load_from_checkpoint(args.load_checkpoint,
                                                  src_vocab=src_vocab,
                                                  tgt_vocab=tgt_vocab)
    else:
        # Only for train mode
        model = Architecture.from_argparse_args(args,
                                                src_vocab=src_vocab,
                                                tgt_vocab=tgt_vocab)

    dm = Seq2SeqDM.from_argparse_args(args)
    if args.mode == "train":
        trainer.fit(model=model,
                    train_dataloaders=dm)
    if args.mode == "test":
        trainer.test(model=model,
                     dataloaders=dm)
    if args.mode == "predict":
        generator = TranslationInferenceBeamSearch(model=model,
                                                   beam_size=1,
                                                   max_len=30,
                                                   pad_token=tgt_vocab.pad_token_idx,
                                                   bos_token=tgt_vocab.bos_token_idx,
                                                   eos_token=tgt_vocab.eos_token_idx)
        dm.setup("predict")
        dl = dm.predict_dataloader()
        with open("result.csv", "w") as f:
            for (src, tgt) in dl:
                generated_tokens = generator.generate(src)
                for i, seq in enumerate(generated_tokens):
                    target = [tgt_vocab.decode(tgt[i])]
                    prediction = [tgt_vocab.decode(option) for option in seq]
                    print(",".join(target + prediction), file=f)
