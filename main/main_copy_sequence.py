from pathlib import Path

from pytorch_lightning.cli import LightningCLI

from models import VanillaTransformerTranslationLightningModule
from data_wrappers import Seq2SeqDM
from synthetic_tasks.copy_sequence.tokenizer import AsciiTokenizer

from callbacks import PredictionWriter, DecodingCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class FlexibleCLI(LightningCLI):
    """
    A CLI that allows using subcommands together with run=False.
    """

    def add_arguments_to_parser(self, parser):
        parser.add_argument(
            "--subcmd",
            "-s",
            type=str,
            choices=["fit", "validate", "test", "predict", "tune"],
            required=True,
            help="""
            'fit' - Run the full optimization routine.
            'validate' - Perform one evaluation epoch over the validation set.
            'test' - Perform one evaluation epoch over the test set.
            'predict' - Run inference on your data.
            'tune' - Runs routines to tune hyperparameters before training."""
        )
        parser.add_argument("--ckpt_path", type=str, default=None,
                            help="""
            Path/URL of the checkpoint from which training is resumed.
            Could also be one of two special keywords ``"last"`` and ``"hpc"``.
            If there is no checkpoint file at the path, an exception is raised.
            If resuming from mid-epoch checkpoint, training will start from the beginning of the next epoch.
            (type: Union[str, null])""")

        parser.add_argument("--src_vocab_path", type=str, default=None,
                            help="""
                    Path to the source sequence tokenizer vocabulary JSON""")

        parser.add_argument("--tgt_vocab_path", type=str, default=None,
                            help="""
                    Path to the target sequence tokenizer vocabulary JSON""")

        parser.add_argument("--write_predictions_path", type=str, default=None,
                            help="""
                            Path to the file to store the decoded predictions to""")

    def before_instantiate_classes(self) -> None:
        # Creating tokenizers
        src_tokenizer = AsciiTokenizer()
        tgt_tokenizer = AsciiTokenizer()
        if self.config.src_vocab_path is not None:
            src_tokenizer.load_vocab(self.config.src_vocab_path)
        else:
            data_dir = Path(self.config.data.data_dir).resolve()
            save_vocab_dir = data_dir / "vocabs"
            save_vocab_dir.mkdir(parents=True, exist_ok=True)
            src_tokenizer.train_tokenizer(data_dir / "src-train.txt")
            src_tokenizer.save_vocab(save_vocab_dir / "src-vocab.json")
        if self.config.tgt_vocab_path is not None:
            tgt_tokenizer.load_vocab(self.config.tgt_vocab_path)
        else:
            data_dir = Path(self.config.data.data_dir).resolve()
            save_vocab_dir = data_dir / "vocabs"
            save_vocab_dir.mkdir(parents=True, exist_ok=True)
            tgt_tokenizer.train_tokenizer(data_dir / "tgt-train.txt")
            tgt_tokenizer.save_vocab(save_vocab_dir / "tgt-vocab.json")

        # Connecting the tokenizers to the model and the data module
        self.config.data.src_tokenizer = src_tokenizer
        self.config.data.tgt_tokenizer = tgt_tokenizer
        self.config.model.init_args.src_vocab_size = src_tokenizer.n_tokens
        self.config.model.init_args.tgt_vocab_size = tgt_tokenizer.n_tokens
        self.config.model.init_args.pad_token_idx = tgt_tokenizer.pad_token_idx
        self.config.model.init_args.bos_token_idx = tgt_tokenizer.bos_token_idx
        self.config.model.init_args.eos_token_idx = tgt_tokenizer.eos_token_idx

        # Creating callbacks
        if self.config.write_predictions_path is None:
            self.config.write_predictions_path = "results/copy_sequence_predictions.csv"
        cb_list = [
            PredictionWriter(self.config.write_predictions_path, tgt_tokenizer),
            DecodingCallback(tgt_tokenizer),
            LearningRateMonitor(logging_interval='step'),
            ModelCheckpoint(
                monitor="val/acc_single_tok",
                auto_insert_metric_name=False,
                filename='step={step}-val_tok_acc={val/acc_single_tok:.3f}-val_l={val/loss:.3f}',
                mode='max',
                save_last=True,
                save_top_k=1,
                every_n_epochs=1,
                save_on_train_epoch_end=False,
                enable_version_counter=False
            )]

        # Connecting callbacks to the trainer instance
        if self.config.trainer.callbacks is None:
            self.config.trainer.callbacks = []
        for cb in cb_list:
            self.config.trainer.callbacks.append(cb)


if __name__ == '__main__':
    cli = FlexibleCLI(
        model_class=None,  # the model class should be provided in the CLI
        datamodule_class=Seq2SeqDM,
        run=False,
        save_config_callback=None,
    )

    if cli.config.subcmd == "fit":
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "validate":
        cli.trainer.validate(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "test":
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "predict":
        cli.trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
