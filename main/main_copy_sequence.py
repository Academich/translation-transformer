from pytorch_lightning.cli import LightningCLI

from synthetic_tasks.copy_sequence.data_module import CopySequenceDM

from callbacks import PredictionWriter, DecodingCallback
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


class FlexibleCLI(LightningCLI):
    """
    A CLI that allows using subcommands together with run=False.
    """

    def add_arguments_to_parser(self, parser):
        parser.link_arguments("data.src_tokenizer", "model.init_args.src_tokenizer", apply_on="instantiate")
        parser.link_arguments("data.tgt_tokenizer", "model.init_args.tgt_tokenizer", apply_on="instantiate")


if __name__ == '__main__':
    cli = FlexibleCLI(
        model_class=None,  # the model class should be provided in the CLI
        datamodule_class=CopySequenceDM,
        save_config_kwargs={"overwrite": True},
        run=True
    )
