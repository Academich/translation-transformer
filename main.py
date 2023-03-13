from pytorch_lightning.cli import LightningCLI

from src import TextTranslationTransformer
from src import CopySequence
from src import model_catalogue


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

        parser.add_argument("--model_name", "-m",
                            type=str,
                            choices=list(model_catalogue),
                            required=True, help="The model to use.")

    def before_instantiate_classes(self) -> None:
        self.model_class.module_class = model_catalogue[self.config.model_name]


if __name__ == '__main__':
    model_class = TextTranslationTransformer
    cli = FlexibleCLI(
        model_class=TextTranslationTransformer,
        datamodule_class=CopySequence,
        run=False
    )

    if cli.config.subcmd == "fit":
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "validate":
        cli.trainer.validate(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "test":
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "predict":
        cli.trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
