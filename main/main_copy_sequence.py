from pathlib import Path

from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.callbacks import BasePredictionWriter

from lightning_model_wrappers import TextTranslationTransformer
from synthetic_tasks.copy_sequence.data_module import CopySequence
from models import model_catalogue


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


class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='batch'):
        super().__init__(write_interval)
        self.output_path = Path(output_dir).resolve()
        self.output_path.unlink(missing_ok=True)
        self.output_path.parent.mkdir(exist_ok=True)

    def write_on_batch_end(
            self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        with open(self.output_path, "a") as f:
            _, tgt = batch
            for i, t in enumerate(tgt.cpu()):
                t_string = pl_module.tgt_vocab.decode(t)
                p_options = pl_module.tgt_vocab.decode_batch(prediction[i].cpu())
                print(",".join([t_string] + p_options), file=f)


if __name__ == '__main__':
    cb_list = [PredictionWriter("results/predictions.csv")]
    model_class = TextTranslationTransformer
    cli = FlexibleCLI(
        model_class=TextTranslationTransformer,
        datamodule_class=CopySequence,
        run=False,
        save_config_callback=None,
        trainer_defaults={"callbacks": cb_list}
    )

    if cli.config.subcmd == "fit":
        cli.trainer.fit(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "validate":
        cli.trainer.validate(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "test":
        cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
    if cli.config.subcmd == "predict":
        cli.trainer.predict(cli.model, datamodule=cli.datamodule, ckpt_path=cli.config.ckpt_path)
