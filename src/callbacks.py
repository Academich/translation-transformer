from pathlib import Path
from typing import Any

from pytorch_lightning import Callback, Trainer, LightningModule
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.callbacks import BasePredictionWriter


class DecodingCallback(Callback):
    def __init__(self):
        super().__init__()
        self.validation_step_outputs = []

    def on_validation_batch_end(
            self,
            trainer: "Trainer",
            pl_module: "LightningModule",
            outputs: STEP_OUTPUT,
            batch: Any,
            batch_idx: int,
            dataloader_idx: int = 0,
    ) -> None:
        self.validation_step_outputs.append(outputs)

    def on_validation_epoch_end(self, trainer: "Trainer", pl_module: "LightningModule") -> None:
        tkz = pl_module.tgt_tokenizer
        total_correct, total = 0, 0
        for o in self.validation_step_outputs:
            pred_tokens = o["pred_tokens"].cpu().numpy()
            target_ahead = o["target_ahead"].cpu().numpy()
            b_size = pred_tokens.shape[0]
            for i in range(b_size):
                target_str = tkz.decode(target_ahead[i])
                predicted_str = tkz.decode(pred_tokens[i])
                total_correct += int(predicted_str == target_str)
                total += 1
        trainer.logger.log_metrics({"val/whole_seq_exact_match_acc_total": total_correct / total})
        self.validation_step_outputs.clear()


class PredictionWriter(BasePredictionWriter):

    def __init__(self, output_dir, write_interval='batch'):
        super().__init__(write_interval)
        self.output_path = Path(output_dir).resolve()
        self.output_path.unlink(missing_ok=True)
        self.output_path.parent.mkdir(exist_ok=True)

    def write_on_batch_end(
            self, trainer, pl_module, prediction, batch_indices, batch, batch_idx, dataloader_idx
    ):
        tkz = pl_module.tgt_tokenizer
        with open(self.output_path, "a") as f:
            if f.tell() == 0:
                print(",".join(["source", "target", "prediction"]), file=f)
            src = batch["src_tokens"].cpu().numpy()
            tgt = batch["tgt_tokens"].cpu().numpy()
            for i, (s, t) in enumerate(zip(src, tgt)):
                s_string = tkz.decode(s)
                t_string = tkz.decode(t)
                # p_options = tkz.decode_batch(prediction[i].cpu().numpy()) # TODO Use this
                p_options = [tkz.decode(prediction[i].cpu().numpy())]
                print(",".join([s_string, t_string] + p_options), file=f)
