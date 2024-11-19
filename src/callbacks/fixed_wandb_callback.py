from copy import deepcopy
import tempfile
import numbers
from pathlib import Path

from transformers.integrations.integration_utils import (
    WandbCallback,
    save_model_architecture_to_file,
)
from transformers.trainer import Trainer
from transformers.utils import logging

logger = logging.get_logger(__name__)


class FixedWandbCallback(WandbCallback):
   
    def on_train_end(self, args, state, control, model=None, tokenizer=None, **kwargs):
        if self._wandb is None:
            return
        if self._log_model.is_enabled and self._initialized and state.is_world_process_zero:
            
            # Copy the args to a new variable
            args_copy = deepcopy(args)
            
            args_copy.do_eval = False
            args_copy.do_train = False
            args_copy.do_predict = False
            
            args_copy.eval_strategy = "no"

            fake_trainer = Trainer(args=args_copy, model=model, processing_class=tokenizer)
            with tempfile.TemporaryDirectory() as temp_dir:
                fake_trainer.save_model(temp_dir)
                metadata = (
                    {
                        k: v
                        for k, v in dict(self._wandb.summary).items()
                        if isinstance(v, numbers.Number) and not k.startswith("_")
                    }
                    if not args.load_best_model_at_end
                    else {
                        f"eval/{args.metric_for_best_model}": state.best_metric,
                        "train/total_floss": state.total_flos,
                        "model/num_parameters": self._wandb.config.get("model/num_parameters"),
                    }
                )
                metadata["final_model"] = True
                logger.info("Logging model artifacts. ...")
                model_name = (
                    f"model-{self._wandb.run.id}"
                    if (args.run_name is None or args.run_name == args.output_dir)
                    else f"model-{self._wandb.run.name}"
                )
                # add the model architecture to a separate text file
                save_model_architecture_to_file(model, temp_dir)

                artifact = self._wandb.Artifact(name=model_name, type="model", metadata=metadata)
                for f in Path(temp_dir).glob("*"):
                    if f.is_file():
                        with artifact.new_file(f.name, mode="wb") as fa:
                            fa.write(f.read_bytes())
                self._wandb.run.log_artifact(artifact, aliases=["final_model"])