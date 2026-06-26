from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytorch_lightning.callbacks as pl_callbacks
import torch
from pytorch_lightning.utilities import rank_zero_only


class MetricsTracker(pl_callbacks.Callback):
    """
    Robust metric tracker that records *already computed* values from
    trainer.callback_metrics (instead of calling Metric.compute()).

    This avoids crashes like:
      ValueError: No samples to concatenate
    from torchmetrics.AveragePrecision when metrics were reset / never updated.
    """

    def __init__(self, prefix_filter=("val_", "test_")):
        self.prefix_filter = tuple(prefix_filter)
        self.data = None

    @rank_zero_only
    def setup(self, trainer, pl_module, stage: str) -> None:
        # store time-series by metric name
        if self.data is None:
            self.data = {}

    @staticmethod
    def _to_float_scalar(x):
        """Return float(x) if scalar tensor/number, else None."""
        if x is None:
            return None
        if isinstance(x, torch.Tensor):
            if x.numel() != 1:
                return None
            x = x.detach()
            if x.is_cuda:
                x = x.cpu()
            x = x.item()
        try:
            return float(x)
        except Exception:
            return None

    @rank_zero_only
    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics  # already computed by Lightning

        for name, value in metrics.items():
            # Keep only validation metrics by default (customize via prefix_filter)
            if self.prefix_filter and not any(str(name).startswith(p) for p in self.prefix_filter):
                continue

            # Skip step-level spam if you want:
            # if str(name).endswith("_step"): continue

            v = self._to_float_scalar(value)
            if v is None:
                continue

            # optional: skip NaNs/infs
            if not np.isfinite(v):
                continue

            self.data.setdefault(str(name), []).append((trainer.current_epoch, v))

    # If you also want test metrics, keep this; otherwise remove it.
    @rank_zero_only
    def on_test_epoch_end(self, trainer, pl_module) -> None:
        if trainer.sanity_checking:
            return
        metrics = trainer.callback_metrics
        for name, value in metrics.items():
            if self.prefix_filter and not any(str(name).startswith(p) for p in self.prefix_filter):
                continue
            v = self._to_float_scalar(value)
            if v is None or not np.isfinite(v):
                continue
            self.data.setdefault(str(name), []).append((trainer.current_epoch, v))

    @rank_zero_only
    def on_save_checkpoint(self, trainer, pl_module, checkpoint: Dict[str, Any]) -> None:
        if not self.data:
            return

        save_dir = Path(trainer.default_root_dir) / "metrics"
        save_dir.mkdir(exist_ok=True)

        for name, values in self.data.items():
            # values is list of (epoch, scalar)
            arr = np.asarray(values, dtype=np.float32)
            np.savetxt(save_dir / f"{name}.gz", arr)

    def state_dict(self):
        return {} if self.data is None else self.data.copy()

    def load_state_dict(self, state_dict):
        if self.data is None:
            self.data = state_dict.copy()
        else:
            self.data.update(state_dict)


# from pathlib import Path
# from typing import Any, Dict

# import numpy as np
# import pytorch_lightning as pl
# import pytorch_lightning.callbacks as pl_callbacks
# import torch
# import torchmetrics
# from pytorch_lightning.utilities import rank_zero_only


# class MetricsTracker(pl_callbacks.Callback):
#     def __init__(self):
#         self.metrics = None
#         self.data = None

#     @rank_zero_only
#     def setup(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule", stage: str
#     ) -> None:
#         self.metrics = {
#             name: value
#             for name, value in pl_module.named_children()
#             if isinstance(value, torchmetrics.Metric)
#         }
#         self.data = {name: [] for name in self.metrics.keys()}

#     @rank_zero_only
#     def on_validation_end(
#         self, trainer: "pl.Trainer", pl_module: "pl.LightningModule"
#     ) -> None:
#         if trainer.sanity_checking:
#             return
#         for name in self.metrics.keys():
#             computed = self.metrics[name].compute()

#             # Some metrics return non-scalar tensors (e.g. ConfusionMatrix returns [C,C]).
#             # This callback tracks scalar time-series only.
#             if isinstance(computed, torch.Tensor) and computed.numel() != 1:
#                 continue

#             value = float(computed)
#             self.data[name].append((trainer.current_epoch, value))

#     @rank_zero_only
#     def on_save_checkpoint(
#         self,
#         trainer: "pl.Trainer",
#         pl_module: "pl.LightningModule",
#         checkpoint: Dict[str, Any],
#     ) -> None:
#         save_dir = Path(trainer.default_root_dir) / "metrics"
#         save_dir.mkdir(exist_ok=True)

#         # easier to load this way quickly for plotting etc.
#         for name, values in self.data.items():
#             np.savetxt(save_dir / f"{name}.gz", values)

#     # easier to save this way to resume training etc.
#     def state_dict(self):
#         if self.data is None:
#             return {}
#         return self.data.copy()

#     def load_state_dict(self, state_dict):
#         if self.data is None:
#             self.data = state_dict.copy()
#         else:
#             self.data.update(state_dict)
