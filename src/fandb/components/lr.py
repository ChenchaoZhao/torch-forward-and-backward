from dataclasses import dataclass
from enum import StrEnum, auto

from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from transformers.optimization import get_scheduler

from fandb.components.configuration import ConfigMixin


class SchedulerNameEnum(StrEnum):
    LINEAR = auto()
    COSINE = auto()
    CONSTANT = auto()


@dataclass
class LrSchedulerConfig(ConfigMixin):
    """Learning rate scheduler configuration.

    Attributes:
        warmup_fraction: Fraction of training steps for warmup (0.0 to 1.0)
        name: Type of learning rate scheduler to use
    """

    warmup_fraction: float = 0.2
    name: SchedulerNameEnum = SchedulerNameEnum.LINEAR

    def __post_init__(self):
        self.name = SchedulerNameEnum(self.name)
        self.warmup_fraction = float(self.warmup_fraction)
        if self.warmup_fraction > 1.0 or self.warmup_fraction < 0.0:
            msg = f"warmup_fraction must be between 0.0 and 1.0, got {self.warmup_fraction}"
            raise ValueError(msg)
        self._num_training_steps: int | None = None

    def set_num_training_steps(self, num_training_steps: int):
        num_training_steps = int(num_training_steps)
        if num_training_steps < 0:
            raise ValueError("num_training_steps should be >= 0")
        self._num_training_steps = num_training_steps

    @property
    def num_training_steps(self) -> int:
        if self._num_training_steps is None:
            raise RuntimeError("num_training_steps has not been set, call `set_num_training_steps`")
        return self._num_training_steps

    @property
    def num_warmup_steps(self) -> int:
        return int(self.num_training_steps * self.warmup_fraction)

    def get_lr_scheduler(self, optimizer: Optimizer) -> LRScheduler:
        kwargs = {
            "name": str(self.name),
            "optimizer": optimizer,
            "num_training_steps": self.num_training_steps,
            "num_warmup_steps": self.num_warmup_steps,
        }
        return get_scheduler(**kwargs)
