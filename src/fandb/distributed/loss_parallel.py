import contextlib
from abc import abstractmethod
from typing import Protocol

import torch
import torch.distributed.tensor.parallel


class TrainContext(Protocol):
    @abstractmethod
    def __call__(self) -> contextlib.AbstractContextManager[None]:
        pass


def get_train_context(enable_loss_parallel: bool) -> TrainContext:
    @contextlib.contextmanager
    def context():
        with contextlib.ExitStack() as stack:
            if enable_loss_parallel:
                stack.enter_context(torch.distributed.tensor.parallel.loss_parallel())

            yield

    return context
