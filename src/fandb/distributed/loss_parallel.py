import contextlib
from abc import abstractmethod
from typing import Protocol

import torch
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel

# from torchtitan.config import CommConfig, DebugConfig, TORCH_DTYPE_MAP
# from torchtitan.distributed.parallel_dims import ParallelDims
# from torchtitan.tools.logging import logger
# from torchtitan.tools.utils import device_module, device_type


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
