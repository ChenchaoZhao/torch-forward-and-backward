from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING

import torch
from loguru import logger
from torch.distributed.algorithms.ddp_comm_hooks import (
    default_hooks,
    powerSGD_hook,
)
from torch.distributed.device_mesh import DeviceMesh
from torch.nn import Module
from torch.nn.parallel import DistributedDataParallel

from fandb.components.configuration import ConfigMixin

if TYPE_CHECKING:
    from collections.abc import Callable


class DDPCommunicationHookType(StrEnum):
    """
    Represents a type of communication hook used in DDP.

    Values:

        - **NO** -- no communication hook
        - **FP16** -- DDP communication hook to compress the gradients in FP16
        - **BF16** -- DDP communication hook to compress the gradients in BF16
        - **POWER_SGD** -- DDP communication hook to use PowerSGD
        - **BATCHED_POWER_SGD** -- DDP communication hook to use batched
          PowerSGD
    """

    NO = "no"
    FP16 = "fp16"
    BF16 = "bf16"
    POWER_SGD = "power_sgd"
    BATCHED_POWER_SGD = "batched_power_sgd"


@dataclass
class DistributedDataParallelConfig(ConfigMixin):
    """Configuration for DistributedDataParallel model wrapping.

    This class provides configuration options for wrapping PyTorch models with
    DistributedDataParallel (DDP). It includes settings for communication hooks,
    gradient compression, and various DDP-specific parameters that control
    how distributed training is performed across multiple devices.

    Attributes:
        dim: The dimension to use for sharding.
        broadcast_buffers: Whether to broadcast buffers from rank 0 to all other processes.
        bucket_cap_mb: The bucket size in MB for gradient reduction.
        find_unused_parameters: Whether to find unused parameters in the model.
        check_reduction: Whether to check if gradients are correctly reduced.
        gradient_as_bucket_view: Whether to use gradient as bucket view.
        static_graph: Whether to use static graph optimization.
        comm_hook: The communication hook type for gradient compression.
        comm_wrapper: The communication wrapper type.
        comm_state_option: Additional options for communication state.
    """

    dim: int = 0
    broadcast_buffers: bool = True
    bucket_cap_mb: int = 25
    find_unused_parameters: bool = False
    check_reduction: bool = False
    gradient_as_bucket_view: bool = False
    static_graph: bool = False

    comm_hook: DDPCommunicationHookType = DDPCommunicationHookType.NO
    comm_wrapper: DDPCommunicationHookType = DDPCommunicationHookType.NO
    comm_state_option: dict = field(default_factory=dict)

    def to_dict(self, ignore_keys=("comm_hook", "comm_wrapper", "comm_state_option")):
        return {k: v for k, v in super().to_dict().items() if k not in ignore_keys}

    def register_comm_hook(self, model: DistributedDataParallel):
        if self.comm_hook == DDPCommunicationHookType.NO:
            return

        hook_map: dict[DDPCommunicationHookType, Callable] = {
            DDPCommunicationHookType.FP16: default_hooks.fp16_compress_hook,
            DDPCommunicationHookType.BF16: default_hooks.bf16_compress_hook,
            DDPCommunicationHookType.POWER_SGD: powerSGD_hook.powerSGD_hook,
            DDPCommunicationHookType.BATCHED_POWER_SGD: powerSGD_hook.batched_powerSGD_hook,
        }

        wrapper_map: dict[DDPCommunicationHookType, Callable] = {
            DDPCommunicationHookType.FP16: default_hooks.fp16_compress_wrapper,
            DDPCommunicationHookType.BF16: default_hooks.bf16_compress_wrapper,
        }

        hook: Callable | None = hook_map.get(self.comm_hook)
        wrapper: Callable | None = wrapper_map.get(self.comm_wrapper)

        if hook and wrapper:
            hook = wrapper(hook)

        if hook:
            state = (
                powerSGD_hook.PowerSGDState(None, **self.comm_state_option)
                if self.comm_hook
                in (
                    DDPCommunicationHookType.POWER_SGD,
                    DDPCommunicationHookType.BATCHED_POWER_SGD,
                )
                else None
            )
            model.register_comm_hook(
                state=state,
                hook=hook,
            )

    def prepare_ddp_model(self, model: Module, dp_mesh: DeviceMesh) -> DistributedDataParallel:
        kwargs = self.to_dict()
        kwargs["device_mesh"] = dp_mesh
        logger.debug(f"DDP Kwargs: {kwargs}")

        return DistributedDataParallel(module=model, **kwargs)

    def maybe_enable_ddp_reducer(self, torch_compile: bool, activation_checkpoint: bool):
        if not (torch_compile and activation_checkpoint):
            logger.info("No config python reducer.")
            return

        # torch compile does not work with ddp + ac
        # python compiled faster than setting to False
        torch_version = torch.__version__.split("+", 1)[0]
        py_ddp = "python_reducer" if torch_version >= "2.7" else "python_reducer_without_compiled_forward"
        torch._dynamo.config.optimize_ddp = py_ddp  # noqa: SLF001

        logger.info(f"Use python ddp reducer for torch compile: {py_ddp}")
