import contextlib
import os
from dataclasses import dataclass, field

import torch
from loguru import logger

from fandb.components.comm import CommConfig
from fandb.components.configuration import ConfigMixin
from fandb.components.precision import TORCH_DTYPE_MAP, TorchDtypeEnum
from fandb.distributed.device_mesh import (
    DeviceMesh,
    DeviceTypeEnum,
    DPBackend,
    get_ddp_device_mesh,
    get_fsdp_device_mesh,
    get_fsdp_tp_device_mesh,
)
from fandb.distributed.utils import set_device


@dataclass
class ParallelConfig(ConfigMixin):
    """
    Configuration for distributed parallel training settings.

    This configuration manages various forms of parallelism for distributed
    training, including data parallelism (DP), tensor parallelism (TP),
    context parallelism (CP), pipeline parallelism (PP), and expert parallelism
    (EP). It supports different data parallel backends like FSDP (Fully Sharded
    Data Parallel) and DDP (Distributed Data Parallel).

    The configuration automatically infers data parallel replicate size based on
    world size and other parallelism dimensions. Call `maybe_init_distributed()`
    to initialize distributed training and set the appropriate device.

    Args:
        device_type (DeviceTypeEnum): Type of device to use for training.
            Defaults to DeviceTypeEnum.CUDA.
        dp_backend (DPBackend): Data parallelism backend to use (FSDP, DDP,
            or NO). Defaults to DPBackend.FSDP.
        dp_shard_size (int): Size of data parallel sharding dimension for FSDP.
            Must be positive. Defaults to 1.
        tp_size (int): Size of tensor parallelism dimension. Splits model
            tensors across processes. Must be positive. Defaults to 1.
        cp_size (int): Size of context parallelism dimension. Splits sequence
            dimension across processes. Defaults to 1.
        pp_size (int): Size of pipeline parallelism dimension. Splits model
            layers across processes. Defaults to 1.
        ep_size (int): Size of expert parallelism dimension for MoE models.
            Defaults to 1.
        etp_size (int): Size of expert tensor parallelism dimension.
            Defaults to 1.
        comm (CommConfig): Communication configuration for distributed training.
            Defaults to CommConfig().

    Properties:
        has_nd_parallel (bool): Whether n-dimensional parallelism is supported.
        is_distributed (bool): Whether distributed training is enabled.
        world_size (int): Total number of processes in distributed training.
        dp_replicate_size (int): Inferred data parallel replicate size based on
            world_size and other parallelism dimensions.
        rank (int): Global rank of current process.
        local_world_size (int): Number of processes on local node.
        local_rank (int): Local rank on current node.
        num_processes (int): Total number of processes (alias for world_size).
        process_index (int): Process index (alias for rank).
        local_process_index (int): Local process index (alias for local_rank).
        is_main_process (bool): Whether current process is main process.
        is_local_main_process (bool): Whether current process is local main.
        dp_world_size (int): Data parallel world size.
        dp_rank (int): Data parallel rank.
        dp_enabled (bool): Whether data parallelism is enabled.
        dp_replicate_enabled (bool): Whether data parallel replication is
            enabled.
        dp_shard_enabled (bool): Whether data parallel sharding is enabled.
        cp_enabled (bool): Whether context parallelism is enabled.
        dp_cp_enabled (bool): Whether data or context parallelism is
            enabled.
        fsdp_enabled (bool): Whether FSDP is enabled.
        tp_enabled (bool): Whether tensor parallelism is enabled.
        pp_enabled (bool): Whether pipeline parallelism is enabled.
        ep_enabled (bool): Whether expert parallelism is enabled.
        etp_enabled (bool): Whether expert tensor parallelism is enabled.
        fsdp_gradient_divide_factor (int): Gradient division factor for FSDP.
        non_data_parallel_size (int): Size of non-data parallel dimensions.
        seq_len_divisor (int): Sequence length divisor for sequence
            parallelism.
        device_mesh (DeviceMesh): N-dimensional device mesh for distributed
            training.
        device (torch.device): Current device for training.

    Methods:
        maybe_init_distributed(): Initializes distributed training if enabled.
            Sets up communication backend, creates device mesh, sets device,
            and infers data parallel replicate size. For non-distributed
            training, only sets the device.
        set_device(): Sets and returns the appropriate device for training.
            For distributed training, uses local_rank to set device. For
            non-distributed, uses the configured device_type.
        maybe_enable_amp_autocast(mixed_precision_param, device_type):
            Returns appropriate autocast context manager for mixed precision
            training. For FSDP backend, returns null context as mixed precision
            is handled internally. For DDP/single-device training, returns
            torch.autocast with specified dtype and device type.

    Note:
        The relationship between parallelism dimensions and world size is:
        world_size = dp_replicate_size * dp_shard_size * tp_size * cp_size *
        pp_size

        Expert parallelism (ep_size, etp_size) operates independently and is
        not part of the main world size calculation.

    Example:
        >>> config = ParallelConfig(dp_shard_size=2, tp_size=4, dp_backend=DPBackend.FSDP)
        >>> config.maybe_init_distributed()
    """

    device_type: DeviceTypeEnum = DeviceTypeEnum.CUDA
    dp_backend: DPBackend = DPBackend.FSDP
    dp_shard_size: int = 1  # fsdp or hsdp
    tp_size: int = 1
    cp_size: int = 1
    pp_size: int = 1
    ep_size: int = 1
    etp_size: int = 1
    comm: CommConfig = field(default_factory=CommConfig)

    def __post_init__(self):
        self._validation()
        self._dp_replicate_size: int | None = None
        self._device_mesh: DeviceMesh | None = None
        self._device: torch.device | None = None

    def _validation(self):
        self.device_type = DeviceTypeEnum(self.device_type)
        self.dp_backend = DPBackend(self.dp_backend)
        self.dp_shard_size = int(self.dp_shard_size)
        self.tp_size = int(self.tp_size)

        # Validate that dp_shard_size and tp_size are positive
        if self.dp_shard_size <= 0:
            msg = f"dp_shard_size must be a positive integer, got {self.dp_shard_size}"
            raise ValueError(msg)

        if self.tp_size <= 0:
            msg = f"tp_size must be a positive integer, got {self.tp_size}"
            raise ValueError(msg)

    def maybe_init_distributed(self):
        if not self.is_distributed:
            self.set_device()
            logger.success(f"No distributed training:\n  current device: {self.device}")
            return

        self.comm.init_distributed()
        self.comm.set_train_timeout()
        self._device_mesh = self._get_device_mesh()
        self.set_device()
        self._dp_replicate_size = self._infer_dp_replicate_size()
        logger.success(
            f"Distributed initialized:\n  current device: {self.device}\n  rank:{self.rank}\n world size:{self.world_size}"
        )

    @property
    def has_nd_parallel(self) -> bool:
        return self.dp_backend == DPBackend.FSDP

    @property
    def is_distributed(self) -> bool:
        return self.dp_backend != DPBackend.NO

    @property
    def world_size(self) -> int:
        return torch.distributed.get_world_size()

    @property
    def dp_replicate_size(self) -> int:
        if self._dp_replicate_size is None:
            self._dp_replicate_size = self._infer_dp_replicate_size()
        return self._dp_replicate_size

    @property
    def rank(self) -> int:
        return torch.distributed.get_rank()

    @property
    def local_world_size(self) -> int:
        return int(os.environ["LOCAL_WORLD_SIZE"])

    @property
    def local_rank(self) -> int:
        return int(os.environ.get("LOCAL_RANK", "-1"))

    @property
    def num_processes(self) -> int:
        return self.world_size

    @property
    def process_index(self) -> int:
        return self.rank

    @property
    def local_process_index(self) -> int:
        return self.local_rank

    @property
    def is_main_process(self) -> bool:
        return self.rank == 0

    @property
    def is_local_main_process(self) -> bool:
        return self.local_rank == 0

    @property
    def dp_world_size(self) -> int:
        # same as self.device_mesh["dp_replicate", "dp_shard"].size()
        return self.dp_replicate_size * self.dp_shard_size

    @property
    def dp_rank(self) -> int:
        # when device mesh is used, specifically with TP
        # then there is need to update process_index and num_processes
        # to bring in the effect of generating same batch across TP ranks
        # and different batch across FSDP and DP ranks.
        # Example:
        # if device mesh is (dp,fsdp,tp) = (2, 2, 3)
        # ranks would range from 0...11
        # from data angle ranks should look like 0 0 0 1 1 1 2 2 2 3 3 3
        # processes with same ranks/ids would receive the same batch
        # for CP the same as TP applies
        # same as self.device_mesh["dp_replicate", "dp_shard"].get_rank()
        return self.rank // self.non_data_parallel_size

    @property
    def dp_enabled(self):
        return self.dp_replicate_size > 1 or self.dp_shard_size > 1

    @property
    def dp_replicate_enabled(self):
        return self.dp_replicate_size > 1

    @property
    def dp_shard_enabled(self):
        return self.dp_shard_size > 1 and self.has_nd_parallel

    @property
    def cp_enabled(self):
        return self.cp_size > 1 and self.has_nd_parallel

    @property
    def dp_cp_enabled(self):
        return (self.dp_enabled or self.cp_enabled) and self.has_nd_parallel

    @property
    def fsdp_enabled(self):
        return (self.dp_shard_enabled or self.cp_enabled) and self.has_nd_parallel

    @property
    def tp_enabled(self):
        return self.tp_size > 1 and self.has_nd_parallel

    @property
    def pp_enabled(self):
        return self.pp_size > 1 and self.has_nd_parallel

    @property
    def ep_enabled(self):
        return self.ep_size > 1 and self.has_nd_parallel

    @property
    def etp_enabled(self):
        return self.etp_size > 1 and self.has_nd_parallel

    @property
    def fsdp_gradient_divide_factor(self) -> int:
        # This is needed for FSDP-sharded experts when Expert Parallel
        # is enabled. Although the FSDP sharding of experts is done on a
        # mesh of a different size than other parameters, the gradient
        # division factor should be consistent with data.
        return self.dp_replicate_size * self.dp_shard_size * self.cp_size

    @property
    def non_data_parallel_size(self):
        # dp_replicate * dp_shard * cp * tp * pp == world_size
        # expert parallels are not part of the world size!
        return self.cp_size * self.tp_size * self.pp_size

    @property
    def seq_len_divisor(self):
        # Sequence Parallel requires that seq_len be divisible by TP degree.
        # https://github.com/pytorch/torchtitan/pull/640#discussion_r1849481001

        # Context Parallel requires that seq_len be divisible by 2 * CP
        # degree, when load balancing is enabled (by default).
        # https://github.com/pytorch/pytorch/blob/4f62dcc/torch/distributed/tensor/experimental/_attention.py#L1246
        return self.tp_size * (self.cp_size * 2)

    def _infer_dp_replicate_size(self) -> int:
        """
        Infer data parallel replicate size from other parameters.

        The relationship is:
        world_size = dp_shard_size * dp_replicate_size * tp_size

        Returns:
            Inferred dp_replicate_size

        Raises:
            ValueError: If world_size is not divisible by
                (dp_shard_size * tp_size)
        """

        if self.dp_backend == DPBackend.NO:
            return 1

        if self.dp_backend == DPBackend.DDP:
            return self.world_size

        denominator = self.dp_shard_size * self.non_data_parallel_size

        if self.world_size % denominator != 0:
            msg = (
                f"world_size ({self.world_size}) must be divisible by "
                f"dp_shard_size ({self.dp_shard_size}) * "
                f"tp_size ({self.tp_size}) = {denominator}"
            )
            raise ValueError(msg)

        return self.world_size // denominator

    @property
    def device_mesh(self) -> DeviceMesh:
        if self._device_mesh is None:
            self._device_mesh = self._get_device_mesh()
        return self._device_mesh

    def _get_device_mesh(self) -> DeviceMesh:
        if not self.is_distributed:
            raise RuntimeError("No device mesh will be created as dp_backend is set to NO")

        if self.dp_backend == DPBackend.DDP:
            return get_ddp_device_mesh(dp_replicate_size=self.dp_replicate_size, device_type=self.device_type)

        # FSDP
        if self.tp_size > 1:
            return get_fsdp_tp_device_mesh(
                dp_replicate_size=self.dp_replicate_size,
                dp_shard_size=self.dp_shard_size,
                tp_size=self.tp_size,
                device_type=self.device_type,
            )

        return get_fsdp_device_mesh(
            dp_replicate_size=self.dp_replicate_size,
            dp_shard_size=self.dp_shard_size,
            device_type=self.device_type,
        )

    def set_device(self) -> torch.device:
        if not self.is_distributed:
            self._device = torch.device(self.device_type)
            return self._device
        self._device = set_device(local_rank=self.local_rank)
        return self._device

    @property
    def device(self) -> torch.device:
        if self._device is None:
            return self.set_device()
        return self._device

    def maybe_enable_amp_autocast(
        self, mixed_precision_param: TorchDtypeEnum, device_type: str = "cuda"
    ) -> contextlib.AbstractContextManager[None] | torch.autocast:
        # FSDP handles mixed precision internally
        if self.dp_backend == DPBackend.FSDP:
            logger.info("Mixed precision training is handled by fully_shard")
            return contextlib.nullcontext()

        # the following code will only be executed for DDP or single-device training
        # self.dp_backend in (DPBackend.DDP, DPBackend.NO):
        logger.info("Mixed precision training is handled by AMP")
        return torch.autocast(
            device_type,
            dtype=TORCH_DTYPE_MAP[mixed_precision_param],
        )
