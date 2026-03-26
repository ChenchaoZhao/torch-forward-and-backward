from enum import StrEnum, auto

from torch.distributed.device_mesh import DeviceMesh, init_device_mesh


class DeviceTypeEnum(StrEnum):
    CPU = auto()
    CUDA = auto()


class DPBackend(StrEnum):
    NO = auto()
    DDP = auto()
    FSDP = auto()


class ParallelDimEnum(StrEnum):
    DATA_PARALLEL_REPLICATE = "dp_replicate"
    DATA_PARALLEL_SHARD = "dp_shard"
    TENSOR_PARALLEL = "tp"
    CONTEXT_PARALLEL = "cp"
    PIPELINE_PARALLEL = "pp"
    EXPERT_PARALLEL = "ep"
    EXPERT_TENSOR_PARALLEL = "etp"
    EXPERT_FSDP = "efsdp"


def get_ddp_device_mesh(dp_replicate_size: int, device_type: DeviceTypeEnum) -> DeviceMesh:
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=(dp_replicate_size,),
        mesh_dim_names=(ParallelDimEnum.DATA_PARALLEL_REPLICATE,),
    )


def get_fsdp_device_mesh(dp_replicate_size: int, dp_shard_size: int, device_type: DeviceTypeEnum) -> DeviceMesh:
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=(
            dp_replicate_size,
            dp_shard_size,
        ),
        mesh_dim_names=(
            ParallelDimEnum.DATA_PARALLEL_REPLICATE,
            ParallelDimEnum.DATA_PARALLEL_SHARD,
        ),
    )


def get_fsdp_tp_device_mesh(
    dp_replicate_size: int, dp_shard_size: int, tp_size: int, device_type: DeviceTypeEnum
) -> DeviceMesh:
    return init_device_mesh(
        device_type=device_type,
        mesh_shape=(dp_replicate_size, dp_shard_size, tp_size),
        mesh_dim_names=(
            ParallelDimEnum.DATA_PARALLEL_REPLICATE,
            ParallelDimEnum.DATA_PARALLEL_SHARD,
            ParallelDimEnum.TENSOR_PARALLEL,
        ),
    )
