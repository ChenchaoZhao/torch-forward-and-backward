import torch
import torch.distributed.tensor._random
import torch.distributed.tensor.parallel
from torch.distributed.device_mesh import DeviceMesh


def create_context_parallel_ctx(
    cp_mesh: DeviceMesh,
    cp_buffers: list[torch.Tensor],
    cp_seq_dims: list[int],
    cp_no_restore_buffers: set[torch.Tensor],
    cp_rotate_method: str,
):
    try:
        from torch.distributed.tensor.experimental import context_parallel
        from torch.distributed.tensor.experimental._attention import set_rotate_method
    except ImportError as e:
        msg = (
            f"PyTorch version {torch.__version__} does not include the experimental "
            "Context Parallel API. Please update to a newer version."
        )
        raise ValueError(msg) from e

    set_rotate_method(cp_rotate_method)
    return context_parallel(
        cp_mesh,
        buffers=cp_buffers,
        buffer_seq_dims=cp_seq_dims,
        no_restore_buffers=cp_no_restore_buffers,
    )
