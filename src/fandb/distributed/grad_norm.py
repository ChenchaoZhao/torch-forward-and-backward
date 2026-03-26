import math
from collections.abc import Iterable

import torch
from torch import distributed as dist
from torch.distributed.device_mesh import DeviceMesh
from torch.distributed.tensor import DTensor


@torch.no_grad()
def clip_grad_norm_(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float = 2.0,
    error_if_nonfinite: bool = False,
    foreach: bool | None = None,
    pp_mesh: DeviceMesh | None = None,
    ep_enabled: bool = False,
) -> torch.Tensor:
    """
    Clip the gradient norm of an iterable of parameters.

    Gradient norm clipping requires computing the gradient norm over the entire model.
    `torch.nn.utils.clip_grad_norm_` only computes gradient norm along DP/FSDP/TP dimensions.
    We need to manually reduce the gradient norm across PP stages.
    See https://github.com/pytorch/torchtitan/issues/596 for details.

    Args:
        parameters: an iterable of Tensors or a single Tensor that will have gradients normalized
        max_norm (float): max norm of the gradients
        norm_type (float): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.
        error_if_nonfinite (bool): if True, an error is thrown if the total
            norm of the gradients from :attr:`parameters` is ``nan``,
            ``inf``, or ``-inf``. Default: False (will switch to True in the future)
        foreach (bool): use the faster foreach-based implementation.
            If ``None``, use the foreach implementation for CUDA and CPU native tensors and silently
            fall back to the slow implementation for other device types.
            Default: ``None``
        pp_mesh: Pipeline Parallel device mesh. If not None, will reduce gradient norm across PP stages.
        ep_dense_params_mesh_ndim: Mesh ndim of the dense params when EP is used. If EP is not used,
            set it to ``None``.

    Returns:
        Total norm of the parameter gradients (viewed as a single vector).

    """
    if ep_enabled:
        return _clip_grad_norm_with_ep(
            parameters,
            max_norm,
            norm_type,
            error_if_nonfinite,
            foreach,
            pp_mesh,
        )

    parameters = (
        [parameters] if isinstance(parameters, torch.Tensor) else list(parameters)
    )  # prevent generators from being exhausted
    grads = [p.grad for p in parameters if p.grad is not None]
    total_norm = torch.nn.utils.get_total_norm(grads, norm_type, error_if_nonfinite, foreach)

    # If total_norm is a DTensor, the placements must be `torch.distributed._tensor.ops.math_ops._NormPartial`.
    # We can simply reduce the DTensor to get the total norm in this tensor's process group
    # and then convert it to a local tensor.
    # NOTE: It has two purposes:
    #       1. to make sure the total norm is computed correctly when PP is used (see below)
    #       2. to return a reduced total_norm tensor whose .item() would return the correct value
    if isinstance(total_norm, DTensor):
        # Will reach here if any non-PP parallelism is used.
        # If only using PP, total_norm will be a local tensor.
        total_norm = total_norm.full_tensor()

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach)
    return total_norm


@torch.no_grad()
def _clip_grad_norm_with_ep(
    parameters: torch.Tensor | Iterable[torch.Tensor],
    max_norm: float,
    norm_type: float,
    error_if_nonfinite: bool,
    foreach: bool | None,
    pp_mesh: DeviceMesh | None,
) -> torch.Tensor:
    ep_params = []
    non_ep_params = []
    ep_grads = []
    non_ep_grads = []

    for p in parameters:
        if p.grad is None:
            continue
        if not isinstance(p, DTensor):
            msg = f"Parameter must be a DTensor, got {type(p)}"
            raise TypeError(msg)
        if not isinstance(p.grad, DTensor):
            msg = f"Parameter gradient must be a DTensor, got {type(p.grad)}"
            raise TypeError(msg)
        mesh_dim_names = p.device_mesh.mesh_dim_names
        if mesh_dim_names is None:
            msg = "mesh_dim_names cannot be None"
            raise ValueError(msg)
        if "ep" in mesh_dim_names:
            ep_params.append(p)
            ep_grads.append(p.grad)
        else:
            non_ep_params.append(p)
            non_ep_grads.append(p.grad)

    # Either list can be empty depending on the parallelization strategy:
    # - In torchtitan with separate dense/sparse meshes, both lists are typically non-empty
    # - In autoparallel, all params may live on a single sparse mesh with "ep" dimension,
    #   so non_ep_grads would be empty
    # - In PP + EP setups, certain PP ranks may only own EP or non-EP layers
    ep_grads_total_norm = torch.nn.utils.get_total_norm(ep_grads, norm_type, error_if_nonfinite, foreach)
    # get_total_norm returns tensor(0.) for empty list, which is a non-DTensor
    if isinstance(ep_grads_total_norm, DTensor):
        ep_grads_total_norm = ep_grads_total_norm.full_tensor()

    non_ep_grads_total_norm = torch.nn.utils.get_total_norm(non_ep_grads, norm_type, error_if_nonfinite, foreach)
    # get_total_norm returns tensor(0.) for empty list, which is a non-DTensor
    if isinstance(non_ep_grads_total_norm, DTensor):
        non_ep_grads_total_norm = non_ep_grads_total_norm.full_tensor()

    if math.isinf(norm_type):
        total_norm = torch.maximum(ep_grads_total_norm, non_ep_grads_total_norm)
    else:
        total_norm = ep_grads_total_norm**norm_type + non_ep_grads_total_norm**norm_type
        total_norm **= 1.0 / norm_type

    if pp_mesh is not None:
        if math.isinf(norm_type):
            dist.all_reduce(total_norm, op=dist.ReduceOp.MAX, group=pp_mesh.get_group())
        else:
            total_norm **= norm_type
            dist.all_reduce(total_norm, op=dist.ReduceOp.SUM, group=pp_mesh.get_group())
            total_norm **= 1.0 / norm_type

    torch.nn.utils.clip_grads_with_norm_(ep_params, max_norm, total_norm, foreach)
    torch.nn.utils.clip_grads_with_norm_(non_ep_params, max_norm, total_norm, foreach)

    return total_norm
