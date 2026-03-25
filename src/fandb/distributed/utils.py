# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import os
from datetime import timedelta
from types import ModuleType

import torch
from loguru import logger
from torch._utils import _get_available_device_type, _get_device_module
from torch.distributed import _local_tensor


def get_device_info() -> tuple[str, ModuleType]:
    device_type = _get_available_device_type() or "cuda"
    device_module = _get_device_module(device_type)  # default device_module:torch.cuda
    return device_type, device_module


DEVICE_TYPE, DEVICE_MODULE = get_device_info()


def init_fake_mode(world_size: int, comm_mode: str = "fake_backend"):
    """Initialize fake backend

    Args:
        world_size: The number of GPUs to simulate
        comm_mode: Communication mode ("fake_backend" or "local_tensor")

    Returns:
        The world size
    """
    torch.distributed.init_process_group(
        "fake",
        rank=0,
        world_size=world_size,
    )

    # If local_tensor mode is enabled, initialize LocalTensorMode context
    if comm_mode == "local_tensor":
        lm = _local_tensor.LocalTensorMode(world_size)
        lm.__enter__()


def _warn_overwrite_env(env: str, val: str):
    if env in os.environ:
        logger.warning(f"ENV[{env}] = {os.environ[env]} will be overridden to {val} based on job config")
    os.environ[env] = val


def _get_distributed_backend(enable_cpu_backend: bool) -> str:
    backend = "nccl"
    if DEVICE_TYPE in torch.distributed.Backend.default_device_backend_map:
        backend = torch.distributed.Backend.default_device_backend_map.get(DEVICE_TYPE)
    if enable_cpu_backend:
        backend = f"{DEVICE_TYPE}:{backend},cpu:gloo"
    return backend


def set_device(local_rank: int) -> torch.device:
    # torch.cuda.device_count
    device_index = local_rank % DEVICE_MODULE.device_count()
    device = torch.device(DEVICE_TYPE, device_index)
    DEVICE_MODULE.set_device(device)
    # torch.cuda.set_device
    return device


def set_pg_timeout(timeout: timedelta):
    """
    Sets the timeout for all PGs in the provided mesh, and the default (world) group.

    Note: synchronizes via a barrier, before changing the timeouts. This is important, because
    otherwise you may face a race where the slow rank has not reached the timeout reduction point
    yet due to slow operations permitted under the old timeout value, but other faster ranks may
    start issuing collectives under the new shorter timeout and then immediately timeout.
    """
    logger.info(f"Synchronizing and adjusting timeout for all ProcessGroups to {timeout}")
    # Ensure that all the ranks have reached the point of setting the new timeout-
    # otherwise, some ranks may issue collectives with the new/shorter timeout and
    # those may time out, before other ranks have finished with initialization done
    # under the old/slow timeout.

    torch.distributed.barrier(device_ids=[DEVICE_MODULE.current_device()])
    # torch.cuda.current_device()
    DEVICE_MODULE.synchronize()
    # torch.cuda.synchronize()
    torch.distributed.distributed_c10d._set_pg_timeout(timeout, None)  # noqa:SLF001
