import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Literal

import torch
from loguru import logger

from fandb.components.configuration import ConfigMixin
from fandb.distributed.utils import (
    _get_distributed_backend,
    _warn_overwrite_env,
    init_fake_mode,
    set_pg_timeout,
)

TRACE_BUFFER_SIZE: str = "TORCH_FR_BUFFER_SIZE"
TRACE_FILE: str = "TORCH_FR_DUMP_TEMP_FILE"
DUMP_ON_TIMEOUT: str = "TORCH_NCCL_DUMP_ON_TIMEOUT"
ASYNC_ERROR_HANDLING: str = "TORCH_NCCL_ASYNC_ERROR_HANDLING"
SKIP_CLEANUP: str = "3"

# Default timeout and buffer size constants
DEFAULT_INIT_TIMEOUT_SECONDS: int = 300
DEFAULT_TRAIN_TIMEOUT_SECONDS: int = 100
DEFAULT_TRACE_BUF_SIZE: int = 20_000


@dataclass
class CommConfig(ConfigMixin):
    """Communication configuration for distributed training.

    Attributes:
        init_timeout_seconds: Timeout for communication operations during initialization and first train step
        train_timeout_seconds: Timeout for communication operations after the first train step
        trace_buf_size: Flight recorder ring buffer size, >0 means recording by default, 0 means disabled
        save_traces_folder: Flight recorder trace files location
        save_traces_file_prefix: Flight recorder trace files prefix
        mode: Communication mode for distributed training
        enable_cpu_backend: Whether to enable CPU backend for communication
    """

    init_timeout_seconds: int = DEFAULT_INIT_TIMEOUT_SECONDS
    train_timeout_seconds: int = DEFAULT_TRAIN_TIMEOUT_SECONDS
    trace_buf_size: int = DEFAULT_TRACE_BUF_SIZE
    save_traces_folder: str = "comm_traces"
    save_traces_file_prefix: str = "rank_"
    mode: Literal["default", "fake_backend", "local_tensor"] = "default"
    enable_cpu_backend: bool = True

    def get_distributed_backend(self) -> str:
        return _get_distributed_backend(enable_cpu_backend=self.enable_cpu_backend)

    def init_distributed(
        self,
        base_folder: str = "",
        ranks: list[int] | None = None,
    ) -> None:
        # based on https://github.com/pytorch/torchtitan/blob/55658e93f12157d09a28f49397b474671815424b/torchtitan/distributed/utils.py#L318
        # Skip initialization if already initialized
        if torch.distributed.is_initialized():
            logger.warning(
                "torch.distributed is already initialized. Skipping init_distributed. "
                "The provided self and other settings will not take effect."
            )
            return

        if self.mode in ("fake_backend", "local_tensor"):
            ngpu_str = os.environ.get("NGPU")
            if ngpu_str is None:
                msg = f"NGPU environment variable must be set when using comm_mode={self.mode}"
                raise ValueError(msg)
            try:
                world_size = int(ngpu_str)
            except ValueError as e:
                msg = f"NGPU environment variable must be a valid integer, got: {ngpu_str}"
                raise ValueError(msg) from e
            init_fake_mode(world_size, self.mode)
            return

        # FlightRecorder is incompatible with =1 mode where watchdog aborts work, must use =3 (skipcleanup)
        # to get flight recorder dumps. See https://github.com/pytorch/pytorch/issues/121055
        # This could be done only when flight recorder is enabled, but its nice to be consistent to avoid subtle
        # behavior differences
        _warn_overwrite_env(ASYNC_ERROR_HANDLING, SKIP_CLEANUP)

        # enable torch nccl flight recorder in the mode that would dump files if timeout is detected
        _warn_overwrite_env(TRACE_BUFFER_SIZE, str(self.trace_buf_size))
        if self.trace_buf_size > 0:
            # dump on timeout by default if trace buffer is enabled
            _warn_overwrite_env(DUMP_ON_TIMEOUT, "1")
            dump_dir = os.path.join(base_folder, self.save_traces_folder)
            prefix = self.save_traces_file_prefix
            os.makedirs(dump_dir, exist_ok=True)
            _warn_overwrite_env(TRACE_FILE, f"{dump_dir}/{prefix}")

        # everything depends on env://
        # MASTER_ADDR, MASTER_PORT, RANK, WORLD_SIZE
        torch.distributed.init_process_group(
            backend=self.get_distributed_backend(),
            timeout=timedelta(seconds=self.init_timeout_seconds),
            _ranks=ranks if ranks is not None else [],
        )

    def set_train_timeout(self):
        set_pg_timeout(timeout=timedelta(seconds=self.train_timeout_seconds))
