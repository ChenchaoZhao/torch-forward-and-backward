import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

from torch import Generator
from torch.utils.data import DataLoader, Dataset, Sampler

from fandb.components.configuration import ConfigMixin

# Default constants for DataLoaderConfig
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_SHUFFLE: bool = False
DEFAULT_NUM_WORKERS: int = 8
DEFAULT_PERSISTENT_WORKERS: bool = True
DEFAULT_PIN_MEMORY: bool = True
DEFAULT_PREFETCH_FACTOR: int | None = 2
DEFAULT_TIMEOUT: float = 0

_worker_init_fn_t = Callable[[int], None]
_collate_fn_t = Callable[[list], Any]


_DATA_LOADER_CONFIG_DOC = f"""
    Configuration for PyTorch DataLoader settings.

    Attributes:
        batch_size: Number of samples per batch (Defaults to {DEFAULT_BATCH_SIZE})
        shuffle: Whether to shuffle data (Defaults to {DEFAULT_SHUFFLE})
        num_workers: Number of worker processes (Defaults to {DEFAULT_NUM_WORKERS})
        persistent_workers: Keep workers alive between epochs (Defaults to {DEFAULT_PERSISTENT_WORKERS})
            Only valid when num_workers > 0.
        pin_memory: Copy tensors to CUDA pinned memory (Defaults to {DEFAULT_PIN_MEMORY})
        prefetch_factor: Number of batches to prefetch per worker (Defaults to {DEFAULT_PREFETCH_FACTOR})
            Only valid when num_workers > 0.
        timeout: Timeout for collecting batches in seconds (Defaults to {DEFAULT_TIMEOUT})
            Zero means wait indefinitely.
"""


@dataclass
class DataLoaderConfig(ConfigMixin):
    __doc__ = _DATA_LOADER_CONFIG_DOC

    batch_size: int = DEFAULT_BATCH_SIZE
    shuffle: bool = DEFAULT_SHUFFLE
    num_workers: int = DEFAULT_NUM_WORKERS
    persistent_workers: bool = DEFAULT_PERSISTENT_WORKERS
    pin_memory: bool = DEFAULT_PIN_MEMORY
    prefetch_factor: int | None = DEFAULT_PREFETCH_FACTOR
    timeout: float = DEFAULT_TIMEOUT

    def __post_init__(self):
        self.batch_size = int(self.batch_size)
        self.num_workers = int(self.num_workers)
        self.timeout = float(self.timeout)

        if self.num_workers == 0:
            if self.prefetch_factor is not None or self.persistent_workers:
                warnings.warn(
                    "When num_workers=0, prefetch_factor and "
                    "persistent_workers will be ignored. "
                    "Setting them to None/False.",
                    UserWarning,
                    stacklevel=2,
                )
            self.prefetch_factor = None
            self.persistent_workers = False

    def get_data_loader(
        self,
        dataset: Dataset,
        sampler: Sampler | Iterable | None = None,
        batch_sampler: Sampler[list] | Iterable[list] | None = None,
        collate_fn: _collate_fn_t | None = None,
        worker_init_fn: _worker_init_fn_t | None = None,
        multiprocessing_context: str | None = None,
        generator: Generator | None = None,
    ) -> DataLoader:
        """Create a PyTorch DataLoader with this config's settings.

        Args:
            dataset: The dataset to load from.
            sampler: Strategy for sampling from dataset.
            batch_sampler: Strategy for yielding batches of indices.
            collate_fn: Function to collate samples into a batch.
            worker_init_fn: Function to initialize each worker.
            multiprocessing_context: Multiprocessing context to use.
            generator: RNG for reproducibility.

        Returns:
            Configured DataLoader instance.
        """
        kwargs = self.to_dict()

        # Remove prefetch_factor and persistent_workers when num_workers=0
        # as they are not valid PyTorch DataLoader parameters in this case
        if self.num_workers == 0:
            kwargs.pop("prefetch_factor", None)
            kwargs.pop("persistent_workers", None)

        return DataLoader(
            dataset=dataset,
            sampler=sampler,
            batch_sampler=batch_sampler,
            collate_fn=collate_fn,
            worker_init_fn=worker_init_fn,
            multiprocessing_context=multiprocessing_context,
            generator=generator,
            **kwargs,
        )
