from collections.abc import Callable

import torch
from accelerate.data_loader import (
    BatchSamplerShard,
    DataLoaderDispatcher,
    DataLoaderShard,
    IterableDatasetShard,
    RNGType,
    SeedableRandomSampler,
    get_sampler,
    is_torch_version,
)
from datasets import IterableDataset as DatasetsIterableDataset
from torch.distributed.device_mesh import DeviceMesh
from torch.utils.data import BatchSampler, DataLoader, IterableDataset, RandomSampler

# kwargs of the DataLoader in min version 2.0
_PYTORCH_DATALOADER_KWARGS = {
    "batch_size": 1,
    "shuffle": False,
    "sampler": None,
    "batch_sampler": None,
    "num_workers": 0,
    "collate_fn": None,
    "pin_memory": True,
    "drop_last": False,
    "timeout": 0,
    "worker_init_fn": None,
    "multiprocessing_context": None,
    "generator": None,
    "prefetch_factor": 2,
    "persistent_workers": True,
    "pin_memory_device": "",
}

# kwargs added after by version
_PYTORCH_DATALOADER_ADDITIONAL_KWARGS = {"2.6.0": {"in_order": True}}

for v, additional_kwargs in _PYTORCH_DATALOADER_ADDITIONAL_KWARGS.items():
    if is_torch_version(">=", v):
        _PYTORCH_DATALOADER_KWARGS.update(additional_kwargs)


# copied from accelerate.data_loader
def prepare_data_loader(
    dataloader: DataLoader,
    dp_world_size: int,
    dp_rank: int,
    device: torch.device | None = None,
    split_batches: bool = False,
    put_on_device: bool = False,
    rng_types: list[str | RNGType] | None = None,
    dispatch_batches: bool | None = None,
    even_batches: bool = True,
    slice_fn_for_dispatch: Callable | None = None,
    use_seedable_sampler: bool = False,
    data_seed: int | None = None,
    non_blocking: bool = True,
    use_stateful_dataloader: bool = False,
    torch_device_mesh: DeviceMesh | None = None,
) -> DataLoader:
    """
    Wraps a PyTorch `DataLoader` to generate batches for one of the processes only.

    Depending on the value of the `drop_last` attribute of the `dataloader` passed, it will either stop the iteration
    at the first batch that would be too small / not present on all processes or loop with indices from the beginning.

    Args:
        dataloader (`torch.utils.data.dataloader.DataLoader`):
            The data loader to split across several devices.
        dp_world_size (`int`):
            The number of processes running concurrently.
        dp_rank (`int`):
            The index of the current process.
        device (`torch.device`, *optional*, defaults to `None`):
            The target device for the returned `DataLoader`.
        split_batches (`bool`, *optional*, defaults to `False`):
            Whether the resulting `DataLoader` should split the batches of the original data loader across devices or
            yield full batches (in which case it will yield batches starting at the `dp_rank`-th and advancing of
            `dp_world_size` batches at each iteration).

            Another way to see this is that the observed batch size will be the same as the initial `dataloader` if
            this option is set to `True`, the batch size of the initial `dataloader` multiplied by `dp_world_size`
            otherwise.

            Setting this option to `True` requires that the batch size of the `dataloader` is a round multiple of
            `dp_world_size`.
        put_on_device (`bool`, *optional*, defaults to `False`):
            Whether or not to put the batches on `device` (only works if the batches are nested list, tuples or
            dictionaries of tensors).
        rng_types (`list[str | RNGType]`, *optional*, defaults to `None`):
            The list of random number generators to synchronize at the beginning of each iteration. Should be one or
            several of:

            - `"torch"`: the base torch random number generator
            - `"cuda"`: the CUDA random number generator (GPU only)
            - `"xla"`: the XLA random number generator (TPU only)
            - `"generator"`: the `torch.Generator` of the sampler (or batch sampler if there is no sampler in your
              dataloader) or of the iterable dataset (if it exists) if the underlying dataset is of that type.

        dispatch_batches (`bool`, *optional*, defaults to `None`):
            If set to `True`, the dataloader prepared is only iterated through on the main process and then the batches
            are split and broadcast to each process. Will default to `True` when the underlying dataset is an
            `IterableDataset`, `False` otherwise.
        even_batches (`bool`, *optional*, defaults to `True`):
            If set to `True`, in cases where the total batch size across all processes does not exactly divide the
            dataset, samples at the start of the dataset will be duplicated so the batch can be divided equally among
            all workers.
        slice_fn_for_dispatch (`Callable`, *optional*, defaults to `None`):
            If passed, this function will be used to slice tensors across `dp_world_size`. Will default to
            [`~utils.slice_tensors`]. This argument is used only when `dispatch_batches` is set to `True` and will be
            ignored otherwise.
        use_seedable_sampler (`bool`, *optional*, defaults to `False`):
            Whether to use the [`~data_loader.SeedableRandomSampler`] instead of a `RandomSampler` for better
            reproducibility. Comes at a cost of potentially different performances due to different shuffling
            algorithms but ensures results will be the *exact* same. Should be paired with `set_seed()` at every
            `self.set_epoch`
        data_seed (`int`, *optional*, defaults to `None`):
            The seed to use for the underlying generator when using `use_seedable_sampler`. If `None`, the generator
            will use the current default seed from torch.
        non_blocking (`bool`, *optional*, defaults to `False`):
            If set to `True`, dataloader will utilize non-blocking host-to-device transfers. If the dataloader has
            `pin_memory` set to `True`, this will help to increase overlap between data transfer and computations.
        use_stateful_dataloader (`bool`, *optional*, defaults to `False`):
            If set to `True`, the dataloader prepared by the Accelerator will be backed by
            [torchdata.StatefulDataLoader](https://github.com/pytorch/data/tree/main/torchdata/stateful_dataloader).
            This requires `torchdata` version 0.8.0 or higher that supports StatefulDataLoader to be installed.
        torch_device_mesh (`torch.distributed.DeviceMesh`, *optional*, defaults to `None`):
            PyTorch device mesh.

    Returns:
        `torch.utils.data.dataloader.DataLoader`: A new data loader that will yield the portion of the batches

    <Tip warning={true}>

    `BatchSampler`s with varying batch sizes are not enabled by default. To enable this behaviour, set `even_batches`
    equal to `False`

    </Tip>
    """
    dispatch_batches = False if not put_on_device else isinstance(dataloader.dataset, IterableDataset)

    if dispatch_batches and not put_on_device:
        raise ValueError("Using `dispatch_batches=True` requires `put_on_device=True`.")

    # Sanity check
    if split_batches:
        if dataloader.batch_size is not None:
            batch_size_for_check = dataloader.batch_size
        elif hasattr(dataloader.batch_sampler, "batch_size"):
            batch_size_for_check = dataloader.batch_sampler.batch_size  # type: ignore[union-attr]
        else:
            msg = (
                "In order to use `split_batches==True` you must have a `batch_size` attribute either in the passed "
                "`dataloader` or `dataloader.batch_sampler` objects, and it has to return a natural number. "
                "Your `dataloader.batch_size` is None and `dataloader.batch_sampler` "
                f"(`{type(dataloader.batch_sampler)}`) does not have the `batch_size` attribute set."
            )
            raise ValueError(msg)

        if batch_size_for_check > 1 and batch_size_for_check % dp_world_size != 0:
            msg = (
                f"To use a `DataLoader` in `split_batches` mode, the batch size ({dataloader.batch_size}) "
                f"needs to be a round multiple of the number of processes ({dp_world_size})."
            )
            raise ValueError(msg)

    new_dataset = dataloader.dataset
    # Iterable dataset doesn't like batch_sampler, but data_loader creates a default one for it
    new_batch_sampler = dataloader.batch_sampler if not isinstance(new_dataset, IterableDataset) else None
    sampler_is_batch_sampler = isinstance(dataloader.sampler, BatchSampler)
    synchronized_generator = None

    sampler = get_sampler(dataloader)
    if isinstance(sampler, RandomSampler) and use_seedable_sampler:
        # When iterating through the dataloader during distributed processes
        # we want to ensure that on each process we are iterating through the same
        # samples in the same order if a seed is set. This requires a tweak
        # to the `torch.utils.data.RandomSampler` class (if used).
        sampler = SeedableRandomSampler(
            data_source=sampler.data_source,
            replacement=sampler.replacement,
            num_samples=sampler._num_samples,  # noqa: SLF001
            generator=getattr(
                sampler,
                "generator",
                torch.Generator(device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"),
            ),
            data_seed=data_seed,
        )

    # No change if no multiprocess
    if dp_world_size != 1 and not dispatch_batches:
        if (
            isinstance(new_dataset, DatasetsIterableDataset)
            and not split_batches
            and new_dataset.n_shards > dp_world_size
        ):
            new_dataset = new_dataset.shard(num_shards=dp_world_size, index=dp_rank)
        elif isinstance(new_dataset, IterableDataset):
            if getattr(dataloader.dataset, "generator", None) is not None:
                synchronized_generator = dataloader.dataset.generator
            new_dataset = IterableDatasetShard(
                new_dataset,
                batch_size=dataloader.batch_size,
                drop_last=dataloader.drop_last,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                split_batches=split_batches,
            )
        else:
            if not use_seedable_sampler and hasattr(sampler, "generator"):
                if sampler.generator is None:
                    sampler.generator = torch.Generator(
                        device=torch.get_default_device() if hasattr(torch, "get_default_device") else "cpu"
                    )
                    seed = int(torch.empty((), dtype=torch.int64).random_().item())
                    sampler.generator.manual_seed(seed)
                synchronized_generator = sampler.generator
            batch_sampler = dataloader.sampler if sampler_is_batch_sampler else dataloader.batch_sampler
            new_batch_sampler = BatchSamplerShard(
                batch_sampler,
                dp_world_size=dp_world_size,
                dp_rank=dp_rank,
                split_batches=split_batches,
                even_batches=even_batches,
            )

    # We ignore all of those since they are all dealt with by our new_batch_sampler
    ignore_kwargs = [
        "batch_size",
        "shuffle",
        "sampler",
        "batch_sampler",
        "drop_last",
    ]

    if rng_types is not None and synchronized_generator is None and "generator" in rng_types:
        rng_types.remove("generator")

    kwargs = {
        k: getattr(dataloader, k, _PYTORCH_DATALOADER_KWARGS[k])
        for k in _PYTORCH_DATALOADER_KWARGS
        if k not in ignore_kwargs
    }

    # Need to provide batch_size as batch_sampler is None for Iterable dataset
    if new_batch_sampler is None:
        kwargs["drop_last"] = dataloader.drop_last
        kwargs["batch_size"] = (
            dataloader.batch_size // dp_world_size if split_batches and not dispatch_batches else dataloader.batch_size  # type: ignore[operator]
        )
    if dispatch_batches:
        kwargs.pop("generator")
        dataloader = DataLoaderDispatcher(
            new_dataset,
            split_batches=split_batches,
            batch_sampler=new_batch_sampler,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            slice_fn=slice_fn_for_dispatch,
            use_stateful_dataloader=use_stateful_dataloader,
            torch_device_mesh=torch_device_mesh,
            **kwargs,
        )
    elif sampler_is_batch_sampler:  # map-style data
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            sampler=new_batch_sampler,
            batch_size=dataloader.batch_size,
            rng_types=rng_types,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            synchronized_generator=synchronized_generator,
            use_stateful_dataloader=use_stateful_dataloader,
            **kwargs,
        )
    else:  # map-style data batch sampler
        dataloader = DataLoaderShard(
            new_dataset,
            device=device if put_on_device else None,
            batch_sampler=new_batch_sampler,
            rng_types=rng_types,
            synchronized_generator=synchronized_generator,
            _drop_last=dataloader.drop_last,
            _non_blocking=non_blocking,
            use_stateful_dataloader=use_stateful_dataloader,
            **kwargs,
        )

    if isinstance(sampler, SeedableRandomSampler) and use_seedable_sampler:
        dataloader.set_sampler(sampler)

    return dataloader
