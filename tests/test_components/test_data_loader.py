import warnings

from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from fandb.components.data_loader import (
    DEFAULT_BATCH_SIZE,
    DEFAULT_NUM_WORKERS,
    DEFAULT_PERSISTENT_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_SHUFFLE,
    DEFAULT_TIMEOUT,
    DataLoaderConfig,
)


class SimpleDataset(Dataset):
    def __init__(self, size: int = 10):
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        return Tensor([idx]), Tensor([idx * 2])


class TestDataLoaderConfigDefaults:
    def test_default_batch_size(self) -> None:
        assert DataLoaderConfig().batch_size == DEFAULT_BATCH_SIZE

    def test_default_shuffle(self) -> None:
        assert DataLoaderConfig().shuffle == DEFAULT_SHUFFLE

    def test_default_num_workers(self) -> None:
        assert DataLoaderConfig().num_workers == DEFAULT_NUM_WORKERS

    def test_default_persistent_workers(self) -> None:
        assert DataLoaderConfig().persistent_workers == DEFAULT_PERSISTENT_WORKERS

    def test_default_pin_memory(self) -> None:
        assert DataLoaderConfig().pin_memory == DEFAULT_PIN_MEMORY

    def test_default_prefetch_factor(self) -> None:
        assert DataLoaderConfig().prefetch_factor == DEFAULT_PREFETCH_FACTOR

    def test_default_timeout(self) -> None:
        assert DataLoaderConfig().timeout == DEFAULT_TIMEOUT


class TestDataLoaderConfigInit:
    def test_custom_values(self) -> None:
        config = DataLoaderConfig(
            batch_size=32,
            shuffle=True,
            num_workers=4,
            persistent_workers=False,
            pin_memory=False,
            prefetch_factor=3,
            timeout=30.0,
        )
        assert config.batch_size == 32
        assert config.shuffle is True
        assert config.num_workers == 4
        assert config.persistent_workers is False
        assert config.pin_memory is False
        assert config.prefetch_factor == 3
        assert config.timeout == 30.0

    def test_type_conversion_batch_size(self) -> None:
        config = DataLoaderConfig(batch_size="16")
        assert config.batch_size == 16
        assert isinstance(config.batch_size, int)

    def test_type_conversion_num_workers(self) -> None:
        config = DataLoaderConfig(num_workers="4")
        assert config.num_workers == 4
        assert isinstance(config.num_workers, int)

    def test_type_conversion_timeout(self) -> None:
        config = DataLoaderConfig(timeout="10.5")
        assert config.timeout == 10.5
        assert isinstance(config.timeout, float)


class TestDataLoaderConfigNumWorkersZero:
    def test_zero_workers_warns(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = DataLoaderConfig(num_workers=0, prefetch_factor=2, persistent_workers=True)
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "prefetch_factor and persistent_workers" in str(w[0].message)

    def test_zero_workers_sets_prefetch_none(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = DataLoaderConfig(num_workers=0, prefetch_factor=2)
            assert config.prefetch_factor is None

    def test_zero_workers_sets_persistent_false(self) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            config = DataLoaderConfig(num_workers=0, persistent_workers=True)
            assert config.persistent_workers is False

    def test_zero_workers_no_warning_when_already_none(self) -> None:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            config = DataLoaderConfig(num_workers=0, prefetch_factor=None, persistent_workers=False)
            assert len(w) == 0


class TestDataLoaderConfigToDict:
    def test_to_dict_defaults(self) -> None:
        config = DataLoaderConfig()
        result = config.to_dict()
        assert result == {
            "batch_size": DEFAULT_BATCH_SIZE,
            "shuffle": DEFAULT_SHUFFLE,
            "num_workers": DEFAULT_NUM_WORKERS,
            "persistent_workers": DEFAULT_PERSISTENT_WORKERS,
            "pin_memory": DEFAULT_PIN_MEMORY,
            "prefetch_factor": DEFAULT_PREFETCH_FACTOR,
            "timeout": DEFAULT_TIMEOUT,
        }

    def test_to_dict_custom(self) -> None:
        config = DataLoaderConfig(batch_size=64, shuffle=True, num_workers=2)
        result = config.to_dict()
        assert result["batch_size"] == 64
        assert result["shuffle"] is True
        assert result["num_workers"] == 2


class TestDataLoaderConfigFromDict:
    def test_from_dict(self) -> None:
        data = {"batch_size": 128, "num_workers": 4, "shuffle": True}
        config = DataLoaderConfig.from_dict(data)
        assert config.batch_size == 128
        assert config.num_workers == 4
        assert config.shuffle is True


class TestDataLoaderConfigGetDataLoader:
    def test_create_basic_dataloader(self) -> None:
        config = DataLoaderConfig(batch_size=2, num_workers=0)
        dataset = SimpleDataset(size=4)
        loader = config.get_data_loader(dataset)
        assert isinstance(loader, DataLoader)
        assert loader.batch_size == 2
        assert loader.dataset is dataset

    def test_get_data_loader_with_sampler(self) -> None:
        from torch.utils.data import SequentialSampler

        config = DataLoaderConfig(batch_size=2, num_workers=0)
        dataset = SimpleDataset(size=4)
        sampler = SequentialSampler(dataset)
        loader = config.get_data_loader(dataset, sampler=sampler)
        assert isinstance(loader, DataLoader)
        assert loader.sampler is not None

    def test_get_data_loader_with_batch_sampler(self) -> None:
        from torch.utils.data import BatchSampler, SequentialSampler

        config = DataLoaderConfig(num_workers=0)
        dataset = SimpleDataset(size=4)
        batch_sampler = BatchSampler(SequentialSampler(dataset), batch_size=2, drop_last=False)
        loader = config.get_data_loader(dataset, batch_sampler=batch_sampler)
        assert isinstance(loader, DataLoader)
        assert loader.batch_sampler is batch_sampler

    def test_get_data_loader_iteration(self) -> None:
        config = DataLoaderConfig(batch_size=2, num_workers=0)
        dataset = SimpleDataset(size=4)
        loader = config.get_data_loader(dataset)
        batches = list(loader)
        assert len(batches) == 2
        inputs, _ = batches[0]
        assert inputs.shape[0] == 2

    def test_zero_workers_removes_prefetch_and_persistent(self) -> None:
        config = DataLoaderConfig(num_workers=0, prefetch_factor=2, persistent_workers=True)
        dataset = SimpleDataset()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            loader = config.get_data_loader(dataset)
            assert loader.prefetch_factor is None
            assert loader.persistent_workers is False

    def test_non_zero_workers_keeps_prefetch_and_persistent(self) -> None:
        config = DataLoaderConfig(num_workers=2, prefetch_factor=2, persistent_workers=True)
        dataset = SimpleDataset()
        loader = config.get_data_loader(dataset)
        assert loader.prefetch_factor == 2
        assert loader.persistent_workers is True
