import pytest
import torch
from torch import nn

from fandb.components.lr import (
    LrSchedulerConfig,
    SchedulerNameEnum,
)


class TestSchedulerNameEnum:
    """Test suite for SchedulerNameEnum."""

    def test_linear_value(self) -> None:
        """Test LINEAR enum value."""
        assert SchedulerNameEnum.LINEAR == "linear"

    def test_cosine_value(self) -> None:
        """Test COSINE enum value."""
        assert SchedulerNameEnum.COSINE == "cosine"

    def test_constant_value(self) -> None:
        """Test CONSTANT enum value."""
        assert SchedulerNameEnum.CONSTANT == "constant"


class TestLrSchedulerConfigDefaults:
    """Test suite for LrSchedulerConfig default values."""

    def test_default_warmup_fraction(self) -> None:
        """Test default warmup fraction is 0.2."""
        config = LrSchedulerConfig()
        assert config.warmup_fraction == 0.2

    def test_default_name(self) -> None:
        """Test default scheduler name is LINEAR."""
        config = LrSchedulerConfig()
        assert config.name == SchedulerNameEnum.LINEAR


class TestLrSchedulerConfigPostInit:
    """Test suite for LrSchedulerConfig __post_init__ conversion."""

    def test_string_name_conversion(self) -> None:
        """Test string to enum conversion for name."""
        config = LrSchedulerConfig(name="cosine")
        assert config.name == SchedulerNameEnum.COSINE

    def test_int_warmup_fraction_conversion(self) -> None:
        """Test integer to float conversion for warmup_fraction."""
        config = LrSchedulerConfig(warmup_fraction=1)
        assert config.warmup_fraction == 1.0
        assert isinstance(config.warmup_fraction, float)

    def test_string_warmup_fraction_conversion(self) -> None:
        """Test string to float conversion for warmup_fraction."""
        config = LrSchedulerConfig(warmup_fraction="0.5")
        assert config.warmup_fraction == 0.5
        assert isinstance(config.warmup_fraction, float)


class TestLrSchedulerConfigValidation:
    """Test suite for LrSchedulerConfig validation."""

    def test_warmup_fraction_above_one_raises(self) -> None:
        """Test that warmup_fraction > 1.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"warmup_fraction must be between 0\.0 and 1\.0"):
            LrSchedulerConfig(warmup_fraction=1.5)

    def test_warmup_fraction_negative_raises(self) -> None:
        """Test that warmup_fraction < 0.0 raises ValueError."""
        with pytest.raises(ValueError, match=r"warmup_fraction must be between 0\.0 and 1\.0"):
            LrSchedulerConfig(warmup_fraction=-0.1)

    def test_warmup_fraction_boundary_zero(self) -> None:
        """Test that warmup_fraction = 0.0 is valid."""
        config = LrSchedulerConfig(warmup_fraction=0.0)
        assert config.warmup_fraction == 0.0

    def test_warmup_fraction_boundary_one(self) -> None:
        """Test that warmup_fraction = 1.0 is valid."""
        config = LrSchedulerConfig(warmup_fraction=1.0)
        assert config.warmup_fraction == 1.0


class TestLrSchedulerConfigNumTrainingSteps:
    """Test suite for LrSchedulerConfig num_training_steps."""

    def test_set_num_training_steps(self) -> None:
        """Test setting num_training_steps."""
        config = LrSchedulerConfig()
        config.set_num_training_steps(1000)
        assert config.num_training_steps == 1000

    def test_set_num_training_steps_float_conversion(self) -> None:
        """Test that float input is converted to int."""
        config = LrSchedulerConfig()
        config.set_num_training_steps(1000.0)
        assert config.num_training_steps == 1000
        assert isinstance(config.num_training_steps, int)

    def test_set_num_training_steps_zero(self) -> None:
        """Test setting num_training_steps to 0."""
        config = LrSchedulerConfig()
        config.set_num_training_steps(0)
        assert config.num_training_steps == 0

    def test_set_num_training_steps_negative_raises(self) -> None:
        """Test that negative num_training_steps raises ValueError."""
        config = LrSchedulerConfig()
        with pytest.raises(ValueError, match="num_training_steps should be >= 0"):
            config.set_num_training_steps(-1)

    def test_num_training_steps_not_set_raises(self) -> None:
        """Test that accessing num_training_steps before setting raises RuntimeError."""
        config = LrSchedulerConfig()
        with pytest.raises(RuntimeError, match="num_training_steps has not been set"):
            _ = config.num_training_steps


class TestLrSchedulerConfigNumWarmupSteps:
    """Test suite for LrSchedulerConfig num_warmup_steps."""

    def test_num_warmup_steps_calculation(self) -> None:
        """Test num_warmup_steps is calculated correctly."""
        config = LrSchedulerConfig(warmup_fraction=0.1)
        config.set_num_training_steps(1000)
        assert config.num_warmup_steps == 100

    def test_num_warmup_steps_fraction_zero(self) -> None:
        """Test num_warmup_steps with warmup_fraction=0."""
        config = LrSchedulerConfig(warmup_fraction=0.0)
        config.set_num_training_steps(1000)
        assert config.num_warmup_steps == 0

    def test_num_warmup_steps_fraction_one(self) -> None:
        """Test num_warmup_steps with warmup_fraction=1."""
        config = LrSchedulerConfig(warmup_fraction=1.0)
        config.set_num_training_steps(1000)
        assert config.num_warmup_steps == 1000


class TestLrSchedulerConfigGetLrScheduler:
    """Test suite for LrSchedulerConfig.get_lr_scheduler method."""

    def test_get_lr_scheduler_linear(self) -> None:
        """Test creating a LINEAR scheduler."""
        config = LrSchedulerConfig(name=SchedulerNameEnum.LINEAR, warmup_fraction=0.1)
        config.set_num_training_steps(1000)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = config.get_lr_scheduler(optimizer)
        assert scheduler is not None

    def test_get_lr_scheduler_cosine(self) -> None:
        """Test creating a COSINE scheduler."""
        config = LrSchedulerConfig(name=SchedulerNameEnum.COSINE, warmup_fraction=0.1)
        config.set_num_training_steps(1000)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = config.get_lr_scheduler(optimizer)
        assert scheduler is not None

    def test_get_lr_scheduler_constant(self) -> None:
        """Test creating a CONSTANT scheduler."""
        config = LrSchedulerConfig(name=SchedulerNameEnum.CONSTANT, warmup_fraction=0.1)
        config.set_num_training_steps(1000)
        model = nn.Linear(10, 10)
        optimizer = torch.optim.AdamW(model.parameters())
        scheduler = config.get_lr_scheduler(optimizer)
        assert scheduler is not None


class TestLrSchedulerConfigSerialization:
    """Test suite for LrSchedulerConfig serialization via ConfigMixin."""

    def test_to_dict(self) -> None:
        """Test serializing to dict."""
        config = LrSchedulerConfig(warmup_fraction=0.3, name=SchedulerNameEnum.COSINE)
        result = config.to_dict()
        assert result["warmup_fraction"] == 0.3
        assert result["name"] == "cosine"

    def test_from_dict(self) -> None:
        """Test deserializing from dict."""
        data = {"warmup_fraction": 0.5, "name": "cosine"}
        config = LrSchedulerConfig.from_dict(data)
        assert config.warmup_fraction == 0.5
        assert config.name == SchedulerNameEnum.COSINE

    def test_from_dict_string_warmup(self) -> None:
        """Test from_dict with string warmup_fraction."""
        data = {"warmup_fraction": "0.25", "name": "linear"}
        config = LrSchedulerConfig.from_dict(data)
        assert config.warmup_fraction == 0.25
