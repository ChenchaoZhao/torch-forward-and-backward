from dataclasses import dataclass

import pytest
from torch.nn import Linear, Module

from fandb.components.model import SEP, ModelConfig, auto_model_name


@dataclass
class ConcreteModelConfig(ModelConfig):
    def get_model(self) -> Module:
        return Linear(10, 5)


class TestModelConfig:
    """Test suite for ModelConfig."""

    def test_default_model_name(self) -> None:
        """Test default model name."""
        config = ModelConfig()
        assert config.model_name == "default_model"

    def test_custom_model_name(self) -> None:
        """Test custom model name."""
        config = ModelConfig(model_name="my_model")
        assert config.model_name == "my_model"

    def test_model_base_name_simple(self) -> None:
        """Test model_base_name with simple name."""
        config = ModelConfig(model_name="bert")
        assert config.model_base_name == "bert"

    def test_model_base_name_with_flavor(self) -> None:
        """Test model_base_name with dotted name."""
        config = ModelConfig(model_name=f"bert{SEP}base")
        assert config.model_base_name == "bert"

    def test_model_base_name_multiple_dots(self) -> None:
        """Test model_base_name with multiple dots."""
        config = ModelConfig(model_name=f"a{SEP}b{SEP}c{SEP}d")
        assert config.model_base_name == "a"

    def test_get_model_raises_not_implemented(self) -> None:
        """Test that get_model raises NotImplementedError."""
        config = ModelConfig()
        with pytest.raises(NotImplementedError):
            config.get_model()

    def test_concrete_get_model(self) -> None:
        """Test that concrete implementation returns a Module."""
        config = ConcreteModelConfig(model_name="linear")
        model = config.get_model()
        assert isinstance(model, Module)


class TestAutoModelName:
    """Test suite for auto_model_name function."""

    def test_basic_concatenation(self) -> None:
        """Test basic base and flavor concatenation."""
        result = auto_model_name("bert", "base")
        assert result == f"bert{SEP}base"

    def test_spaces_replaced_with_underscore(self) -> None:
        """Test that spaces in flavor are replaced with underscores."""
        result = auto_model_name("model", "large version")
        assert result == f"model{SEP}large_version"

    def test_spaces_in_base(self) -> None:
        """Test behavior with spaces in base."""
        result = auto_model_name("model name", "flavor")
        assert result == f"model_name{SEP}flavor"

    def test_empty_strings(self) -> None:
        """Test with empty strings."""
        result = auto_model_name("", "")
        assert result == f"{SEP}"

    def test_single_part(self) -> None:
        """Test when one part is empty."""
        result = auto_model_name("bert", "")
        assert result == f"bert{SEP}"
        result = auto_model_name("", "base")
        assert result == f"{SEP}base"
