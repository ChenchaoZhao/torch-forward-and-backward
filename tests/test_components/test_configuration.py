from dataclasses import dataclass, field

from fandb.components.configuration import (
    ConfigMixin,
    dot_path_dict_to_nested_dict,
    merge_nested_dicts,
    nested_dict_to_dot_path_dict,
)


@dataclass
class SampleConfig(ConfigMixin):
    """Test configuration class for testing ConfigMixin functionality."""

    name: str
    port: int
    debug: bool = False
    database: dict = field(default_factory=lambda: {"host": "localhost", "port": 5432})


class TestConfigMixin:
    """Test suite for ConfigMixin functionality."""

    def test_to_dict_basic(self) -> None:
        """Test basic to_dict conversion."""
        # Arrange
        config = SampleConfig(name="test_app", port=8080, debug=True)

        # Act
        result = config.to_dict()

        # Assert
        expected = {"name": "test_app", "port": 8080, "debug": True, "database": {"host": "localhost", "port": 5432}}
        assert result == expected

    def test_to_dict_with_defaults(self) -> None:
        """Test to_dict with default values."""
        # Arrange
        config = SampleConfig(name="app", port=8080)

        # Act
        result = config.to_dict()

        # Assert
        expected = {"name": "app", "port": 8080, "debug": False, "database": {"host": "localhost", "port": 5432}}
        assert result == expected

    def test_from_dict_basic(self) -> None:
        """Test basic from_dict creation."""
        # Arrange
        data = {"name": "new_app", "port": 9000, "debug": True, "database": {"host": "remote", "port": 3306}}

        # Act
        config = SampleConfig.from_dict(data)

        # Assert
        assert config.name == "new_app"
        assert config.port == 9000
        assert config.debug is True
        assert config.database == {"host": "remote", "port": 3306}

    def test_from_dict_with_defaults(self) -> None:
        """Test from_dict with missing optional fields."""
        # Arrange
        data = {"name": "minimal_app", "port": 8080}

        # Act
        config = SampleConfig.from_dict(data)

        # Assert
        assert config.name == "minimal_app"
        assert config.port == 8080
        assert config.debug is False  # Should use default
        assert config.database == {"host": "localhost", "port": 5432}  # Should use default

    def test_to_dot_path_dict_simple(self) -> None:
        """Test to_dot_path_dict with simple structure."""
        # Arrange
        config = SampleConfig(name="app", port=8080)

        # Act
        result = config.to_dot_path_dict()

        # Assert
        expected = {"name": "app", "port": 8080, "debug": False, "database.host": "localhost", "database.port": 5432}
        assert result == expected

    def test_to_dot_path_dict_custom_separator(self) -> None:
        """Test to_dot_path_dict with custom separator."""
        # Arrange
        config = SampleConfig(name="app", port=8080)

        # Act
        result = config.to_dot_path_dict(dot_char="_")

        # Assert
        expected = {"name": "app", "port": 8080, "debug": False, "database_host": "localhost", "database_port": 5432}
        assert result == expected

    def test_from_dot_path_dict_simple(self) -> None:
        """Test from_dot_path_dict with simple structure."""
        # Arrange
        data = {"name": "dot_app", "port": 7000, "debug": True, "database.host": "remote", "database.port": 3306}

        # Act
        config = SampleConfig.from_dot_path_dict(data)

        # Assert
        assert config.name == "dot_app"
        assert config.port == 7000
        assert config.debug is True
        assert config.database == {"host": "remote", "port": 3306}

    def test_from_dot_path_dict_custom_separator(self) -> None:
        """Test from_dot_path_dict with custom separator."""
        # Arrange
        data = {"name": "custom_app", "port": 6000, "database_host": "custom_host", "database_port": 5433}

        # Act
        config = SampleConfig.from_dot_path_dict(data, dot_char="_")

        # Assert
        assert config.name == "custom_app"
        assert config.port == 6000
        assert config.database == {"host": "custom_host", "port": 5433}


class TestHelperFunctions:
    """Test suite for helper functions."""

    def test_dot_path_dict_to_nested_dict_simple(self) -> None:
        """Test dot_path_dict_to_nested_dict with simple structure."""
        # Arrange
        input_data = {"a.b.c": 1, "a.b.d": 2, "e": 3}

        # Act
        result = dot_path_dict_to_nested_dict(input_data)

        # Assert
        expected = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}
        assert result == expected

    def test_dot_path_dict_to_nested_dict_empty(self) -> None:
        """Test dot_path_dict_to_nested_dict with empty input."""
        # Arrange
        # Act
        result = dot_path_dict_to_nested_dict({})

        # Assert
        assert result == {}

    def test_dot_path_dict_to_nested_dict_deep_nesting(self) -> None:
        """Test dot_path_dict_to_nested_dict with deep nesting."""
        # Arrange
        input_data = {"a.b.c.d.e": "deep_value", "a.b.c.d.f": "another_deep", "x.y": "shallow"}

        # Act
        result = dot_path_dict_to_nested_dict(input_data)

        # Assert
        expected = {"a": {"b": {"c": {"d": {"e": "deep_value", "f": "another_deep"}}}}, "x": {"y": "shallow"}}
        assert result == expected

    def test_nested_dict_to_dot_path_dict_simple(self) -> None:
        """Test nested_dict_to_dot_path_dict with simple structure."""
        # Arrange
        input_data = {"a": {"b": {"c": 1, "d": 2}}, "e": 3}

        # Act
        result = nested_dict_to_dot_path_dict(input_data)

        # Assert
        expected = {"a.b.c": 1, "a.b.d": 2, "e": 3}
        assert result == expected

    def test_nested_dict_to_dot_path_dict_empty(self) -> None:
        """Test nested_dict_to_dot_path_dict with empty input."""
        # Arrange
        # Act
        result = nested_dict_to_dot_path_dict({})

        # Assert
        assert result == {}

    def test_nested_dict_to_dot_path_dict_custom_separator(self) -> None:
        """Test nested_dict_to_dot_path_dict with custom separator."""
        # Arrange
        input_data = {"database": {"host": "localhost", "port": 5432}, "app": {"name": "test"}}

        # Act
        result = nested_dict_to_dot_path_dict(input_data, dot_char="_")

        # Assert
        expected = {"database_host": "localhost", "database_port": 5432, "app_name": "test"}
        assert result == expected

    def test_merge_nested_dicts_simple(self) -> None:
        """Test merge_nested_dicts with simple structures."""
        # Arrange
        dict1 = {"a": {"b": 1, "c": 2}, "d": 4}
        dict2 = {"a": {"b": 10, "e": 5}, "f": 6}

        # Act
        result = merge_nested_dicts(dict1, dict2)

        # Assert
        expected = {"a": {"b": 10, "c": 2, "e": 5}, "d": 4, "f": 6}
        assert result == expected

    def test_merge_nested_dicts_empty_first(self) -> None:
        """Test merge_nested_dicts with empty first dict."""
        # Arrange
        dict1 = {}
        dict2 = {"a": 1, "b": {"c": 2}}

        # Act
        result = merge_nested_dicts(dict1, dict2)

        # Assert
        assert result == {"a": 1, "b": {"c": 2}}

    def test_merge_nested_dicts_empty_second(self) -> None:
        """Test merge_nested_dicts with empty second dict."""
        # Arrange
        dict1 = {"a": 1, "b": {"c": 2}}
        dict2 = {}

        # Act
        result = merge_nested_dicts(dict1, dict2)

        # Assert
        assert result == {"a": 1, "b": {"c": 2}}

    def test_merge_nested_dicts_both_empty(self) -> None:
        """Test merge_nested_dicts with both empty dicts."""
        # Arrange
        # Act
        result = merge_nested_dicts({}, {})

        # Assert
        assert result == {}

    def test_merge_nested_dicts_deep_merge(self) -> None:
        """Test merge_nested_dicts with deep nested structures."""
        # Arrange
        dict1 = {"level1": {"level2": {"level3": {"value1": "original", "value2": "keep"}}}}
        dict2 = {"level1": {"level2": {"level3": {"value1": "updated", "value3": "new"}, "new_level": "added"}}}

        # Act
        result = merge_nested_dicts(dict1, dict2)

        # Assert
        expected = {
            "level1": {
                "level2": {"level3": {"value1": "updated", "value2": "keep", "value3": "new"}, "new_level": "added"}
            }
        }
        assert result == expected
