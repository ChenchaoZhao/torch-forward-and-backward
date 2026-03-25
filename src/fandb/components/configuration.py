from dataclasses import asdict, dataclass
from typing import Any, Self


@dataclass
class ConfigMixin:
    """
    A mixin class for dataclasses that provides dictionary serialization and
    deserialization.

    This mixin provides utility methods for converting dataclass instances to
    and from dictionaries, enabling easy configuration serialization,
    JSON/YAML integration, and data persistence workflows.

    The class leverages the standard library's `dataclasses.asdict()` function
    and the `dacite` library's `from_dict()` function to handle complex nested
    structures, type conversion, and validation.

    Usage:
        @dataclass
        class MyConfig(ConfigMixin):
            name: str
            port: int
            debug: bool = False

        # Create instance
        config = MyConfig(name="app", port=8080)

        # Convert to dictionary
        config_dict = config.to_dict()
        # {'name': 'app', 'port': 8080, 'debug': False}

        # Create from dictionary
        new_config = MyConfig.from_dict({
            'name': 'new_app',
            'port': 9000,
            'debug': True
        })

    Note:
        This mixin is designed to work with dataclasses and requires the
        `dacite` library for dictionary deserialization. The class being
        mixed in must be decorated with @dataclass.
    """

    def to_dict(self) -> dict[str, Any]:
        """
        Convert the dataclass instance to a dictionary.

        Returns:
            dict: A dictionary representation of the dataclass instance with
                  field names as keys and field values as values. Nested
                  dataclasses are recursively converted to dictionaries.

        Example:
            >>> @dataclass
            >>> class Config(ConfigMixin):
            ...     name: str
            ...     port: int
            >>> config = Config(name="test", port=8080)
            >>> config.to_dict()
            {'name': 'test', 'port': 8080}
        """
        return asdict(self)

    def to_dot_path_dict(self, dot_char: str = ".") -> dict[str, Any]:
        """
        Convert the dataclass instance to a dictionary with dot-notation keys.

        This method flattens the dataclass instance by converting nested
        structures into a flat dictionary where keys use dot notation to
        represent the hierarchical structure.

        Args:
            dot_char: The character to use for separating path components
                      (default: ".")

        Returns:
            dict: A flat dictionary with dot-notation keys representing
                  the nested structure of the dataclass instance.

        Example:
            >>> @dataclass
            >>> class Config(ConfigMixin):
            ...     database: dict = field(
            ...         default_factory=lambda: {"host": "localhost", "port": 5432}
            ...     )
            ...     debug: bool = False
            >>> config = Config()
            >>> config.to_dot_path_dict()
            {'database.host': 'localhost', 'database.port': 5432,
             'debug': False}
        """
        return nested_dict_to_dot_path_dict(self.to_dict(), dot_char=dot_char)

    @classmethod
    def from_dict(cls, nested_dict: dict[str, Any]) -> Self:
        """
        Create a dataclass instance from a dictionary.

        Args:
            nested_dict: A dictionary containing the field names and values
                        to create the dataclass instance. The dictionary can
                        contain nested dictionaries which will be converted
                        to nested dataclass instances.

        Returns:
            Self: An instance of the dataclass created from the dictionary data.

        Raises:
            TypeError: If the dictionary contains invalid data types that
                      cannot be converted to the expected field types.
            ValueError: If required fields are missing from the dictionary.
            DaciteError: If the dacite library encounters conversion issues.

        Example:
            >>> @dataclass
            >>> class Config(ConfigMixin):
            ...     name: str
            ...     port: int
            ...     debug: bool = False
            >>> data = {"name": "app", "port": 8080, "debug": True}
            >>> config = Config.from_dict(data)
            >>> config.name
            'app'
            >>> config.port
            8080
        """
        from dacite import Config as DaciteConfig
        from dacite import from_dict

        return from_dict(data_class=cls, data=nested_dict, config=DaciteConfig(check_types=False))

    @classmethod
    def from_dot_path_dict(cls, dot_path_dict: dict[str, Any], dot_char: str = ".") -> Self:
        """
        Create a dataclass instance from a dictionary with dot-notation
        keys.

        This method reconstructs a dataclass instance from a flat
        dictionary where keys use dot notation to represent hierarchical
        structure. It first converts the flat dictionary to a nested
        structure, then creates the dataclass instance.

        Args:
            dot_path_dict: A dictionary with dot-notation keys
                          (e.g., 'database.host': 'localhost')
            dot_char: The character used for separating path components
                      (default: ".")

        Returns:
            Self: An instance of the dataclass created from the dot-notation
                  dictionary data.

        Raises:
            TypeError: If the dictionary contains invalid data types that
                      cannot be converted to the expected field types.
            ValueError: If required fields are missing from the dictionary.
            DaciteError: If the dacite library encounters conversion issues.

        Example:
            >>> @dataclass
            >>> class Config(ConfigMixin):
            ...     database: dict
            ...     debug: bool = False
            >>> data = {"database.host": "localhost", "database.port": 5432, "debug": True}
            >>> config = Config.from_dot_path_dict(data)
            >>> config.database
            {'host': 'localhost', 'port': 5432}
            >>> config.debug
            True
        """
        return cls.from_dict(dot_path_dict_to_nested_dict(dot_path_dict, dot_char=dot_char))


def dot_path_dict_to_nested_dict(dot_path_dict: dict[str, Any], dot_char: str = ".") -> dict[str, Any]:
    """
    Convert a dictionary with dot-notation keys to a nested dictionary.

    Args:
        dot_path_dict: Dictionary with keys containing dots
                      (e.g., 'key.subkey')

    Returns:
        Nested dictionary structure

    Example:
        Input: {'a.b.c': 1, 'a.b.d': 2, 'e': 3}
        Output: {'a': {'b': {'c': 1, 'd': 2}}, 'e': 3}
    """
    if not dot_path_dict:
        return {}

    result: dict[str, Any] = {}

    for key, value in dot_path_dict.items():
        # Split key into parts
        parts = key.split(dot_char)

        # Navigate/create nested structure
        current = result
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            elif not isinstance(current[part], dict):
                # Convert non-dict value to dict when needed for nesting
                current[part] = {}
            current = current[part]

        # Set the final value
        final_key = parts[-1]
        current[final_key] = value

    return result


# flatten dict for logging
def nested_dict_to_dot_path_dict(nested_dict: dict[str, Any], dot_char: str = ".") -> dict[str, Any]:
    """
    Convert a nested dictionary to a dictionary with dot-notation keys.

    This is the reverse operation of dot_path_dict_to_nested_dict.
    It flattens a nested dictionary structure by concatenating the keys
    with dots.

    Args:
        nested_dict: The nested dictionary to flatten
        dot_char: The character to use for separating path components
                  (default: ".")

    Returns:
        Dictionary with dot-notation keys

    Example:
        Input: {'a': {'b': {'c': 1, 'd': 2}}, 'e': 3}
        Output: {'a.b.c': 1, 'a.b.d': 2, 'e': 3}
    """
    if not nested_dict:
        return {}

    result: dict[str, Any] = {}

    def _flatten(obj: Any, prefix: str = "") -> None:
        if isinstance(obj, dict):
            for key, value in obj.items():
                new_key = f"{prefix}{dot_char}{key}" if prefix else key
                # Only recurse if dict is non-empty
                if isinstance(value, dict) and value:
                    _flatten(value, new_key)
                else:
                    result[new_key] = value
        else:
            # If obj is not a dict, store it directly
            result[prefix] = obj

    _flatten(nested_dict)
    return result


def merge_nested_dicts(nested_dict_1: dict[str, Any], nested_dict_2: dict[str, Any]) -> dict[str, Any]:
    """
    Merge two nested dictionaries, with values from nested_dict_2 taking
    precedence.

    This function performs a deep merge of two nested dictionaries. When both
    dictionaries contain dictionaries at the same key, they are merged
    recursively. Otherwise, values from nested_dict_2 will overwrite values
    from nested_dict_1.

    Args:
        nested_dict_1: First dictionary (lower precedence)
        nested_dict_2: Second dictionary (higher precedence)

    Returns:
        A new merged dictionary

    Example:
        Input:
            nested_dict_1 = {'a': {'b': 1, 'c': 2}, 'd': 4}
            nested_dict_2 = {'a': {'b': 10, 'e': 5}, 'f': 6}
        Output: {'a': {'b': 10, 'c': 2, 'e': 5}, 'd': 4, 'f': 6}
    """
    if not nested_dict_1:
        return nested_dict_2.copy() if nested_dict_2 else {}
    if not nested_dict_2:
        return nested_dict_1.copy()

    result = nested_dict_1.copy()

    for key, value in nested_dict_2.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            # Both values are dicts, merge them recursively
            result[key] = merge_nested_dicts(result[key], value)
        else:
            # Either key doesn't exist in result, or one of the values is
            # not a dict. In either case, value from nested_dict_2 takes
            # precedence
            result[key] = value

    return result
