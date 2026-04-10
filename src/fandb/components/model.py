from dataclasses import dataclass

from torch.nn import Module

from fandb.components.configuration import ConfigMixin

SEP = ":"


@dataclass
class ModelConfig(ConfigMixin):
    """Configuration class for model settings.

    Attributes:
        model_name: The name of the model, formatted as "base:flavor".
                    Defaults to "default_model".
    """

    model_name: str = "default_model"

    def get_model(self) -> Module:
        """Get the PyTorch model instance.

        Returns:
            A PyTorch Module representing the configured model.

        Raises:
            NotImplementedError: This method must be implemented by subclasses.
        """
        raise NotImplementedError

    @property
    def model_base_name(self) -> str:
        """Extract the base name from the model name.

        The model name is expected to be in the format "base:flavor".
        This property returns the "base" part by splitting on the first colon.

        Returns:
            The base name portion of the model name.
        """
        return self.model_name.split(SEP, 1)[0]


def auto_model_name(base: str, flavor: str) -> str:
    """Generate a model name by combining base and flavor components.

    Args:
        base: The base name of the model.
        flavor: The flavor or variant of the model.

    Returns:
        A formatted model name string in the format "base:flavor",
        with any spaces replaced by underscores.
    """
    return f"{base}{SEP}{flavor}".replace(" ", "_")
