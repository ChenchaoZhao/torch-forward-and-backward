import torch
from torch import nn

from fandb.components.optim import (
    OptimizerConfig,
    OptimizerImplementation,
    OptimizerNameEnum,
    default_param_policy,
)


class TestOptimizerNameEnum:
    """Test suite for OptimizerNameEnum."""

    def test_adamw_value(self) -> None:
        """Test ADAMW enum value."""
        assert OptimizerNameEnum.ADAMW == "adamw"

    def test_adam_value(self) -> None:
        """Test ADAM enum value."""
        assert OptimizerNameEnum.ADAM == "adam"


class TestOptimizerImplementation:
    """Test suite for OptimizerImplementation."""

    def test_fused_value(self) -> None:
        """Test FUSED enum value."""
        assert OptimizerImplementation.FUSED == "fused"

    def test_foreach_value(self) -> None:
        """Test FOREACH enum value."""
        assert OptimizerImplementation.FOREACH == "foreach"

    def test_forloop_value(self) -> None:
        """Test FORLOOP enum value."""
        assert OptimizerImplementation.FORLOOP == "forloop"


class TestOptimizerConfigDefaults:
    """Test suite for OptimizerConfig default values."""

    def test_default_name(self) -> None:
        """Test default optimizer name is ADAMW."""
        config = OptimizerConfig()
        assert config.name == OptimizerNameEnum.ADAMW

    def test_default_lr(self) -> None:
        """Test default learning rate."""
        config = OptimizerConfig()
        assert config.lr == 8e-4

    def test_default_beta1(self) -> None:
        """Test default beta1 value."""
        config = OptimizerConfig()
        assert config.beta1 == 0.9

    def test_default_beta2(self) -> None:
        """Test default beta2 value."""
        config = OptimizerConfig()
        assert config.beta2 == 0.999

    def test_default_eps(self) -> None:
        """Test default epsilon value."""
        config = OptimizerConfig()
        assert config.eps == 1e-8

    def test_default_weight_decay(self) -> None:
        """Test default weight decay value."""
        config = OptimizerConfig()
        assert config.weight_decay == 0.1

    def test_default_implementation(self) -> None:
        """Test default implementation is FUSED."""
        config = OptimizerConfig()
        assert config.implementation == OptimizerImplementation.FUSED


class TestOptimizerConfigPostInit:
    """Test suite for OptimizerConfig __post_init__ conversion."""

    def test_string_name_conversion(self) -> None:
        """Test string to enum conversion for name."""
        config = OptimizerConfig(name="adam")
        assert config.name == OptimizerNameEnum.ADAM

    def test_string_implementation_conversion(self) -> None:
        """Test string to enum conversion for implementation."""
        config = OptimizerConfig(implementation="foreach")
        assert config.implementation == OptimizerImplementation.FOREACH


class TestOptimizerConfigGetKwargs:
    """Test suite for OptimizerConfig._get_kwargs method."""

    def test_get_kwargs_adamw_fused(self) -> None:
        """Test _get_kwargs for ADAMW with FUSED implementation."""
        config = OptimizerConfig(
            name=OptimizerNameEnum.ADAMW,
            lr=1e-3,
            beta1=0.9,
            beta2=0.99,
            eps=1e-6,
            weight_decay=0.01,
            implementation=OptimizerImplementation.FUSED,
        )
        kwargs = config._get_kwargs()
        assert kwargs == {
            "lr": 1e-3,
            "betas": (0.9, 0.99),
            "eps": 1e-6,
            "weight_decay": 0.01,
            "foreach": False,
            "fused": True,
        }

    def test_get_kwargs_adam_foreach(self) -> None:
        """Test _get_kwargs for ADAM with FOREACH implementation."""
        config = OptimizerConfig(
            name=OptimizerNameEnum.ADAM,
            implementation=OptimizerImplementation.FOREACH,
        )
        kwargs = config._get_kwargs()
        assert kwargs["foreach"] is True
        assert kwargs["fused"] is False

    def test_get_kwargs_forloop(self) -> None:
        """Test _get_kwargs with FORLOOP implementation."""
        config = OptimizerConfig(implementation=OptimizerImplementation.FORLOOP)
        kwargs = config._get_kwargs()
        assert kwargs["foreach"] is False
        assert kwargs["fused"] is False


class TestOptimizerConfigGetOptimizer:
    """Test suite for OptimizerConfig.get_optimizer method."""

    def test_get_optimizer_adamw(self) -> None:
        """Test creating ADAMW optimizer."""
        config = OptimizerConfig(name=OptimizerNameEnum.ADAMW)
        params = [torch.randn(3, 3, requires_grad=True)]
        optimizer = config.get_optimizer(params)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_get_optimizer_adam(self) -> None:
        """Test creating Adam optimizer."""
        config = OptimizerConfig(name=OptimizerNameEnum.ADAM)
        params = [torch.randn(3, 3, requires_grad=True)]
        optimizer = config.get_optimizer(params)
        assert isinstance(optimizer, torch.optim.Adam)

    def test_get_optimizer_params_passed(self) -> None:
        """Test optimizer receives correct parameters."""
        config = OptimizerConfig(lr=1e-3, weight_decay=0.05)
        param = torch.randn(5, 5, requires_grad=True)
        optimizer = config.get_optimizer([param])
        assert len(optimizer.param_groups) == 1
        assert optimizer.param_groups[0]["lr"] == 1e-3
        assert optimizer.param_groups[0]["weight_decay"] == 0.05


class TestOptimizerConfigGetOptimizerWithParamPolicy:
    """Test suite for OptimizerConfig.get_optimizer_with_param_policy method."""

    def test_get_optimizer_with_default_policy(self) -> None:
        """Test optimizer with default param policy."""
        config = OptimizerConfig()
        model = nn.Linear(10, 10)
        optimizer = config.get_optimizer_with_param_policy(model)
        assert isinstance(optimizer, torch.optim.AdamW)

    def test_get_optimizer_with_custom_policy(self) -> None:
        """Test optimizer with custom param policy."""
        config = OptimizerConfig()

        def custom_policy(model: nn.Module, optim_config: OptimizerConfig):
            return [{"params": list(model.parameters()), "lr": 1e-2}]

        model = nn.Linear(10, 10)
        optimizer = config.get_optimizer_with_param_policy(model, custom_policy)
        assert optimizer.param_groups[0]["lr"] == 1e-2


class TestDefaultParamPolicy:
    """Test suite for default_param_policy function."""

    def test_separates_norm_params(self) -> None:
        """Test that norm layer parameters are separated."""
        model = nn.Sequential(nn.LayerNorm(10), nn.Linear(10, 10))
        config = OptimizerConfig()
        groups = default_param_policy(model, config)

        assert len(groups) == 2
        no_decay_group = groups[0]
        decay_group = groups[1]
        assert no_decay_group["weight_decay"] == 0.0
        assert decay_group["weight_decay"] == config.weight_decay

    def test_separates_bias_params(self) -> None:
        """Test that bias parameters are separated."""
        model = nn.Linear(10, 10)
        config = OptimizerConfig()
        groups = default_param_policy(model, config)

        assert len(groups) == 2
        bias_params = groups[0]["params"]
        has_bias = any("bias" in name for name, _ in model.named_parameters())
        assert has_bias

    def test_weight_decay_assignment(self) -> None:
        """Test correct weight decay values in param groups."""
        custom_config = OptimizerConfig(weight_decay=0.05)
        model = nn.Linear(10, 10)
        groups = default_param_policy(model, custom_config)

        assert groups[0]["weight_decay"] == 0.0
        assert groups[1]["weight_decay"] == 0.05

    def test_empty_model(self) -> None:
        """Test with model that has no parameters."""
        config = OptimizerConfig()

        class EmptyModel(nn.Module):
            pass

        model = EmptyModel()
        groups = default_param_policy(model, config)
        assert len(groups) == 2
        assert len(groups[0]["params"]) == 0
        assert len(groups[1]["params"]) == 0

    def test_mixed_param_types(self) -> None:
        """Test model with bias, norm, and linear weights."""
        model = nn.Sequential(nn.BatchNorm1d(10), nn.Linear(10, 10))
        config = OptimizerConfig()
        groups = default_param_policy(model, config)

        assert len(groups) == 2
        no_decay_params = set(groups[0]["params"])
        decay_params = set(groups[1]["params"])
        all_params = {p for _, p in model.named_parameters()}

        assert len(no_decay_params & decay_params) == 0
        assert no_decay_params | decay_params == all_params


class TestOptimizerConfigSerialization:
    """Test suite for OptimizerConfig serialization via ConfigMixin."""

    def test_to_dict(self) -> None:
        """Test serializing to dict."""
        config = OptimizerConfig(lr=1e-3, weight_decay=0.05)
        result = config.to_dict()
        assert result["lr"] == 1e-3
        assert result["weight_decay"] == 0.05
        assert result["name"] == "adamw"
        assert result["implementation"] == "fused"

    def test_from_dict(self) -> None:
        """Test deserializing from dict."""
        data = {"lr": 5e-4, "weight_decay": 0.02, "name": "adam"}
        config = OptimizerConfig.from_dict(data)
        assert config.lr == 5e-4
        assert config.weight_decay == 0.02
        assert config.name == OptimizerNameEnum.ADAM
