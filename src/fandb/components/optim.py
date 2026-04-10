from collections.abc import Callable, Iterable
from dataclasses import dataclass
from enum import StrEnum, auto
from typing import Any, NotRequired, TypedDict

import torch
from torch.nn import Module, Parameter
from torch.optim import Adam, AdamW, Optimizer
from torch.optim.optimizer import ParamsT

from fandb.components.configuration import ConfigMixin


class OptimizerNameEnum(StrEnum):
    ADAMW = auto()
    ADAM = auto()


class OptimizerImplementation(StrEnum):
    FUSED = auto()
    FOREACH = auto()
    FORLOOP = auto()


class ParamGroup(TypedDict):
    params: torch.Tensor | Iterable[torch.Tensor]
    lr: NotRequired[float]
    weight_decay: NotRequired[float]


OptimParamPolicy = Callable[[Module, "OptimizerConfig"], list[ParamGroup]]


@dataclass
class OptimizerConfig(ConfigMixin):
    """
    Configuration for optimizer settings.

    Attributes:
        name: Optimizer to use (default: ADAMW)
        lr: Learning rate to use (default: 8e-4)
        beta1: Exponential moving average hyperparameter (default: 0.9)
        beta2: Exponential moving average hyperparameter (default: 0.999)
            - torch beta2 default: 0.999
            - torchtitan beta2 default: 0.95
            Smaller beta2 is chosen when batch size is very large
        eps: Epsilon value to use (default: 1e-8)
        weight_decay: Weight decay to use (default: 0.1)
            - torch default: 0.01
            - torchtitan default: 0.1
        implementation: Optimizer implementation to use (default: FUSED)
            - 'fused': Use fused implementation (CUDA only) for best performance.
            - 'foreach': Use some horizontal fusion of tensors for better performance.
            - 'for-loop': Use the default implementation for the optimizer (slowest).
            - more info: https://pytorch.org/docs/stable/optim.html
    """

    name: OptimizerNameEnum = OptimizerNameEnum.ADAMW
    lr: float = 8e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    weight_decay: float = 0.1
    implementation: OptimizerImplementation = OptimizerImplementation.FUSED

    def __post_init__(self):
        self.name = OptimizerNameEnum(self.name)
        self.implementation = OptimizerImplementation(self.implementation)

    def _get_kwargs(self) -> dict[str, Any]:
        return {
            "lr": self.lr,
            "betas": (self.beta1, self.beta2),
            "eps": self.eps,
            "weight_decay": self.weight_decay,
            "foreach": self.implementation == OptimizerImplementation.FOREACH,
            "fused": self.implementation == OptimizerImplementation.FUSED,
        }

    def get_optimizer(self, params: ParamsT) -> Optimizer:
        kwargs = self._get_kwargs()

        if self.name == OptimizerNameEnum.ADAMW:
            return AdamW(params=params, **kwargs)
        if self.name == OptimizerNameEnum.ADAM:
            return Adam(params=params, **kwargs)

        raise NotImplementedError

    def get_optimizer_with_param_policy(self, model: Module, param_policy: OptimParamPolicy | None = None) -> Optimizer:
        if not param_policy:
            param_policy = default_param_policy
        # in optimizer, param group will use default if key missing
        # param_group_dict.setdefault(**optim_kwargs)
        return self.get_optimizer(params=param_policy(model, self))


def default_param_policy(model: Module, optim_config: OptimizerConfig) -> list[ParamGroup]:
    decay_params: list[Parameter] = []
    no_decay_params: list[Parameter] = []
    for name, param in model.named_parameters():
        # normalization layer or bias terms
        if "norm" in name.lower() or "bias" in name.lower():
            no_decay_params.append(param)
        else:
            decay_params.append(param)

    return [
        ParamGroup(params=no_decay_params, weight_decay=0.0),
        ParamGroup(params=decay_params, weight_decay=optim_config.weight_decay),
    ]
