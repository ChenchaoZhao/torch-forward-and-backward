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


DEFAULT_LR: float = 1e-3
DEFAULT_BETA1: float = 0.9
DEFAULT_BETA2: float = 0.999
DEFAULT_EPS: float = 1e-8
DEFAULT_WEIGHT_DECAY: float = 0.01

OptimParamPolicy = Callable[[Module, "OptimizerConfig"], list[ParamGroup]]


_OPTIMIZER_CONFIG_DOC = f"""
    Configuration for optimizer settings.

    Attributes:
        name: Optimizer to use (default: ADAMW)
        lr: Learning rate to use (Defaults to {DEFAULT_LR})
        beta1: Exponential moving average hyperparameter (Defaults to {DEFAULT_BETA1})
        beta2: Exponential moving average hyperparameter (Defaults to {DEFAULT_BETA2})
            - torch beta2 default: 0.999
            - torchtitan beta2 default: 0.95
            Smaller beta2 is chosen when batch size is very large
        eps: Epsilon value to use (Defaults to {DEFAULT_EPS})
        weight_decay: Weight decay to use (Defaults to {DEFAULT_WEIGHT_DECAY})
            - torch default: 0.01
            - torchtitan default: 0.1
        implementation: Optimizer implementation to use (default: FUSED)
            - 'fused': Use fused implementation (CUDA only) for best performance.
            - 'foreach': Use some horizontal fusion of tensors for better performance.
            - 'for-loop': Use the default implementation for the optimizer (slowest).
            - more info: https://pytorch.org/docs/stable/optim.html
"""


@dataclass
class OptimizerConfig(ConfigMixin):
    __doc__ = _OPTIMIZER_CONFIG_DOC

    name: OptimizerNameEnum = OptimizerNameEnum.ADAMW
    lr: float = DEFAULT_LR
    beta1: float = DEFAULT_BETA1
    beta2: float = DEFAULT_BETA2
    eps: float = DEFAULT_EPS
    weight_decay: float = DEFAULT_WEIGHT_DECAY
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
