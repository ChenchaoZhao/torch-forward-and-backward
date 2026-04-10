from enum import StrEnum, auto

import torch


class TorchDtypeEnum(StrEnum):
    FP16 = auto()
    FP32 = auto()
    BF16 = auto()


TORCH_DTYPE_MAP = {
    TorchDtypeEnum.FP16: torch.float16,
    TorchDtypeEnum.FP32: torch.float32,
    TorchDtypeEnum.BF16: torch.bfloat16,
}
