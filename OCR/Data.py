from enum import Enum
from typing import List
from dataclasses import dataclass


class Labelformat(Enum):
    t_unicode = 0
    wylie = 1


class TargetEncoding(Enum):
    stacks = 0
    wyile = 1


@dataclass
class OCRModelConfig:
    model_file: str
    onnx_model: str
    architecture: str
    input_width: int
    input_height: int
    input_layer: str
    output_layer: str
    squeeze_channel: bool
    swap_hw: bool
    encoder: str
    charset: List[str]
    add_blank: bool