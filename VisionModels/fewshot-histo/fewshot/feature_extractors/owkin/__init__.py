import hashlib
from torch import nn
import torch
from pathlib import Path
import numpy as np
from typing import Literal, Union
from loguru import logger


import fewshot
from .ibot_vit import iBOTViT
from ..utils import download_file


class Owkin(nn.Module):
    def __init__(self, weights_path: Union[Path, str] = Path(fewshot.__file__).parent.parent / "weights" / "owkin.pth", encoder: Literal["teacher", "student"] = "teacher"):
        super().__init__()
        download_file(
            weights_path,
            url="https://drive.google.com/u/0/uc?id=1uxsoNVhQFoIDxb4RYIiOtk044s6TTQXY&export=download",
            checksum="3bc6e4e353ebdd75b31979ff470ffa4d67349828057957dcc8d0f13e9d224d3f",
        )
        self.model = iBOTViT(architecture="vit_base_pancan", encoder=encoder, weights_path=weights_path)

        self.model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
