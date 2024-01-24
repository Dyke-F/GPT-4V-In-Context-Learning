import hashlib
from torch import nn
import torch
from pathlib import Path
import numpy as np
from typing import Literal, Union
from loguru import logger


import fewshot
from .swin_transformer import swin_tiny_patch4_window7_224, ConvStem
from ..utils import download_file


class CTransPath(nn.Module):
    def __init__(
        self, weights_path: Union[Path, str] = Path(fewshot.__file__).parent.parent / "weights" / "ctranspath.pth"
    ):
        super().__init__()
        download_file(
            weights_path,
            url="https://drive.google.com/u/0/uc?id=1DoDx_70_TLj98gTf6YTXnu4tFhsFocDX&export=download",
            checksum="7c998680060c8743551a412583fac689db43cec07053b72dfec6dcd810113539",
        )
        self.model = swin_tiny_patch4_window7_224(embed_layer=ConvStem, pretrained=False)
        self.model.head = nn.Identity()
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu"))["model"], strict=True)

        self.model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
