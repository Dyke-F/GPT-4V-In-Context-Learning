from torch import nn
import torch
from pathlib import Path
from typing import Union
from loguru import logger


import fewshot
from .resnet import resnet50
from ..utils import download_file


class RetCCL(nn.Module):
    def __init__(
        self, weights_path: Union[Path, str] = Path(fewshot.__file__).parent.parent / "weights" / "retccl.pth"
    ):
        super().__init__()

        download_file(
            weights_path,
            url="https://drive.google.com/u/0/uc?id=1EOqdXSkIHg2Pcl3P8S4SGB5elDylw8O2&export=download",
            checksum="931956f31d3f1a3f6047f3172b9e59ee3460d29f7c0c2bb219cbc8e9207795ff",
        )

        self.model = resnet50(num_classes=128, mlp=False, two_branch=False, normlinear=True)
        self.model.fc = nn.Identity()
        self.model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")), strict=True)

        self.model.eval()

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
