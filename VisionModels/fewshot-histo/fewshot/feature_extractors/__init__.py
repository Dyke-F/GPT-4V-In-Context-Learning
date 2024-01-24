from torchvision.models import resnet50, ResNet50_Weights, swin_t, Swin_T_Weights, vit_b_16, ViT_B_16_Weights
from torch import nn
import torch
from torchvision import transforms as T
from typing import Callable, Optional
import timm

from .ctranspath import CTransPath
from .retccl import RetCCL
from .owkin import Owkin
from .lunit import resnet50 as lunit_resnet50, vit_small as lunit_vit_small
from ..utils.images import IMAGENET_MEAN, IMAGENET_STD, LUNIT_MEAN, LUNIT_STD

__all__ = [
    "load_feature_extractor",
    "FEATURE_EXTRACTORS",
    "FeatureExtractor",
]


class ResNet50(nn.Module):
    """ResNet50 feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2 if pretrained else None)
        self.model.fc = nn.Identity()

    def forward(self, x):
        return self.model(x)


class SwinTransformer(nn.Module):
    """SwinTransformer feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = swin_t(weights=Swin_T_Weights.IMAGENET1K_V1 if pretrained else None)
        self.model.head = nn.Identity()

    def forward(self, x):
        return self.model(x)


class ViTB(nn.Module):
    """ViT-B feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)

    def forward(self, x):
        # https://stackoverflow.com/a/75875049
        feats = self.model._process_input(x)

        # Expand the class token to the full batch
        batch_class_token = self.model.class_token.expand(x.shape[0], -1, -1)
        feats = torch.cat([batch_class_token, feats], dim=1)

        feats = self.model.encoder(feats)

        # We're only interested in the representation of the classifier token that we appended at position 0
        feats = feats[:, 0]

        return feats


class ViTS(nn.Module):
    """ViT-S feature extractor."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model("vit_small_patch16_224.augreg_in1k", pretrained=pretrained, num_classes=0).eval()

    def forward(self, x):
        return self.model.forward(x)


class FeatureExtractor(nn.Module):
    def __init__(self, model: nn.Module, transform: Callable[[torch.Tensor], torch.Tensor], name: Optional[str] = None):
        super().__init__()
        model.eval()
        self.model = model
        self.transform = transform
        self.name = name or model.__class__.__name__

    def forward(self, x):
        return self.model(self.transform(x))


_imagenet_transform = T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
_lunit_transform = T.Normalize(mean=LUNIT_MEAN, std=LUNIT_STD)
_vits_transform = T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])

FEATURE_EXTRACTORS = {
    "ctranspath": lambda: FeatureExtractor(CTransPath(), name="ctranspath", transform=_imagenet_transform),
    "retccl": lambda: FeatureExtractor(RetCCL(), name="retccl", transform=_imagenet_transform),
    "resnet50": lambda: FeatureExtractor(ResNet50(), name="resnet50", transform=_imagenet_transform),
    "swin": lambda: FeatureExtractor(SwinTransformer(), name="swin", transform=_imagenet_transform),
    "phikon": lambda: FeatureExtractor(Owkin(encoder="teacher"), name="phikon", transform=_imagenet_transform),
    "phikon_student": lambda: FeatureExtractor(Owkin(encoder="student"), name="phikon_student", transform=_imagenet_transform),
    "vit": lambda: FeatureExtractor(ViTB(), name="vit", transform=_imagenet_transform),
    "bt": lambda: FeatureExtractor(
        lunit_resnet50(key="BT", pretrained=True, progress=True), name="bt", transform=_lunit_transform
    ),
    "mocov2": lambda: FeatureExtractor(
        lunit_resnet50(key="MoCoV2", pretrained=True, progress=True), name="mocov2", transform=_lunit_transform
    ),
    "swav": lambda: FeatureExtractor(
        lunit_resnet50(key="SwAV", pretrained=True, progress=True), name="swav", transform=_lunit_transform
    ),
    "dino_p16": lambda: FeatureExtractor(
        lunit_vit_small(key="DINO_p16", pretrained=True, progress=True), name="dino_p16", transform=_lunit_transform
    ),
    "dino_p8": lambda: FeatureExtractor(
        lunit_vit_small(key="DINO_p8", pretrained=True, progress=True), name="dino_p8", transform=_lunit_transform
    ),
    "vits": lambda: FeatureExtractor(ViTS(), name="vits", transform=_vits_transform),
}


def load_feature_extractor(model_name: str) -> FeatureExtractor:
    try:
        return FEATURE_EXTRACTORS[model_name]()
    except KeyError:
        raise ValueError(f"Unknown feature extractor model {model_name}.")
