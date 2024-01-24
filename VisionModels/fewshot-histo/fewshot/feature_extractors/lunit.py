# code from https://github.com/lunit-io/benchmark-ssl-pathology

import torch
from torchvision.models.resnet import Bottleneck, ResNet
from timm.models.vision_transformer import VisionTransformer


class ResNetTrunk(ResNet):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        del self.fc  # remove FC layer

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return x


def get_pretrained_url(key):
    URL_PREFIX = "https://github.com/lunit-io/benchmark-ssl-pathology/releases/download/pretrained-weights"
    model_zoo_registry = {
        "BT": "bt_rn50_ep200.torch",
        "MoCoV2": "mocov2_rn50_ep200.torch",
        "SwAV": "swav_rn50_ep200.torch",
        "DINO_p16": "dino_vit_small_patch16_ep200.torch",
        "DINO_p8": "dino_vit_small_patch8_ep200.torch",
    }
    pretrained_url = f"{URL_PREFIX}/{model_zoo_registry.get(key)}"
    return pretrained_url


def resnet50(pretrained, progress, key, **kwargs):
    model = ResNetTrunk(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model


def vit_small(pretrained, progress, key, **kwargs):
    patch_size = kwargs.get("patch_size", None)
    patch_size = patch_size or (16 if "p16" in key else 8)
    model = VisionTransformer(img_size=224, patch_size=patch_size, embed_dim=384, num_heads=6, num_classes=0)
    if pretrained:
        pretrained_url = get_pretrained_url(key)
        verbose = model.load_state_dict(torch.hub.load_state_dict_from_url(pretrained_url, progress=progress))
        print(verbose)
    return model
