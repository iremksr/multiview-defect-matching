
import torch
import torch.nn as nn

from torchvision.models import ResNet18_Weights, resnet18
from torchvision.ops import FeaturePyramidNetwork


class MultiViewBackbone(nn.Module):
    def __init__(self, pretrained=True, freeze_backbone=True, out_channels=128):
        super().__init__()

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        resnet = resnet18(weights=weights)

        # RGB pretrained conv -> grayscale conv
        old_conv = resnet.conv1
        new_conv = nn.Conv2d(
            1, 64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        with torch.no_grad():
            new_conv.weight.copy_(
                old_conv.weight.mean(dim=1, keepdim=True) * 3
            )

        resnet.conv1 = new_conv

        self.stem = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool
        )

        self.layer1 = resnet.layer1  # stride 4, 64 channels
        self.layer2 = resnet.layer2  # stride 8, 128 channels
        self.layer3 = resnet.layer3  # stride 16, 256 channels

        self.freeze_backbone = freeze_backbone

        if freeze_backbone:
            for module in [self.stem, self.layer1, self.layer2, self.layer3]:
                for param in module.parameters():
                    param.requires_grad = False

        self.fpn = FeaturePyramidNetwork(
            in_channels_list=[64, 128, 256],
            out_channels=out_channels
        )

    def train(self, mode=True):
        super().train(mode)

        # Keep pretrained BatchNorm statistics fixed.
        if self.freeze_backbone:
            for module in [self.stem, self.layer1, self.layer2, self.layer3]:
                module.eval()

        return self

    def forward(self, images):
        """
        images: [B, V, 1, H, W]

        Returns:
            dict of feature maps:
            {
                "p2": [B, V, C, H/4, W/4],
                "p3": [B, V, C, H/8, W/8],
                "p4": [B, V, C, H/16, W/16]
            }
        """
        B, V, C, H, W = images.shape

        assert C == 1, "Expected grayscale images"

        x = images.reshape(B * V, C, H, W)

        # Backbone
        x = self.stem(x)

        c2 = self.layer1(x)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)

        features = self.fpn({
            "p2": c2,
            "p3": c3,
            "p4": c4
        })

        return {
            name: feat.reshape(B, V, *feat.shape[1:])
            for name, feat in features.items()
        }
