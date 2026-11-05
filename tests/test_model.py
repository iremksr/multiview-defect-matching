
import torch

from models.backbone import MultiViewBackbone


def test_backbone():
    model = MultiViewBackbone(
        pretrained=False,
        freeze_backbone=True,
        out_channels=128
    )

    model.eval()

    images = torch.randn(1, 9, 1, 128, 128)

    with torch.no_grad():
        features = model(images)

    for name, feature in features.items():
        print(f"{name}: {tuple(feature.shape)}")

    assert features["p2"].shape == (1, 9, 128, 32, 32)
    assert features["p3"].shape == (1, 9, 128, 16, 16)
    assert features["p4"].shape == (1, 9, 128, 8, 8)

    print("Backbone test PASSED")


if __name__ == "__main__":
    test_backbone()
