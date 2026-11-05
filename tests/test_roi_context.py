
import torch

from models.backbone import MultiViewBackbone
from models.roi_context import MultiScaleROIContext


def test_roi_context():
    backbone = MultiViewBackbone(
        pretrained=False,
        freeze_backbone=True,
        out_channels=128,
    )
    backbone.eval()

    roi_extractor = MultiScaleROIContext(
        output_size=7,
        context_scales=(1.0, 3.0, 5.0),
    )

    images = torch.randn(1, 9, 1, 128, 128)

    # Two boxes per view
    boxes = [[
        torch.tensor([
            [20.0, 30.0, 28.0, 38.0],
            [70.0, 65.0, 78.0, 73.0],
        ])
        for _ in range(9)
    ]]

    with torch.no_grad():
        features = backbone(images)
        roi_features, metadata = roi_extractor(
            features,
            boxes,
            image_size=(128, 128),
        )

    print("ROI features:", roi_features.shape)
    print("Metadata:", metadata.shape)

    assert roi_features.shape == (18, 9, 128, 7, 7)
    assert metadata.shape == (18, 3)
    assert torch.isfinite(roi_features).all()

    print("ROI Context test PASSED")


if __name__ == "__main__":
    test_roi_context()
