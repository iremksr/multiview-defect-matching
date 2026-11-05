
import torch
import torch.nn as nn
from torchvision.ops import roi_align


class MultiScaleROIContext(nn.Module):
    def __init__(
        self,
        output_size=7,
        context_scales=(1.0, 3.0, 5.0),
        min_context_size=8.0,
    ):
        super().__init__()
        self.output_size = output_size
        self.context_scales = context_scales
        self.min_context_size = min_context_size

        self.feature_strides = {
            "p2": 4,
            "p3": 8,
            "p4": 16,
        }

    def expand_boxes(self, boxes, scale, image_size):
        """
        boxes: [N, 4], xyxy in image pixels
        image_size: (H, W)
        """
        H, W = image_size

        x1, y1, x2, y2 = boxes.unbind(dim=-1)

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2

        width = (x2 - x1).clamp(min=1.0)
        height = (y2 - y1).clamp(min=1.0)

        width = (width * scale).clamp(
            min=self.min_context_size
        )
        height = (height * scale).clamp(
            min=self.min_context_size
        )

        x1 = (cx - width / 2).clamp(0, W)
        y1 = (cy - height / 2).clamp(0, H)
        x2 = (cx + width / 2).clamp(0, W)
        y2 = (cy + height / 2).clamp(0, H)

        return torch.stack([x1, y1, x2, y2], dim=-1)

    def forward(self, features, boxes, image_size):
        """
        features:
            p2: [B, V, C, H/4, W/4]
            p3: [B, V, C, H/8, W/8]
            p4: [B, V, C, H/16, W/16]

        boxes:
            List of length B.
            Each element is a list of length V.
            boxes[b][v]: Tensor [N_bv, 4]

        Returns:
            roi_features: [N_total, 9, C, 7, 7]
            metadata: [N_total, 3]
                      (batch_id, view_id, box_id)
        """
        B, V = features["p2"].shape[:2]
        device = features["p2"].device
        dtype = features["p2"].dtype

        all_boxes = []
        metadata = []

        for b in range(B):
            for v in range(V):
                current_boxes = boxes[b][v].to(
                    device=device, dtype=dtype
                )

                for box_id in range(len(current_boxes)):
                    all_boxes.append(current_boxes[box_id])
                    metadata.append([b, v, box_id])

        if not all_boxes:
            C = features["p2"].shape[2]
            return (
                features["p2"].new_empty(
                    (0, 9, C, self.output_size, self.output_size)
                ),
                torch.empty(
                    (0, 3), dtype=torch.long, device=device
                ),
            )

        all_boxes = torch.stack(all_boxes)
        metadata = torch.tensor(
            metadata, dtype=torch.long, device=device
        )

        # Flatten B and V into one image-batch dimension
        image_indices = (
            metadata[:, 0] * V + metadata[:, 1]
        )

        outputs = []

        for level_name, stride in self.feature_strides.items():
            feat = features[level_name]
            _, _, C, h, w = feat.shape

            flat_feat = feat.reshape(B * V, C, h, w)

            for scale in self.context_scales:
                expanded = self.expand_boxes(
                    all_boxes, scale, image_size
                )

                rois = torch.cat(
                    [
                        image_indices[:, None].to(expanded.dtype),
                        expanded,
                    ],
                    dim=1,
                )

                pooled = roi_align(
                    input=flat_feat,
                    boxes=rois,
                    output_size=self.output_size,
                    spatial_scale=1.0 / stride,
                    sampling_ratio=2,
                    aligned=True,
                )

                outputs.append(pooled)

        # [N_total, 9, C, 7, 7]
        roi_features = torch.stack(outputs, dim=1)

        return roi_features, metadata
