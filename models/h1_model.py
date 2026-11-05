
import torch
import torch.nn as nn

from models.backbone import MultiViewBackbone
from models.roi_context import MultiScaleROIContext
from models.context_fusion import ContextFusion
from models.multiview_transformer import MultiViewTransformer
from models.matching_head import MatchingHead


class H1Model(nn.Module):
    """
    H1: Multi-View Local Context Matching.

    Inputs:
        images: [B, V, 1, H, W]
        boxes: nested list
            boxes[b][v]: [N_bv, 4] in xyxy format

    Outputs:
        matching logits and valid pair mask,
        alongside embeddings and ROI metadata.
    """

    def __init__(
        self,
        pretrained=True,
        freeze_backbone=True,
        fpn_channels=128,
        embedding_dim=256,
        projection_dim=128,
        transformer_layers=2,
        transformer_heads=4,
        transformer_ffn_dim=512,
        dropout=0.1,
        temperature=0.1,
        use_geometry=True,
    ):
        super().__init__()

        self.backbone = MultiViewBackbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            out_channels=fpn_channels,
        )

        self.roi_context = MultiScaleROIContext()

        self.context_fusion = ContextFusion()

        self.transformer = MultiViewTransformer(
            embedding_dim=embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
            feedforward_dim=transformer_ffn_dim,
            use_geometry=use_geometry,
        )

        self.matching_head = MatchingHead(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            temperature=temperature,
        )

    @staticmethod
    def normalize_boxes(boxes, metadata, image_size):
        """
        Convert xyxy boxes to normalized cxcywh.

        Args:
            boxes:
                boxes[b][v]: [N_bv, 4]

            metadata:
                [N, 3] (batch_id, view_index, box_index)

            image_size:
                (H, W)

        Returns:
            normalized_boxes: [N, 4]
        """
        H, W = image_size

        if metadata.numel() == 0:
            return torch.empty(
                (0, 4),
                dtype=torch.float32,
                device=metadata.device,
            )

        normalized = []

        for batch_id, view_id, box_id in metadata.tolist():
            box = boxes[batch_id][view_id][box_id]

            x1, y1, x2, y2 = box.unbind()

            normalized.append(
                torch.stack([
                    (x1 + x2) / (2 * W),
                    (y1 + y2) / (2 * H),
                    (x2 - x1) / W,
                    (y2 - y1) / H,
                ])
            )

        return torch.stack(normalized).to(
            device=metadata.device
        )

    def forward(self, images, boxes):
        """
        Args:
            images: [B, V, 1, H, W]
            boxes: nested list [B][V]

        Returns:
            Dictionary containing matching predictions.
        """
        if images.ndim != 5:
            raise ValueError(
                "images must have shape [B, V, 1, H, W]"
            )

        B, V, C, H, W = images.shape

        if C != 1:
            raise ValueError(
                "H1 expects grayscale images (C=1)"
            )

        if len(boxes) != B:
            raise ValueError(
                "boxes must contain B samples"
            )

        # 1. Backbone + FPN
        features = self.backbone(images)

        # 2. Multi-scale ROI context
        roi_features, metadata = self.roi_context(
            features,
            boxes,
            image_size=(H, W),
        )

        # 3. Context fusion
        embeddings = self.context_fusion(
            roi_features
        )

        # 4. Normalize geometry in ROI order
        normalized_boxes = self.normalize_boxes(
            boxes,
            metadata,
            image_size=(H, W),
        )

        # 5. Multi-view Transformer
        (
            contextual_embeddings,
            padded_embeddings,
            padding_mask,
            padded_indices,
        ) = self.transformer(
            embeddings=embeddings,
            metadata=metadata,
            normalized_boxes=normalized_boxes,
            batch_size=B,
        )

        # 6. Matching head
        (
            matching_logits,
            valid_mask,
            projected_embeddings,
        ) = self.matching_head(
            contextual_embeddings,
            metadata,
        )

        return {
            "matching_logits": matching_logits,
            "valid_mask": valid_mask,
            "metadata": metadata,
            "normalized_boxes": normalized_boxes,
            "embeddings": embeddings,
            "contextual_embeddings": contextual_embeddings,
            "projected_embeddings": projected_embeddings,
            "padded_embeddings": padded_embeddings,
            "padding_mask": padding_mask,
            "padded_indices": padded_indices,
        }
