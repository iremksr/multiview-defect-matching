
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
            boxes[b][v]: [N_bv, 4]
            Bounding boxes in xyxy pixel coordinates.

    Outputs:
        Matching logits, valid pair mask,
        contextual embeddings, attention weights,
        ROI metadata and geometry information.
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

        # --------------------------------------------------
        # 1. Backbone: ResNet18 + FPN
        # --------------------------------------------------

        self.backbone = MultiViewBackbone(
            pretrained=pretrained,
            freeze_backbone=freeze_backbone,
            out_channels=fpn_channels,
        )

        # --------------------------------------------------
        # 2. Multi-scale ROI Context
        # --------------------------------------------------

        self.roi_context = MultiScaleROIContext()

        # --------------------------------------------------
        # 3. Context Fusion
        # --------------------------------------------------

        self.context_fusion = ContextFusion(
            in_channels=fpn_channels,
            embedding_dim=embedding_dim,
            num_contexts=9,
            dropout=dropout,
        )

        # --------------------------------------------------
        # 4. Multi-View Transformer
        # --------------------------------------------------

        self.transformer = MultiViewTransformer(
            embedding_dim=embedding_dim,
            num_layers=transformer_layers,
            num_heads=transformer_heads,
            dropout=dropout,
            feedforward_dim=transformer_ffn_dim,
            use_geometry=use_geometry,
        )

        # --------------------------------------------------
        # 5. Matching Head
        # --------------------------------------------------

        self.matching_head = MatchingHead(
            embedding_dim=embedding_dim,
            projection_dim=projection_dim,
            temperature=temperature,
        )

    @staticmethod
    def normalize_boxes(boxes, metadata, image_size):
        """
        Convert xyxy pixel coordinates to normalized cxcywh.

        Args:
            boxes:
                boxes[b][v]: [N_bv, 4]

            metadata:
                [N, 3] containing
                (batch_id, view_index, box_index)

            image_size:
                (H, W)

        Returns:
            normalized_boxes:
                [N, 4] containing
                (cx, cy, width, height)
                normalized to [0, 1]
        """

        H, W = image_size

        if H <= 0 or W <= 0:
            raise ValueError("Invalid image size")

        if metadata.numel() == 0:
            return torch.empty(
                (0, 4),
                dtype=torch.float32,
                device=metadata.device,
            )

        normalized = []

        for batch_id, view_id, box_id in metadata.tolist():

            box = boxes[batch_id][view_id][box_id]

            box = torch.as_tensor(
                box,
                dtype=torch.float32,
                device=metadata.device,
            )

            x1, y1, x2, y2 = box.unbind()

            cx = (x1 + x2) / (2.0 * W)
            cy = (y1 + y2) / (2.0 * H)

            width = (x2 - x1) / W
            height = (y2 - y1) / H

            normalized_box = torch.stack(
                [cx, cy, width, height]
            )

            normalized.append(normalized_box)

        return torch.stack(normalized, dim=0)

    def forward(self, images, boxes):
        """
        Args:
            images:
                [B, V, 1, H, W]

            boxes:
                Nested list [B][V].
                Each element is [N_bv, 4].

        Returns:
            Dictionary with model outputs.
        """

        # --------------------------------------------------
        # Input validation
        # --------------------------------------------------

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

        for b in range(B):
            if len(boxes[b]) != V:
                raise ValueError(
                    f"Sample {b} has {len(boxes[b])} views, "
                    f"but images contain {V} views"
                )

        # ==================================================
        # STEP 1: Backbone + FPN
        # ==================================================

        features = self.backbone(images)

        # features:
        # {
        #     "p2": [B, V, 128, H/4, W/4],
        #     "p3": [B, V, 128, H/8, W/8],
        #     "p4": [B, V, 128, H/16, W/16]
        # }

        # ==================================================
        # STEP 2: Multi-Scale ROI Context
        # ==================================================

        roi_features, metadata = self.roi_context(
            features,
            boxes,
            image_size=(H, W),
        )

        # roi_features: [N, 9, 128, 7, 7]
        # metadata: [N, 3]

        # ==================================================
        # STEP 3: Context Fusion
        # ==================================================

        embeddings, attention_weights = self.context_fusion(
            roi_features
        )

        # embeddings: [N, 256]
        # attention_weights: [N, 9]

        # ==================================================
        # STEP 4: Normalize Bounding Box Geometry
        # ==================================================

        normalized_boxes = self.normalize_boxes(
            boxes,
            metadata,
            image_size=(H, W),
        )

        normalized_boxes = normalized_boxes.to(
            device=embeddings.device,
            dtype=embeddings.dtype,
        )

        # normalized_boxes: [N, 4]

        # ==================================================
        # STEP 5: Multi-View Transformer
        # ==================================================

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

        # contextual_embeddings: [N, 256]
        # padded_embeddings: [B, Nmax, 256]
        # padding_mask: [B, Nmax]
        # padded_indices: [B, Nmax]

        # ==================================================
        # STEP 6: Matching Head
        # ==================================================

        (
            matching_logits,
            valid_mask,
            projected_embeddings,
        ) = self.matching_head(
            contextual_embeddings,
            metadata,
        )

        # matching_logits: [N, N]
        # valid_mask: [N, N]
        # projected_embeddings: [N, 128]

        # ==================================================
        # STEP 7: Return Outputs
        # ==================================================

        return {
            "matching_logits": matching_logits,
            "valid_mask": valid_mask,
            "metadata": metadata,
            "normalized_boxes": normalized_boxes,
            "embeddings": embeddings,
            "attention_weights": attention_weights,
            "contextual_embeddings": contextual_embeddings,
            "projected_embeddings": projected_embeddings,
            "padded_embeddings": padded_embeddings,
            "padding_mask": padding_mask,
            "padded_indices": padded_indices,
        }
