
import torch
import torch.nn as nn


class MultiViewTransformer(nn.Module):
    """
    Camera-ID-free Multi-View Transformer.

    Each token represents a detected bounding box.

    Input:
        embeddings: [N, D]
        metadata: [N, 3]
            (batch_id, view_index, box_index)
        normalized_boxes: [N, 4]
            (cx, cy, w, h), normalized to [0, 1]
        batch_size: int

    Output:
        contextual_embeddings: [N, D]
        padded_embeddings: [B, Nmax, D]
        padding_mask: [B, Nmax]
        padded_indices: [B, Nmax]

    The transformer uses no camera identity embeddings
    and no sequence positional embeddings.
    """

    def __init__(
        self,
        embedding_dim=256,
        num_layers=2,
        num_heads=4,
        dropout=0.1,
        feedforward_dim=512,
        use_geometry=True,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.use_geometry = use_geometry

        if use_geometry:
            self.geometry_encoder = nn.Sequential(
                nn.Linear(4, 128),
                nn.GELU(),
                nn.Linear(128, embedding_dim),
            )
        else:
            self.geometry_encoder = None

        self.input_norm = nn.LayerNorm(embedding_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(
        self,
        embeddings,
        metadata,
        normalized_boxes,
        batch_size,
    ):
        """
        Args:
            embeddings:
                [N, D] visual context embeddings.

            metadata:
                [N, 3] containing
                (batch_id, view_index, box_index).

            normalized_boxes:
                [N, 4] containing
                (cx, cy, width, height).

            batch_size:
                Number of PCB samples.

        Returns:
            contextual_embeddings:
                [N, D] in original ROI order.

            padded_embeddings:
                [B, Nmax, D].

            padding_mask:
                [B, Nmax], True means padding.

            padded_indices:
                [B, Nmax], original ROI indices,
                -1 for padding.
        """

        if embeddings.ndim != 2:
            raise ValueError(
                "embeddings must have shape [N, D]"
            )

        N, D = embeddings.shape

        if D != self.embedding_dim:
            raise ValueError(
                f"Expected dimension {self.embedding_dim}, "
                f"got {D}"
            )

        if metadata.shape != (N, 3):
            raise ValueError(
                "metadata must have shape [N, 3]"
            )

        if self.use_geometry:
            if normalized_boxes is None:
                raise ValueError(
                    "normalized_boxes required when "
                    "use_geometry=True"
                )

            if normalized_boxes.shape != (N, 4):
                raise ValueError(
                    "normalized_boxes must have shape [N, 4]"
                )

        if batch_size < 1:
            raise ValueError(
                "batch_size must be at least 1"
            )

        device = embeddings.device

        # No detections in the entire batch.
        if N == 0:
            empty_padded = embeddings.new_zeros(
                (batch_size, 0, D)
            )

            empty_mask = torch.ones(
                (batch_size, 0),
                dtype=torch.bool,
                device=device,
            )

            empty_indices = torch.empty(
                (batch_size, 0),
                dtype=torch.long,
                device=device,
            )

            return (
                embeddings,
                empty_padded,
                empty_mask,
                empty_indices,
            )

        metadata = metadata.to(
            device=device,
            dtype=torch.long,
        )

        batch_ids = metadata[:, 0]

        if torch.any(batch_ids < 0) or torch.any(
            batch_ids >= batch_size
        ):
            raise ValueError(
                "Batch IDs must be in [0, batch_size)"
            )

        # Build tokens without camera embeddings.
        tokens = embeddings

        if self.use_geometry:
            normalized_boxes = normalized_boxes.to(
                device=device,
                dtype=embeddings.dtype,
            )

            geometry_features = self.geometry_encoder(
                normalized_boxes
            )

            tokens = tokens + geometry_features

        tokens = self.input_norm(tokens)

        # Count boxes belonging to each PCB sample.
        counts = torch.bincount(
            batch_ids,
            minlength=batch_size,
        )

        max_tokens = int(counts.max().item())

        padded = tokens.new_zeros(
            (batch_size, max_tokens, D)
        )

        padding_mask = torch.ones(
            (batch_size, max_tokens),
            dtype=torch.bool,
            device=device,
        )

        padded_indices = torch.full(
            (batch_size, max_tokens),
            -1,
            dtype=torch.long,
            device=device,
        )

        for b in range(batch_size):
            indices = torch.nonzero(
                batch_ids == b,
                as_tuple=True,
            )[0]

            count = indices.numel()

            if count == 0:
                continue

            padded[b, :count] = tokens[indices]

            padding_mask[b, :count] = False

            padded_indices[b, :count] = indices

        # Exclude completely empty PCB samples
        # to prevent attention over all-masked rows.
        valid_samples = counts > 0

        encoded = padded.new_zeros(
            padded.shape
        )

        if valid_samples.any():
            valid_tokens = padded[valid_samples]

            valid_mask = padding_mask[valid_samples]

            encoded_valid = self.transformer(
                valid_tokens,
                src_key_padding_mask=valid_mask,
            )

            encoded[valid_samples] = encoded_valid

        encoded = self.output_norm(encoded)

        # Keep padded positions exactly zero.
        encoded = encoded.masked_fill(
            padding_mask.unsqueeze(-1),
            0.0,
        )

        # Restore original flattened ROI order.
        contextual_embeddings = embeddings.new_zeros(
            (N, D)
        )

        valid_positions = ~padding_mask

        original_indices = padded_indices[
            valid_positions
        ]

        contextual_embeddings[original_indices] = (
            encoded[valid_positions]
        )

        return (
            contextual_embeddings,
            encoded,
            padding_mask,
            padded_indices,
        )
