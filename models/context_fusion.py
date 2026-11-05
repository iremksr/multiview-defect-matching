
import torch
import torch.nn as nn


class ContextFusion(nn.Module):
    def __init__(
        self,
        in_channels=128,
        embedding_dim=256,
        num_contexts=9,
        dropout=0.1,
    ):
        super().__init__()

        self.num_contexts = num_contexts

        # Spatial information is processed before pooling.
        self.context_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels,
                in_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )

        self.projection = nn.Sequential(
            nn.Linear(in_channels, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Learnable identity for each FPN-level/context-scale pair.
        self.context_embeddings = nn.Parameter(
            torch.zeros(1, num_contexts, embedding_dim)
        )
        nn.init.normal_(
            self.context_embeddings, std=0.02
        )

        self.attention = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 2),
            nn.Tanh(),
            nn.Linear(embedding_dim // 2, 1),
        )

        self.output_norm = nn.LayerNorm(embedding_dim)

    def forward(self, roi_features):
        """
        roi_features: [N, 9, 128, 7, 7]

        Returns:
            embeddings: [N, 256]
            attention_weights: [N, 9]
        """
        N, K, C, H, W = roi_features.shape

        if K != self.num_contexts:
            raise ValueError(
                f"Expected {self.num_contexts} contexts, got {K}"
            )

        if N == 0:
            return (
                roi_features.new_empty(
                    (0, self.output_norm.normalized_shape[0])
                ),
                roi_features.new_empty((0, K)),
            )

        x = roi_features.reshape(N * K, C, H, W)

        x = self.context_encoder(x)  # [N*K, 128]
        x = self.projection(x)       # [N*K, 256]

        x = x.reshape(N, K, -1)
        x = x + self.context_embeddings

        logits = self.attention(x).squeeze(-1)
        weights = torch.softmax(logits, dim=1)

        fused = torch.sum(
            x * weights.unsqueeze(-1),
            dim=1,
        )

        embeddings = self.output_norm(fused)

        return embeddings, weights
