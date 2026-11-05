
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingHead(nn.Module):
    def __init__(
        self,
        embedding_dim=256,
        projection_dim=128,
        temperature=0.1,
    ):
        super().__init__()

        if temperature <= 0:
            raise ValueError("temperature must be positive")

        self.temperature = temperature

        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, projection_dim),
        )

    def forward(self, embeddings, metadata):
        """
        embeddings: [N, D]
        metadata: [N, 3]

        Returns:
            logits: [N, N]
            valid_mask: [N, N]
            projected_embeddings: [N, projection_dim]
        """

        if embeddings.ndim != 2:
            raise ValueError("embeddings must be [N, D]")

        N = embeddings.shape[0]

        if metadata.shape != (N, 3):
            raise ValueError("metadata must be [N, 3]")

        projected = self.projection(embeddings)

        projected = F.normalize(
            projected,
            p=2,
            dim=-1,
        )

        logits = (
            projected @ projected.T
        ) / self.temperature

        metadata = metadata.to(
            device=embeddings.device,
            dtype=torch.long,
        )

        batch_ids = metadata[:, 0]
        view_ids = metadata[:, 1]

        same_batch = (
            batch_ids[:, None] == batch_ids[None, :]
        )

        different_view = (
            view_ids[:, None] != view_ids[None, :]
        )

        valid_mask = same_batch & different_view

        return logits, valid_mask, projected
