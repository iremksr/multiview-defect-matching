
import torch
import torch.nn as nn
import torch.nn.functional as F


class MatchingLoss(nn.Module):
    def __init__(self, positive_weight=0.5):
        super().__init__()

        if not 0 <= positive_weight <= 1:
            raise ValueError(
                "positive_weight must be between 0 and 1"
            )

        self.positive_weight = positive_weight

    def forward(self, logits, valid_mask, metadata, instance_ids):
        """
        Args:
            logits: [N, N]
            valid_mask: [N, N]
            metadata: [N, 3] (batch_id, view_id, box_id)
            instance_ids: nested list [B][V][N_bv]

        Returns:
            loss: scalar tensor
            stats: dictionary
        """
        N = metadata.shape[0]

        if logits.shape != (N, N):
            raise ValueError("logits must have shape [N, N]")

        if valid_mask.shape != (N, N):
            raise ValueError("valid_mask must have shape [N, N]")

        # Retrieve GT instance IDs in ROI order.
        # IDs are only comparable within the same scene.
        keys = []

        for batch_id, view_id, box_id in metadata.tolist():
            instance_id = instance_ids[batch_id][view_id][box_id]
            keys.append((batch_id, str(instance_id)))

        # Construct ground-truth matching matrix.
        gt = torch.zeros(
            (N, N),
            dtype=logits.dtype,
            device=logits.device,
        )

        for i in range(N):
            for j in range(i + 1, N):
                if keys[i] == keys[j]:
                    gt[i, j] = 1.0
                    gt[j, i] = 1.0

        # Only count unique, valid pairs.
        upper_triangle = torch.triu(
            torch.ones(
                (N, N),
                dtype=torch.bool,
                device=logits.device,
            ),
            diagonal=1,
        )

        pair_mask = valid_mask.bool() & upper_triangle

        positive_mask = pair_mask & (gt == 1)
        negative_mask = pair_mask & (gt == 0)

        num_pos = int(positive_mask.sum().item())
        num_neg = int(negative_mask.sum().item())

        if num_pos > 0:
            pos_loss = F.binary_cross_entropy_with_logits(
                logits[positive_mask],
                gt[positive_mask],
            )
        else:
            pos_loss = logits.sum() * 0.0

        if num_neg > 0:
            neg_loss = F.binary_cross_entropy_with_logits(
                logits[negative_mask],
                gt[negative_mask],
            )
        else:
            neg_loss = logits.sum() * 0.0

        # Renormalize weights when one class is absent.
        pos_w = self.positive_weight if num_pos else 0.0
        neg_w = (1 - self.positive_weight) if num_neg else 0.0

        weight_sum = pos_w + neg_w

        if weight_sum > 0:
            loss = (
                pos_w * pos_loss + neg_w * neg_loss
            ) / weight_sum
        else:
            loss = logits.sum() * 0.0

        stats = {
            "loss": loss.detach().item(),
            "positive_loss": pos_loss.detach().item(),
            "negative_loss": neg_loss.detach().item(),
            "num_positive_pairs": num_pos,
            "num_negative_pairs": num_neg,
        }

        return loss, stats
