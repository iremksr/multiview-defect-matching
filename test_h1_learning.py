
import random
import numpy as np
import torch

from datasets.messytable import MessyTableDataset
from models.h1_model import H1Model
from losses.matching_loss import MatchingLoss


# ============================================================
# CONFIG
# ============================================================

DATA_ROOT = r"D:\irem_ssh\MessyTableH1"

SEED = 42
STEPS = 100
LR = 1e-3
LOG_EVERY = 10

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ============================================================
# BUILD GROUND-TRUTH PAIR MASKS
# ============================================================

def build_pair_masks(outputs, instance_ids):

    metadata = outputs["metadata"]
    valid_mask = outputs["valid_mask"]

    keys = []

    for b, v, i in metadata.tolist():
        keys.append(
            (int(b), str(instance_ids[b][v][i]))
        )

    N = len(keys)

    gt_positive = torch.zeros(
        (N, N),
        dtype=torch.bool,
        device=valid_mask.device,
    )

    for i in range(N):
        for j in range(i + 1, N):
            if keys[i] == keys[j]:
                gt_positive[i, j] = True
                gt_positive[j, i] = True

    upper_triangle = torch.triu(
        torch.ones_like(valid_mask, dtype=torch.bool),
        diagonal=1,
    )

    pair_mask = valid_mask.bool() & upper_triangle

    positive_mask = pair_mask & gt_positive
    negative_mask = pair_mask & ~gt_positive

    return positive_mask, negative_mask


# ============================================================
# MATCHING METRICS
# ============================================================

@torch.no_grad()
def compute_metrics(outputs, instance_ids):

    logits = outputs["matching_logits"]

    probabilities = torch.sigmoid(logits)

    positive_mask, negative_mask = build_pair_masks(
        outputs,
        instance_ids,
    )

    pos_scores = probabilities[positive_mask]
    neg_scores = probabilities[negative_mask]

    if pos_scores.numel() == 0 or neg_scores.numel() == 0:
        raise RuntimeError(
            "Both positive and negative pairs are required."
        )

    pos_mean = pos_scores.mean().item()
    neg_mean = neg_scores.mean().item()

    tp = (pos_scores >= 0.5).sum().item()
    fn = (pos_scores < 0.5).sum().item()

    fp = (neg_scores >= 0.5).sum().item()
    tn = (neg_scores < 0.5).sum().item()

    accuracy = (tp + tn) / (tp + tn + fp + fn)

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    return {
        "pos_mean": pos_mean,
        "neg_mean": neg_mean,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


# ============================================================
# FIND SUITABLE SCENE
# ============================================================

def find_training_scene(dataset):

    for idx in range(len(dataset)):

        sample = dataset[idx]
        ids = sample["instance_ids"]

        positive = 0
        negative = 0

        for v1 in range(len(ids)):
            for v2 in range(v1 + 1, len(ids)):
                for id1 in ids[v1]:
                    for id2 in ids[v2]:

                        if id1 == id2:
                            positive += 1
                        else:
                            negative += 1

        if positive > 0 and negative > 0:
            return sample, positive, negative

    raise RuntimeError(
        "No scene with positive and negative pairs found."
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("H1 SINGLE-BATCH LEARNING TEST")
    print("=" * 65)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    dataset = MessyTableDataset(
        root=DATA_ROOT,
        split="train",
    )

    sample, num_pos, num_neg = find_training_scene(dataset)

    print("\n=== DATASET ===")
    print("Total scenes:", len(dataset))
    print("Selected scene:", sample["scene_id"])
    print("Images:", sample["images"].shape)
    print("Positive pairs:", num_pos)
    print("Negative pairs:", num_neg)

    images = sample["images"].unsqueeze(0).to(DEVICE)

    boxes = [[
        box.to(DEVICE)
        for box in sample["boxes"]
    ]]

    instance_ids = [sample["instance_ids"]]

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = H1Model(
        pretrained=True,
        freeze_backbone=True,
        temperature=0.1,
    ).to(DEVICE)

    criterion = MatchingLoss(
        positive_weight=0.5
    )

    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters(),
        ),
        lr=LR,
        weight_decay=1e-4,
    )

    # --------------------------------------------------------
    # Initial evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(images, boxes)

        initial_loss, initial_stats = criterion(
            logits=outputs["matching_logits"],
            valid_mask=outputs["valid_mask"],
            metadata=outputs["metadata"],
            instance_ids=instance_ids,
        )

        initial_metrics = compute_metrics(
            outputs,
            instance_ids,
        )

    print("\n=== INITIAL EVALUATION ===")
    print(f"Loss: {initial_loss.item():.6f}")
    print(f"Positive mean: {initial_metrics['pos_mean']:.6f}")
    print(f"Negative mean: {initial_metrics['neg_mean']:.6f}")
    print(f"Accuracy: {initial_metrics['accuracy']:.4f}")
    print(f"Precision: {initial_metrics['precision']:.4f}")
    print(f"Recall: {initial_metrics['recall']:.4f}")
    print(f"F1: {initial_metrics['f1']:.4f}")

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------

    print("\n=== TRAINING ===")

    for step in range(1, STEPS + 1):

        model.train()

        optimizer.zero_grad(set_to_none=True)

        outputs = model(images, boxes)

        loss, stats = criterion(
            logits=outputs["matching_logits"],
            valid_mask=outputs["valid_mask"],
            metadata=outputs["metadata"],
            instance_ids=instance_ids,
        )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at step {step}: {loss.item()}"
            )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
            error_if_nonfinite=True,
        )

        optimizer.step()

        if step == 1 or step % LOG_EVERY == 0:

            # Evaluate without dropout.
            model.eval()

            with torch.no_grad():

                eval_outputs = model(images, boxes)

                eval_loss, _ = criterion(
                    logits=eval_outputs["matching_logits"],
                    valid_mask=eval_outputs["valid_mask"],
                    metadata=eval_outputs["metadata"],
                    instance_ids=instance_ids,
                )

                metrics = compute_metrics(
                    eval_outputs,
                    instance_ids,
                )

            print(
                f"Step {step:03d}/{STEPS} | "
                f"Loss: {eval_loss.item():.5f} | "
                f"Pos: {metrics['pos_mean']:.4f} | "
                f"Neg: {metrics['neg_mean']:.4f} | "
                f"Acc: {metrics['accuracy']:.4f} | "
                f"F1: {metrics['f1']:.4f}",
                flush=True,
            )

    # --------------------------------------------------------
    # Final evaluation
    # --------------------------------------------------------

    model.eval()

    with torch.no_grad():

        outputs = model(images, boxes)

        final_loss, final_stats = criterion(
            logits=outputs["matching_logits"],
            valid_mask=outputs["valid_mask"],
            metadata=outputs["metadata"],
            instance_ids=instance_ids,
        )

        final_metrics = compute_metrics(
            outputs,
            instance_ids,
        )

    print("\n=== FINAL RESULTS ===")

    print(f"Initial loss: {initial_loss.item():.6f}")
    print(f"Final loss:   {final_loss.item():.6f}")

    print(
        f"Positive mean: "
        f"{initial_metrics['pos_mean']:.4f} -> "
        f"{final_metrics['pos_mean']:.4f}"
    )

    print(
        f"Negative mean: "
        f"{initial_metrics['neg_mean']:.4f} -> "
        f"{final_metrics['neg_mean']:.4f}"
    )

    print(
        f"F1: "
        f"{initial_metrics['f1']:.4f} -> "
        f"{final_metrics['f1']:.4f}"
    )

    print("\nConfusion matrix counts:")
    print("TP:", final_metrics["tp"])
    print("FP:", final_metrics["fp"])
    print("TN:", final_metrics["tn"])
    print("FN:", final_metrics["fn"])

    print("\n=== TEST COMPLETED ===")


if __name__ == "__main__":
    main()
