
import argparse
import csv
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from datasets.messytable import (
    MessyTableDataset,
    messytable_collate_fn,
)
from models.h1_model import H1Model
from losses.matching_loss import MatchingLoss


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        default=r"D:\irem_ssh\MessyTableH1",
    )
    parser.add_argument(
        "--checkpoint",
        default=r"runs\h1_bs4\best.pt",
    )
    parser.add_argument(
        "--split",
        choices=["val", "test"],
        default="val",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Fixed threshold. Required for test split.",
    )
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    return parser.parse_args()


def make_ground_truth(metadata, instance_ids, device):
    """
    Construct GT labels for valid cross-view pairs.

    Each key includes batch index so instance IDs
    from different scenes cannot be confused.
    """

    meta = metadata.detach().cpu().tolist()

    keys = []

    for b, v, i in meta:
        keys.append(
            (int(b), str(instance_ids[b][v][i]))
        )

    return keys


@torch.no_grad()
def collect_predictions(
    model,
    loader,
    criterion,
    device,
    amp,
):
    all_scores = []
    all_labels = []

    total_loss = 0.0
    total_pairs = 0

    total_positive = 0
    total_negative = 0

    for batch_idx, batch in enumerate(loader, 1):

        images = batch["images"].to(device)

        boxes = [
            [
                box.to(device)
                for box in scene_boxes
            ]
            for scene_boxes in batch["boxes"]
        ]

        instance_ids = batch["instance_ids"]

        with torch.autocast(
            device_type=device.type,
            enabled=amp,
        ):
            outputs = model(images, boxes)

            loss, stats = criterion(
                outputs["matching_logits"],
                outputs["valid_mask"],
                outputs["metadata"],
                instance_ids,
            )

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Non-finite loss at batch {batch_idx}"
            )

        logits = outputs["matching_logits"]
        valid_mask = outputs["valid_mask"].bool()
        metadata = outputs["metadata"]

        n = metadata.shape[0]

        if n == 0:
            continue

        # Upper triangle prevents counting pairs twice.
        indices = torch.triu_indices(
            n,
            n,
            offset=1,
            device=logits.device,
        )

        ii, jj = indices

        keep = valid_mask[ii, jj]
        ii, jj = ii[keep], jj[keep]

        if ii.numel() == 0:
            continue

        keys = make_ground_truth(
            metadata,
            instance_ids,
            device,
        )

        # Match the same scene + instance ID.
        ii_cpu = ii.cpu().tolist()
        jj_cpu = jj.cpu().tolist()

        labels = torch.tensor(
            [
                int(keys[i] == keys[j])
                for i, j in zip(ii_cpu, jj_cpu)
            ],
            dtype=torch.int64,
        )

        scores = torch.sigmoid(
            logits[ii, jj].float()
        ).cpu()

        if not torch.isfinite(scores).all():
            raise RuntimeError(
                f"Non-finite scores in batch {batch_idx}"
            )

        all_scores.append(scores)
        all_labels.append(labels)

        num_pairs = len(labels)

        total_loss += loss.item() * num_pairs
        total_pairs += num_pairs

        total_positive += int(labels.sum().item())
        total_negative += int(
            (labels == 0).sum().item()
        )

        if batch_idx == 1 or batch_idx % 50 == 0:
            print(
                f"Processed {batch_idx}/{len(loader)} batches",
                flush=True,
            )

    if not all_scores:
        raise RuntimeError(
            "No valid cross-view pairs were found."
        )

    scores = torch.cat(all_scores)
    labels = torch.cat(all_labels)

    mean_loss = total_loss / max(total_pairs, 1)

    return {
        "scores": scores,
        "labels": labels,
        "loss": mean_loss,
        "positive": total_positive,
        "negative": total_negative,
    }


def compute_metrics(scores, labels, threshold):

    predictions = scores >= threshold

    positives = labels == 1
    negatives = labels == 0

    tp = int((predictions & positives).sum().item())
    fp = int((predictions & negatives).sum().item())
    tn = int((~predictions & negatives).sum().item())
    fn = int((~predictions & positives).sum().item())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall > 0
        else 0.0
    )

    accuracy = (tp + tn) / max(
        tp + tn + fp + fn, 1
    )

    return {
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "accuracy": accuracy,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
    }


def print_metrics(metrics):

    print(
        f"Threshold={metrics['threshold']:.4f} | "
        f"P={metrics['precision']:.4f} | "
        f"R={metrics['recall']:.4f} | "
        f"F1={metrics['f1']:.4f} | "
        f"Acc={metrics['accuracy']:.4f}"
    )


def main():

    args = parse_args()

    if args.split == "test" and args.threshold is None:
        raise ValueError(
            "For test split, provide the threshold "
            "selected on validation using --threshold."
        )

    if args.threshold is not None:
        if not 0.0 <= args.threshold <= 1.0:
            raise ValueError(
                "Threshold must be between 0 and 1."
            )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp = device.type == "cuda" and not args.no_amp

    checkpoint_path = Path(args.checkpoint)

    output_dir = Path(
        args.output
        if args.output is not None
        else checkpoint_path.parent / "evaluation"
    )

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 65)
    print("H1 MULTI-VIEW MATCHING EVALUATION")
    print("=" * 65)

    print("Device:", device)
    print("AMP:", amp)
    print("Split:", args.split)
    print("Checkpoint:", checkpoint_path)

    # ========================================================
    # DATASET
    # ========================================================

    dataset = MessyTableDataset(
        root=args.data_root,
        split=args.split,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=messytable_collate_fn,
        pin_memory=device.type == "cuda",
    )

    print("Scenes:", len(dataset))
    print("Batches:", len(loader))

    # ========================================================
    # MODEL
    # ========================================================

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    model = H1Model(
        pretrained=False,
        freeze_backbone=True,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True,
    )

    model.eval()

    criterion = MatchingLoss(
        positive_weight=0.5,
    )

    print(
        "Checkpoint epoch:",
        checkpoint.get("epoch", "unknown"),
    )

    # ========================================================
    # COLLECT PAIR SCORES
    # ========================================================

    print("\n=== COLLECTING PREDICTIONS ===")

    with torch.inference_mode():
        result = collect_predictions(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            amp=amp,
        )

    scores = result["scores"]
    labels = result["labels"]

    positive_scores = scores[labels == 1]
    negative_scores = scores[labels == 0]

    print("\n=== SCORE DISTRIBUTION ===")
    print("Loss:", round(result["loss"], 6))
    print("Total pairs:", len(scores))
    print("Positive pairs:", result["positive"])
    print("Negative pairs:", result["negative"])

    if len(positive_scores):
        print(
            "Positive mean:",
            round(positive_scores.mean().item(), 6),
        )

    if len(negative_scores):
        print(
            "Negative mean:",
            round(negative_scores.mean().item(), 6),
        )

    # ========================================================
    # THRESHOLD EVALUATION
    # ========================================================

    if args.threshold is None:

        thresholds = [
            round(x / 100, 2)
            for x in range(5, 100, 5)
        ]

        thresholds += [
            0.975,
            0.99,
            0.995,
            0.999,
        ]

        thresholds = sorted(set(thresholds))

    else:
        thresholds = [args.threshold]

    all_metrics = []

    print("\n=== THRESHOLD RESULTS ===")

    for threshold in thresholds:

        metrics = compute_metrics(
            scores,
            labels,
            threshold,
        )

        all_metrics.append(metrics)
        print_metrics(metrics)

    # ========================================================
    # SAVE CSV
    # ========================================================

    csv_path = output_dir / (
        f"{args.split}_threshold_results.csv"
    )

    with csv_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as f:

        writer = csv.DictWriter(
            f,
            fieldnames=list(all_metrics[0].keys()),
        )

        writer.writeheader()
        writer.writerows(all_metrics)

    # ========================================================
    # BEST VALIDATION THRESHOLD
    # ========================================================

    if args.split == "val" and args.threshold is None:

        best = max(
            all_metrics,
            key=lambda x: (
                x["f1"],
                x["precision"],
            ),
        )

        print("\n=== BEST VALIDATION THRESHOLD ===")
        print_metrics(best)

        selected_path = (
            output_dir / "selected_threshold.txt"
        )

        selected_path.write_text(
            str(best["threshold"]),
            encoding="utf-8",
        )

        print(
            "Selected threshold saved:",
            selected_path,
        )

    # ========================================================
    # FINAL CONFUSION MATRIX
    # ========================================================

    if args.split == "test":

        metrics = all_metrics[0]

        print("\n=== TEST CONFUSION MATRIX ===")

        print("TP:", metrics["tp"])
        print("FP:", metrics["fp"])
        print("TN:", metrics["tn"])
        print("FN:", metrics["fn"])

    print("\nResults saved:", csv_path)
    print("\nEVALUATION COMPLETED")


if __name__ == "__main__":
    main()
