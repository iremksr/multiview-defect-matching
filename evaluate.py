
import argparse

import torch
from torch.utils.data import DataLoader

from datasets.messytable import (
    MessyTableDataset,
    messytable_collate_fn,
)
from models.h1_model import H1Model
from losses.matching_loss import MatchingLoss

from train import epoch_run


def main():

    parser = argparse.ArgumentParser(
        description="Evaluate H1 Multi-View Matching"
    )

    parser.add_argument(
        "--data-root",
        default=r"D:\irem_ssh\MessyTableH1",
    )

    parser.add_argument(
        "--checkpoint",
        default="runs/h1_bs4/best.pt",
    )

    parser.add_argument(
        "--split",
        choices=["train", "val", "test"],
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
        default=0.5,
    )

    parser.add_argument(
        "--no-amp",
        action="store_true",
    )

    args = parser.parse_args()

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp = device.type == "cuda" and not args.no_amp

    # ========================================================
    # LOAD CHECKPOINT
    # ========================================================

    checkpoint = torch.load(
        args.checkpoint,
        map_location=device,
        weights_only=False,
    )

    model = H1Model(
        pretrained=False,
        freeze_backbone=True,
    ).to(device)

    model.load_state_dict(
        checkpoint["model"]
    )

    model.eval()

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

    criterion = MatchingLoss(
        positive_weight=0.5,
    )

    # ========================================================
    # EVALUATION
    # ========================================================

    with torch.no_grad():

        metrics = epoch_run(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            amp=amp,
        )

    print("\n" + "=" * 65)
    print("H1 EVALUATION RESULTS")
    print("=" * 65)

    print("Split:", args.split)
    print("Checkpoint epoch:", checkpoint["epoch"])
    print("Threshold:", args.threshold)

    print(f"Loss:      {metrics['loss']:.6f}")
    print(f"Precision: {metrics['precision']:.6f}")
    print(f"Recall:    {metrics['recall']:.6f}")
    print(f"F1:        {metrics['f1']:.6f}")

    print("\nConfusion Matrix:")
    print("TP:", metrics["tp"])
    print("FP:", metrics["fp"])
    print("TN:", metrics["tn"])
    print("FN:", metrics["fn"])

    print("\nTotal evaluated pairs:", metrics["pairs"])
    print("Skipped batches:", metrics["skipped_batches"])


if __name__ == "__main__":
    main()
