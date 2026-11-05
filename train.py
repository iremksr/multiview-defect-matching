
import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from datasets.messytable import (
    MessyTableDataset,
    messytable_collate_fn,
)
from models.h1_model import H1Model
from losses.matching_loss import MatchingLoss


# ============================================================
# CONFIGURATION
# ============================================================

def parse_args():
    p = argparse.ArgumentParser(
        description="Train H1 Multi-View Matching"
    )

    p.add_argument(
        "--data-root",
        default=r"D:\irem_ssh\MessyTableH1",
    )
    p.add_argument("--output", default="runs/h1_bs4")
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.5)

    p.add_argument("--resume", action="store_true")
    p.add_argument("--no-amp", action="store_true")

    p.add_argument("--train-limit", type=int, default=0)
    p.add_argument("--val-limit", type=int, default=0)

    return p.parse_args()


# ============================================================
# PAIRWISE METRICS
# ============================================================

@torch.no_grad()
def pair_counts(
    logits,
    valid_mask,
    metadata,
    instance_ids,
    threshold,
):
    n = metadata.shape[0]

    keys = [
        (
            int(b),
            str(instance_ids[int(b)][int(v)][int(i)]),
        )
        for b, v, i in metadata.tolist()
    ]

    indices = torch.triu_indices(
        n,
        n,
        offset=1,
        device=logits.device,
    )

    ii, jj = indices

    keep = valid_mask[ii, jj].bool()
    ii, jj = ii[keep], jj[keep]

    if ii.numel() == 0:
        return 0, 0, 0, 0

    gt = torch.tensor(
        [
            keys[i] == keys[j]
            for i, j in zip(ii.tolist(), jj.tolist())
        ],
        device=logits.device,
        dtype=torch.bool,
    )

    pred = torch.sigmoid(logits[ii, jj]) >= threshold

    tp = int((pred & gt).sum().item())
    fp = int((pred & ~gt).sum().item())
    tn = int((~pred & ~gt).sum().item())
    fn = int((~pred & gt).sum().item())

    return tp, fp, tn, fn


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def epoch_run(
    model,
    loader,
    criterion,
    device,
    threshold,
    optimizer=None,
    scaler=None,
    amp=False,
):
    training = optimizer is not None

    model.train(training)

    total_loss = 0.0
    total_pairs = 0

    tp = fp = tn = fn = 0
    skipped = 0

    for batch_idx, batch in enumerate(loader, 1):

        images = batch["images"].to(
            device,
            non_blocking=True,
        )

        boxes = [
            [
                box.to(device, non_blocking=True)
                for box in sample
            ]
            for sample in batch["boxes"]
        ]

        instance_ids = batch["instance_ids"]

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):

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

            num_pairs = (
                stats["num_positive_pairs"]
                + stats["num_negative_pairs"]
            )

            if not torch.isfinite(loss):
                raise RuntimeError(
                    f"Non-finite loss in batch {batch_idx}: "
                    f"{loss.item()}"
                )

            # -----------------------------------------------
            # Backward + optimizer
            # -----------------------------------------------

            if training and num_pairs > 0:

                if scaler is not None and amp:

                    scaler.scale(loss).backward()

                    scaler.unscale_(optimizer)

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )

                    scaler.step(optimizer)
                    scaler.update()

                else:

                    loss.backward()

                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(),
                        max_norm=1.0,
                        error_if_nonfinite=True,
                    )

                    optimizer.step()

            # -----------------------------------------------
            # Metrics
            # -----------------------------------------------

            if num_pairs > 0:

                total_loss += loss.detach().item() * num_pairs
                total_pairs += num_pairs

                a, b, c, d = pair_counts(
                    outputs["matching_logits"].detach(),
                    outputs["valid_mask"],
                    outputs["metadata"],
                    instance_ids,
                    threshold,
                )

                tp += a
                fp += b
                tn += c
                fn += d

            else:
                skipped += 1

        if training and (
            batch_idx == 1 or batch_idx % 25 == 0
        ):

            print(
                f"  Batch {batch_idx}/{len(loader)} "
                f"Loss={loss.item():.5f}",
                flush=True,
            )

    # --------------------------------------------------------
    # Epoch metrics
    # --------------------------------------------------------

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    f1 = (
        2 * precision * recall
        / max(precision + recall, 1e-12)
    )

    return {
        "loss": (
            total_loss / total_pairs
            if total_pairs > 0
            else float("nan")
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "pairs": total_pairs,
        "skipped_batches": skipped,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_args()

    # --------------------------------------------------------
    # Seed
    # --------------------------------------------------------

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    amp = device.type == "cuda" and not args.no_amp

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = MessyTableDataset(
        root=args.data_root,
        split="train",
    )

    val_dataset = MessyTableDataset(
        root=args.data_root,
        split="val",
    )

    if args.train_limit > 0:
        train_dataset = torch.utils.data.Subset(
            train_dataset,
            range(min(args.train_limit, len(train_dataset))),
        )

    if args.val_limit > 0:
        val_dataset = torch.utils.data.Subset(
            val_dataset,
            range(min(args.val_limit, len(val_dataset))),
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=messytable_collate_fn,
        pin_memory=device.type == "cuda",
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=messytable_collate_fn,
        pin_memory=device.type == "cuda",
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = H1Model(
        pretrained=not args.resume,
        freeze_backbone=True,
    ).to(device)

    criterion = MatchingLoss(
        positive_weight=0.5,
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        (
            p for p in model.parameters()
            if p.requires_grad
        ),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # --------------------------------------------------------
    # Scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=5,
    )

    # --------------------------------------------------------
    # Mixed precision
    # --------------------------------------------------------

    scaler = (
        torch.amp.GradScaler("cuda", enabled=True)
        if amp
        else None
    )

    # --------------------------------------------------------
    # Resume
    # --------------------------------------------------------

    start_epoch = 1
    best_val_loss = float("inf")
    bad_epochs = 0

    last_path = output_dir / "last.pt"
    best_path = output_dir / "best.pt"

    if args.resume:

        if not last_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {last_path}"
            )

        checkpoint = torch.load(
            last_path,
            map_location=device,
            weights_only=False,
        )

        model.load_state_dict(checkpoint["model"])

        optimizer.load_state_dict(
            checkpoint["optimizer"]
        )

        scheduler.load_state_dict(
            checkpoint["scheduler"]
        )

        if (
            scaler is not None
            and checkpoint.get("scaler") is not None
        ):
            scaler.load_state_dict(checkpoint["scaler"])

        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_loss"]
        bad_epochs = checkpoint["bad_epochs"]

        print(
            f"Resuming from epoch {start_epoch}",
            flush=True,
        )

    # --------------------------------------------------------
    # CSV History
    # --------------------------------------------------------

    history_path = output_dir / "history.csv"

    columns = [
        "epoch",
        "lr",
        "train_loss",
        "train_precision",
        "train_recall",
        "train_f1",
        "val_loss",
        "val_precision",
        "val_recall",
        "val_f1",
        "train_pairs",
        "val_pairs",
    ]

    if not args.resume or not history_path.exists():

        with history_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as f:
            csv.writer(f).writerow(columns)

    print("=" * 65)
    print("H1 MULTI-VIEW MATCHING TRAINING")
    print("=" * 65)

    print("Device:", device)
    print("AMP:", amp)
    print("Batch size:", args.batch_size)
    print("Train scenes:", len(train_dataset))
    print("Val scenes:", len(val_dataset))
    print("Learning rate:", args.lr)
    print("Output:", output_dir)

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(start_epoch, args.epochs + 1):

        print(
            f"\n{'=' * 65}\n"
            f"EPOCH {epoch}/{args.epochs}\n"
            f"{'=' * 65}",
            flush=True,
        )

        train_metrics = epoch_run(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
            optimizer=optimizer,
            scaler=scaler,
            amp=amp,
        )

        with torch.no_grad():

            val_metrics = epoch_run(
                model=model,
                loader=val_loader,
                criterion=criterion,
                device=device,
                threshold=args.threshold,
                amp=amp,
            )

        val_loss = val_metrics["loss"]

        if not np.isfinite(val_loss):
            raise RuntimeError(
                "Validation loss is not finite. "
                "Check valid pairs and model outputs."
            )

        # ----------------------------------------------------
        # Scheduler
        # ----------------------------------------------------

        scheduler.step(val_loss)

        # ----------------------------------------------------
        # Best model
        # ----------------------------------------------------

        improved = (
            val_loss < best_val_loss - args.min_delta
        )

        if improved:
            best_val_loss = val_loss
            bad_epochs = 0
        else:
            bad_epochs += 1

        # ----------------------------------------------------
        # Checkpoint
        # ----------------------------------------------------

        checkpoint = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "scaler": (
                scaler.state_dict()
                if scaler is not None
                else None
            ),
            "best_loss": best_val_loss,
            "bad_epochs": bad_epochs,
            "args": vars(args),
        }

        torch.save(checkpoint, last_path)

        if improved:
            torch.save(checkpoint, best_path)

        # ----------------------------------------------------
        # CSV
        # ----------------------------------------------------

        with history_path.open(
            "a",
            newline="",
            encoding="utf-8",
        ) as f:

            csv.writer(f).writerow([
                epoch,
                optimizer.param_groups[0]["lr"],
                train_metrics["loss"],
                train_metrics["precision"],
                train_metrics["recall"],
                train_metrics["f1"],
                val_metrics["loss"],
                val_metrics["precision"],
                val_metrics["recall"],
                val_metrics["f1"],
                train_metrics["pairs"],
                val_metrics["pairs"],
            ])

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print("\n=== EPOCH RESULTS ===")

        print(
            f"TRAIN | "
            f"Loss={train_metrics['loss']:.5f} | "
            f"P={train_metrics['precision']:.4f} | "
            f"R={train_metrics['recall']:.4f} | "
            f"F1={train_metrics['f1']:.4f}"
        )

        print(
            f"VAL   | "
            f"Loss={val_metrics['loss']:.5f} | "
            f"P={val_metrics['precision']:.4f} | "
            f"R={val_metrics['recall']:.4f} | "
            f"F1={val_metrics['f1']:.4f}"
        )

        print(
            f"Best val loss: {best_val_loss:.5f}"
        )

        print(
            f"Patience: {bad_epochs}/{args.patience}"
        )

        if device.type == "cuda":
            print(
                "Peak GPU memory:",
                f"{torch.cuda.max_memory_allocated() / 1024**3:.2f} GB"
            )
            torch.cuda.reset_peak_memory_stats()

        # ----------------------------------------------------
        # Early stopping
        # ----------------------------------------------------

        if bad_epochs >= args.patience:

            print("\nEarly stopping triggered.")
            break

    print("\nTRAINING COMPLETED")
    print("Best checkpoint:", best_path)
    print("Last checkpoint:", last_path)
    print("History:", history_path)


if __name__ == "__main__":
    main()
