
import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from datasets.messytable import MessyTableDataset
from models.h1_model import H1Model


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--data-root",
        default=r"D:\irem_ssh\MessyTableH1"
    )
    p.add_argument(
        "--checkpoint",
        default=r"runs\h1_bs4\best.pt"
    )
    p.add_argument("--split", default="val")
    p.add_argument("--scene-index", type=int, default=-1)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--max-refs", type=int, default=5)
    p.add_argument(
        "--output",
        default=r"runs\h1_bs4\visualizations"
    )
    return p.parse_args()


def get_font(size=16):
    try:
        return ImageFont.truetype("arial.ttf", size)
    except OSError:
        return ImageFont.load_default()


def to_image(tensor):
    arr = tensor.squeeze().detach().cpu().numpy()
    arr = np.clip(arr, 0, 1)
    arr = (arr * 255).astype(np.uint8)
    return Image.fromarray(arr, mode="L").convert("RGB")


def draw_box(draw, box, color, label, width=3):
    x1, y1, x2, y2 = [float(x) for x in box]

    draw.rectangle(
        (x1, y1, x2, y2),
        outline=color,
        width=width
    )

    font = get_font(13)
    bbox = draw.textbbox((0, 0), label, font=font)
    tw = bbox[2] - bbox[0]
    th = bbox[3] - bbox[1]

    label_y = max(0, y1 - th - 8)

    draw.rectangle(
        (x1, label_y, x1 + tw + 8, label_y + th + 6),
        fill=color
    )
    draw.text(
        (x1 + 4, label_y + 2),
        label,
        fill="black",
        font=font
    )


def make_panel(image, title, boxes_to_draw):
    canvas = Image.new(
        "RGB",
        (image.width, image.height + 36),
        (25, 30, 38)
    )
    canvas.paste(image, (0, 36))

    draw = ImageDraw.Draw(canvas)
    draw.text(
        (10, 9),
        title,
        fill="white",
        font=get_font(17)
    )

    for box, color, label in boxes_to_draw:
        # Bounding boxes use original image coordinates.
        shifted = [
            box[0], box[1] + 36,
            box[2], box[3] + 36
        ]
        draw_box(draw, shifted, color, label)

    return canvas


def make_grid(panels, columns=3, gap=12):
    if not panels:
        raise ValueError("No panels")

    w = max(p.width for p in panels)
    h = max(p.height for p in panels)
    rows = (len(panels) + columns - 1) // columns

    grid = Image.new(
        "RGB",
        (
            columns * w + (columns + 1) * gap,
            rows * h + (rows + 1) * gap
        ),
        (18, 22, 28)
    )

    for i, panel in enumerate(panels):
        x = gap + (i % columns) * (w + gap)
        y = gap + (i // columns) * (h + gap)
        grid.paste(panel, (x, y))

    return grid


def choose_scene(dataset, scene_index):
    if scene_index >= 0:
        return dataset[scene_index], scene_index

    # Prefer a scene with several instances,
    # so the visualization is informative.
    for i in range(len(dataset)):
        sample = dataset[i]
        ids = set(
            str(instance_id)
            for view_ids in sample["instance_ids"]
            for instance_id in view_ids
        )
        if len(ids) >= 3:
            return sample, i

    return dataset[0], 0


def main():
    args = parse_args()

    if not 0 <= args.threshold <= 1:
        raise ValueError("Threshold must be in [0,1]")

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = MessyTableDataset(
        root=args.data_root,
        split=args.split
    )

    sample, scene_index = choose_scene(
        dataset, args.scene_index
    )

    print("Scene:", sample["scene_id"])
    print("Scene index:", scene_index)
    print("Views:", len(sample["images"]))

    checkpoint = torch.load(
        args.checkpoint,
        map_location="cpu",
        weights_only=False
    )

    model = H1Model(
        pretrained=False,
        freeze_backbone=True
    ).to(device)

    model.load_state_dict(
        checkpoint["model"],
        strict=True
    )
    model.eval()

    images = sample["images"].unsqueeze(0).to(device)

    boxes = [[
        box.to(device)
        for box in sample["boxes"]
    ]]

    with torch.inference_mode():
        outputs = model(images, boxes)

    scores = torch.sigmoid(
        outputs["matching_logits"].float()
    ).cpu()

    valid = outputs["valid_mask"].cpu().bool()
    metadata = outputs["metadata"].cpu().tolist()

    # Map each model token to its original view and box.
    token_to_box = {}
    box_to_token = {}

    for token_idx, (b, v, box_idx) in enumerate(metadata):
        if int(b) != 0:
            continue

        key = (int(v), int(box_idx))
        token_to_box[token_idx] = key
        box_to_token[key] = token_idx

    scene_dir = (
        output_dir /
        f"{args.split}_{scene_index}_{sample['scene_id']}"
    )
    scene_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================
    # 1. ALL BOXES OVERVIEW
    # ========================================================

    panels = []

    for v in range(len(sample["images"])):
        img = to_image(sample["images"][v])
        annotations = []

        for k, box in enumerate(sample["boxes"][v]):
            instance_id = str(
                sample["instance_ids"][v][k]
            )
            annotations.append((
                box.tolist(),
                "#45c7ff",
                f"B{k} ID:{instance_id}"
            ))

        panels.append(
            make_panel(
                img,
                f"View {v} | {len(annotations)} boxes",
                annotations
            )
        )

    overview = make_grid(panels)
    overview_path = scene_dir / "01_all_boxes.png"
    overview.save(overview_path)

    # ========================================================
    # 2. REFERENCE MATCH VISUALIZATION
    # ========================================================

    references = []

    # Prioritize boxes that have ground-truth matches
    # in multiple views.
    for token_idx, (v, k) in token_to_box.items():
        instance_id = str(sample["instance_ids"][v][k])

        gt_count = sum(
            1
            for other_idx, (ov, ok) in token_to_box.items()
            if ov != v
            and str(sample["instance_ids"][ov][ok])
            == instance_id
        )

        references.append(
            (gt_count, token_idx, v, k)
        )

    references.sort(reverse=True)

    for ref_number, (_, ref_token, ref_view, ref_box_idx) in enumerate(
        references[:args.max_refs], start=1
    ):
        ref_id = str(
            sample["instance_ids"][ref_view][ref_box_idx]
        )

        ref_box = sample["boxes"][ref_view][
            ref_box_idx
        ].tolist()

        match_panels = []
        match_rows = []

        for v in range(len(sample["images"])):
            img = to_image(sample["images"][v])
            annotations = []

            if v == ref_view:
                annotations.append((
                    ref_box,
                    "#45c7ff",
                    f"REFERENCE B{ref_box_idx} ID:{ref_id}"
                ))
            else:
                for k, box in enumerate(sample["boxes"][v]):
                    token = box_to_token.get((v, k))

                    if token is None:
                        continue

                    if not valid[ref_token, token]:
                        continue

                    score = float(
                        scores[ref_token, token].item()
                    )

                    target_id = str(
                        sample["instance_ids"][v][k]
                    )

                    is_gt = target_id == ref_id
                    predicted = score >= args.threshold

                    if predicted and is_gt:
                        color = "#35d69b"
                        status = "TP"
                    elif predicted and not is_gt:
                        color = "#ff5252"
                        status = "FP"
                    elif not predicted and is_gt:
                        color = "#ffb347"
                        status = "FN"
                    else:
                        # Hide true negatives to reduce clutter.
                        continue

                    annotations.append((
                        box.tolist(),
                        color,
                        f"{status} B{k} {score:.3f}"
                    ))

                    match_rows.append({
                        "view": v,
                        "box": k,
                        "instance_id": target_id,
                        "score": score,
                        "status": status,
                    })

            match_panels.append(
                make_panel(
                    img,
                    f"View {v}" + (
                        " [REFERENCE]"
                        if v == ref_view else ""
                    ),
                    annotations
                )
            )

        grid = make_grid(match_panels)

        output_path = scene_dir / (
            f"02_reference_{ref_number}"
            f"_view{ref_view}_box{ref_box_idx}.png"
        )
        grid.save(output_path)

        print(
            f"\nReference {ref_number}: "
            f"View {ref_view}, Box {ref_box_idx}, "
            f"GT instance {ref_id}"
        )

        for row in match_rows:
            print(
                f"  View {row['view']} "
                f"Box {row['box']} "
                f"Score={row['score']:.4f} "
                f"GT={row['instance_id']} "
                f"Status={row['status']}"
            )

        print("Saved:", output_path)

    print("\nOverview saved:", overview_path)
    print("Output directory:", scene_dir)
    print("\nVISUALIZATION COMPLETED")


if __name__ == "__main__":
    main()
