
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class MessyTableDataset(Dataset):
    def __init__(self, root, split="train"):
        self.root = Path(root)
        self.split = split

        self.annotation_files = sorted(
            (self.root / split).glob("*/annotations.json")
        )

        if not self.annotation_files:
            raise FileNotFoundError(
                f"No annotations found: {self.root / split}"
            )

    def __len__(self):
        return len(self.annotation_files)

    def __getitem__(self, index):
        annotation_path = self.annotation_files[index]

        with open(annotation_path, encoding="utf-8") as f:
            data = json.load(f)

        images = []
        boxes = []
        instance_ids = []

        for view in data["views"]:
            image_path = Path(view["image"])

            if not image_path.is_file():
                raise FileNotFoundError(image_path)

            with Image.open(image_path) as image:
                image_array = np.asarray(
                    image.convert("L"), dtype=np.float32
                ).copy() / 255.0

            images.append(
                torch.from_numpy(image_array).unsqueeze(0)
            )

            view_boxes = [
                item["bbox"] for item in view["boxes"]
            ]

            view_ids = [
                str(item["instance_id"])
                for item in view["boxes"]
            ]

            if view_boxes:
                box_tensor = torch.tensor(
                    view_boxes, dtype=torch.float32
                )
            else:
                box_tensor = torch.empty(
                    (0, 4), dtype=torch.float32
                )

            boxes.append(box_tensor)
            instance_ids.append(view_ids)

        return {
            "images": torch.stack(images),
            "boxes": boxes,
            "instance_ids": instance_ids,
            "scene_id": data["scene_id"],
        }


def messytable_collate_fn(batch):
    return {
        "images": torch.stack([
            item["images"] for item in batch
        ]),
        "boxes": [
            item["boxes"] for item in batch
        ],
        "instance_ids": [
            item["instance_ids"] for item in batch
        ],
        "scene_ids": [
            item["scene_id"] for item in batch
        ],
    }
