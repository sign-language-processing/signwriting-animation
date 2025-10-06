import os
import random
from typing import Literal

import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.pose import Pose
from pose_anonymization.data.normalization import normalize_mean_std
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image
from transformers import CLIPProcessor


class DynamicPosePredictionDataset(Dataset):
    """
    A PyTorch Dataset for dynamic sampling of normalized pose sequences,
    conditioned on SignWriting images and optional scalar metadata.
    Each sample includes past and future pose segments, associated masks,
    and a CLIP-ready rendering of the SignWriting annotation.
    """
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        num_past_frames: int = 40,
        num_future_frames: int = 20,
        with_metadata: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        split: Literal['train', 'dev', 'test'] = 'train'
    ):
        super().__init__()

        assert split in ['train', 'test', 'dev']

        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.with_metadata = with_metadata
        df_records = pd.read_csv(csv_path)
        df_records = df_records[df_records['split'] == split]
        self.records = df_records.to_dict(orient="records")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __len__(self):
        """
        Return the number of samples in the dataset.
        """
        return len(self.records)

    def __getitem__(self, idx):
        """
        Retrieve and process a single sample from the dataset.
        Loads pose data and associated SignWriting text, extracts past and future segments,
        and prepares masked tensors and CLIP-ready images.
        """
        rec = self.records[idx]

        pose_path = os.path.join(self.data_dir, rec["pose"])
        with open(pose_path, "rb") as f:
            # Read only the relevant frames from the pose file (based on time, in milliseconds)
            raw = Pose.read(f, start_time=rec["start"] or None, end_time=rec["end"] or None)
        pose = normalize_mean_std(raw)

        # The model expects constant size "input" and "target" windows
        total_frames = len(pose.body.data)
        min_pivot = 1
        max_pivot = max(1, total_frames - 1)
        pivot_frame = random.randint(min_pivot, max_pivot) # Choose a frame to separate the windows

        # Crop pose around the pivot. Window might not be of "constant" size, but it will be padded.
        input_start = max(0, pivot_frame - self.num_past_frames)
        # TODO: consider reversing input_pose, since it will be right-padded
        input_pose = pose.body[input_start:pivot_frame].torch()
        target_end = min(total_frames, pivot_frame + self.num_future_frames)
        target_pose = pose.body[pivot_frame:target_end].torch()

        input_data = input_pose.data
        target_data = target_pose.data

        input_mask = input_pose.data.mask
        target_mask = target_pose.data.mask

        pil_img = signwriting_to_clip_image(rec.get("text", ""))
        sign_img = self.clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)

        sample = {
            "data": target_data,
            "conditions": {
                "input_pose": input_data,
                "input_mask": input_mask,
                "target_mask": target_mask,
                "sign_image": sign_img,
            },
            "id": rec.get("id", os.path.basename(rec["pose"])),
        }

        if self.with_metadata:
            meta = {
                "total_frames": total_frames,
                "sample_start": pivot_frame,
                "sample_end": target_end,
                "orig_start": rec.get("start", 0),
                "orig_end": rec.get("end", total_frames),
            }
            sample["metadata"] = {
                k: torch.tensor([v], dtype=torch.long) for k, v in meta.items()
            }

        return sample

def get_num_workers():
    """
    Determine appropriate number of workers based on CPU availability.
    """
    cpu_count = os.cpu_count()
    return 0 if cpu_count is None or cpu_count <= 1 else cpu_count

def main():
    """
    Run a test batch through the dataset and dataloader.
    """
    data_dir = "/scratch/yayun/pose_data/raw_poses"
    csv_path = os.path.join(os.path.dirname(data_dir), "data.csv")

    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split='train'
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=get_num_workers(),
        pin_memory=False,
    )

    batch = next(iter(loader))
    print("Batch:", batch["data"].shape)
    print("Input pose:", batch["conditions"]["input_pose"].shape)
    print("Input mask:", batch["conditions"]["input_mask"].shape)
    print("Target mask:", batch["conditions"]["target_mask"].shape)
    print("Sign image:", batch["conditions"]["sign_image"].shape)
    if "metadata" in batch:
        for k, v in batch["metadata"].items():
            print(f"Metadata {k}:", v.shape)


if __name__ == "__main__":
    main()
