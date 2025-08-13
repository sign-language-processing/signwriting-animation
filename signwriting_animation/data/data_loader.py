import os
import random
from dataclasses import dataclass
from typing import Literal
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.pose import Pose
from pose_anonymization.data.normalization import normalize_mean_std
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image
from transformers import CLIPProcessor

@dataclass
class DatasetConfig:
    """
    Configuration for dataset paths and frame sampling.
    """
    data_dir: str
    csv_path: str
    num_past_frames: int = 40
    num_future_frames: int = 20
    split: Literal['train', 'test', 'dev'] = 'train'

class DynamicPosePredictionDataset(Dataset):
    """
    A PyTorch Dataset for dynamic sampling of normalized pose sequences,
    conditioned on SignWriting images and optional scalar metadata.
    Each sample includes past and future pose segments, associated masks,
    and a CLIP-ready rendering of the SignWriting annotation.
    """
    def __init__(
        self,
        config: DatasetConfig,
        with_metadata: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        assert config.split in ['train', 'test', 'dev']
        self.data_dir = config.data_dir
        self.num_past_frames = config.num_past_frames
        self.num_future_frames = config.num_future_frames
        self.with_metadata = with_metadata
        df_records = pd.read_csv(config.csv_path)
        df_records = df_records[df_records['split'] == config.split]
        self.records = df_records.to_dict(orient="records")
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __len__(self):
        return len(self.records)

    def _extract_pose_windows(self, pose):
        """
        Extract past and future windows from the pose object.
        Returns a dictionary with pose tensors and metadata.
        """
        total_frames = len(pose.body.data)
        pivot_frame = random.randint(0, total_frames - 1)

        input_start = max(0, pivot_frame - self.num_past_frames)
        input_pose = pose.body[input_start:pivot_frame].torch()
        target_end = min(total_frames, pivot_frame + self.num_future_frames)
        target_pose = pose.body[pivot_frame:target_end].torch()

        return {
            "input_data": input_pose.data.zero_filled(),
            "target_data": target_pose.data.zero_filled(),
            "input_mask": input_pose.data.mask,
            "target_mask": target_pose.data.mask,
            "target_length": torch.tensor([len(target_pose.data)], dtype=torch.float32),
            "pivot_frame": pivot_frame,
            "target_end": target_end,
            "total_frames": total_frames,
        }

    def _process_signwriting_image(self, text: str) -> torch.Tensor:
        pil_img = signwriting_to_clip_image(text)
        return self.clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)

    def _build_sample_dict(self, info: dict):
        sample = {
            "data": info["target_data"],
            "conditions": {
                "input_pose": info["input_data"],
                "input_mask": info["input_mask"],
                "target_mask": info["target_mask"],
                "sign_image": info["sign_img"],
            },
            "id": info["rec"].get("id", os.path.basename(info["rec"]["pose"])),
            "length_target": info["target_length"],
        }

        if self.with_metadata:
            meta = {
                "total_frames": info["total_frames"],
                "sample_start": info["pivot_frame"],
                "sample_end": info["target_end"],
                "orig_start": info["rec"].get("start", 0),
                "orig_end": info["rec"].get("end", info["total_frames"]),
            }
            sample["metadata"] = {
                k: torch.tensor([v], dtype=torch.long)
                for k, v in meta.items()
            }

        return sample

    def __getitem__(self, idx):
        rec = self.records[idx]
        pose_path = os.path.join(self.data_dir, rec["pose"])

        if not os.path.isfile(pose_path):
            return self[random.randint(0, len(self.records) - 1)]

        with open(pose_path, "rb") as f:
            raw = Pose.read(
                f,
                start_time=rec.get("start") or None,
                end_time=rec.get("end") or None
            )

        pose = normalize_mean_std(raw)
        window = self._extract_pose_windows(pose)
        sign_img = self._process_signwriting_image(rec.get("text", ""))

        return self._build_sample_dict({
            **window,
            "sign_img": sign_img,
            "rec": rec,
        })

def get_num_workers():
    """
    Determine appropriate number of workers based on CPU availability.
    """
    cpu_count = os.cpu_count()
    return 0 if cpu_count is None or cpu_count <= 1 else cpu_count

def main():
    config = DatasetConfig(
        data_dir="/scratch/yayun/pose_data/raw_poses",
        csv_path="/scratch/yayun/pose_data/data.csv",
        num_past_frames=40,
        num_future_frames=20,
        split='train'
    )

    dataset = DynamicPosePredictionDataset(
        config=config,
        with_metadata=True,
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

# if __name__ == "__main__":
#     main()
