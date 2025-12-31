# pylint: disable=too-many-instance-attributes,too-many-arguments,too-many-locals
import os
import math
import random
from typing import Literal, Optional
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.pose import Pose
from pose_format.utils.generic import reduce_holistic
from pose_anonymization.data.normalization import pre_process_pose
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image
from transformers import CLIPProcessor


def _coalesce_maybe_nan(x) -> Optional[int]:
    """
    Convert NaN/None values to None, otherwise return the value.
    
    Args:
        x: Value to check (can be None, NaN, or numeric)
        
    Returns:
        None if input is None/NaN, otherwise the input value
    """
    if x is None:
        return None
    if isinstance(x, float) and math.isnan(x):
        return None
    return x


class DynamicPosePredictionDataset(Dataset):
    """
    PyTorch Dataset for dynamic sampling of pose sequences conditioned on SignWriting.
    
    This dataset provides past and future pose windows for training diffusion models.
    Data is returned in raw (unnormalized) format - normalization is handled by the
    LightningModule to ensure consistency with precomputed statistics.
    
    Data Pipeline:
        Raw pose → reduce_holistic (586→178 keypoints) → pre_process_pose → return
        
    Note: This preprocessing pipeline must match the one used to generate the
          normalization statistics (mean_std_178_with_preprocess.pt).
    
    Args:
        data_dir: Root directory containing .pose files
        csv_path: Path to CSV file with pose metadata and SignWriting text
        num_past_frames: Number of past frames for conditioning (default: 60)
        num_future_frames: Number of future frames to predict (default: 30)
        with_metadata: Whether to include frame timing metadata (default: True)
        clip_model_name: HuggingFace model name for CLIP processor
        split: Data split to use ('train', 'dev', or 'test')
        use_reduce_holistic: Whether to reduce keypoints to 178 (default: True)
    """

    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        num_past_frames: int = 40,
        num_future_frames: int = 20,
        with_metadata: bool = True,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        split: Literal["train", "dev", "test"] = "train",
        use_reduce_holistic: bool = True,
    ):
        super().__init__()
        assert split in ["train", "dev", "test"], f"Invalid split: {split}"

        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.with_metadata = with_metadata
        self.use_reduce_holistic = use_reduce_holistic

        self.mean_std = None

        df_records = pd.read_csv(csv_path)
        df_records = df_records[df_records["split"] == split].reset_index(drop=True)
        self.records = df_records.to_dict(orient="records")

        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        """
        Load and process a single training sample.
        
        Returns a dictionary containing:
            - data: Future pose sequence [T_future, J, C] (target for prediction)
            - conditions:
                - input_pose: Past pose sequence [T_past, J, C] (conditioning)
                - input_mask: Validity mask for past poses [T_past]
                - target_mask: Validity mask for future poses [T_future]
                - sign_image: CLIP-processed SignWriting image [3, H, W]
            - id: Sample identifier
            - metadata: (optional) Frame timing information
            
        If the requested pose file is too short or corrupted, recursively tries
        the next sample to ensure training doesn't crash.
        """
        rec = self.records[idx]

        pose_path = os.path.join(self.data_dir, rec["pose"])
        if not pose_path.endswith(".pose"):
            pose_path += ".pose"

        start = _coalesce_maybe_nan(rec.get("start"))
        end = _coalesce_maybe_nan(rec.get("end"))

        if not os.path.exists(pose_path):
            raise FileNotFoundError(f"Pose file not found: {pose_path}")

        # Load raw pose data
        with open(pose_path, "rb") as f:
            raw = Pose.read(f)

        # Check if sequence is too short before preprocessing
        total_frames = len(raw.body.data)
        if total_frames < 5:
            print(f"[SKIP SHORT FILE] idx={idx} | total_frames={total_frames} | "
                  f"file={os.path.basename(pose_path)}")
            return self.__getitem__((idx + 1) % len(self.records))

        if self.use_reduce_holistic:
            raw = reduce_holistic(raw)
        raw = pre_process_pose(raw)
        pose = raw  # Keep in raw scale (no normalization)

        # Verify sequence is still valid after preprocessing
        total_frames = len(pose.body.data)
        if total_frames < 5:
            print(f"[SKIP SHORT CLIP] idx={idx} | total_frames={total_frames}")
            return self.__getitem__((idx + 1) % len(self.records))

        # Sample time windows intelligently
        if total_frames <= (self.num_past_frames + self.num_future_frames + 2):
            # Short sequence: use centered sampling to maximize data usage
            pivot_frame = total_frames // 2
            input_start = max(0, pivot_frame - self.num_past_frames // 2)
            target_end = min(total_frames, input_start + self.num_past_frames + self.num_future_frames)
        else:
            # Long sequence: random sampling with proper boundaries
            pivot_min = self.num_past_frames
            pivot_max = total_frames - self.num_future_frames
            pivot_frame = random.randint(pivot_min, pivot_max)
            input_start = pivot_frame - self.num_past_frames
            target_end = pivot_frame + self.num_future_frames

        # Extract pose windows
        input_pose = pose.body[input_start:pivot_frame].torch()
        target_pose = pose.body[pivot_frame:target_end].torch()

        # Debug logging for first few samples
        if idx < 3:
            print(f"[DEBUG SPLIT] idx={idx} | total={total_frames} | pivot={pivot_frame} | "
                  f"input={input_start}:{pivot_frame} ({input_pose.data.shape[0]}f) | "
                  f"target={pivot_frame}:{target_end} ({target_pose.data.shape[0]}f) | "
                  f"file={os.path.basename(pose_path)}")

        # Extract data and masks
        input_data = input_pose.data
        target_data = target_pose.data
        input_mask = input_pose.data.mask
        target_mask = target_pose.data.mask

        # Process SignWriting image through CLIP
        pil_img = signwriting_to_clip_image(rec.get("text", ""))
        sign_img = self.clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)

        # Build output sample
        sample = {
            "data": target_data,  # Future window (prediction target, unnormalized)
            "conditions": {
                "input_pose": input_data,   # Past window (conditioning, unnormalized)
                "input_mask": input_mask,   # Validity mask for past frames
                "target_mask": target_mask, # Validity mask for future frames
                "sign_image": sign_img,     # CLIP-processed SignWriting [3, H, W]
            },
            "id": rec.get("id", os.path.basename(rec["pose"])),
        }

        # Add optional metadata for analysis
        if self.with_metadata:
            meta = {
                "total_frames": total_frames,
                "sample_start": pivot_frame,
                "sample_end": pivot_frame + len(target_data),
                "orig_start": start or 0,
                "orig_end": end or total_frames,
            }
            sample["metadata"] = {
                k: torch.tensor([int(v)], dtype=torch.long)
                for k, v in meta.items()
            }

        return sample


def get_num_workers() -> int:
    """
    Determine appropriate number of DataLoader workers based on CPU availability.
    
    Returns:
        0 if CPU count is unavailable or ≤1, otherwise the CPU count
    """
    cpu_count = os.cpu_count()
    return 0 if cpu_count is None or cpu_count <= 1 else cpu_count


def main():
    """Test dataset loading and print sample batch statistics."""
    data_dir = "/home/yayun/data/pose_data"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"

    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=60,
        num_future_frames=30,
        with_metadata=True,
        split="train",
        use_reduce_holistic=True,
    )

    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=get_num_workers(),
        pin_memory=False,
    )

    # Load and inspect a batch
    batch = next(iter(loader))
    print("Batch shapes:")
    print(f"  Data (target): {batch['data'].shape}")
    print(f"  Input pose: {batch['conditions']['input_pose'].shape}")
    print(f"  Input mask: {batch['conditions']['input_mask'].shape}")
    print(f"  Target mask: {batch['conditions']['target_mask'].shape}")
    print(f"  Sign image: {batch['conditions']['sign_image'].shape}")

    # Check data range (should be unnormalized)
    data = batch["data"]
    if hasattr(data, "tensor"):
        data = data.tensor
    print("\nData statistics (should be in raw range):")
    print(f"  Min: {data.min().item():.4f}")
    print(f"  Max: {data.max().item():.4f}")
    print(f"  Mean: {data.mean().item():.4f}")
    print(f"  Std: {data.std().item():.4f}")

    if abs(data.mean().item()) < 0.1 and abs(data.std().item() - 1.0) < 0.2:
        print(" Warning: Data appears normalized (should be raw)")
    else:
        print(" Data is in raw range (correct)")

    if "metadata" in batch:
        print("\nMetadata:")
        for k, v in batch["metadata"].items():
            print(f"  {k}: {v.shape}")

#if __name__ == "__main__":
    #main()
