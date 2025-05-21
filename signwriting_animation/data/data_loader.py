import os
import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.pose import Pose
from pose_anonymization.data.normalization import normalize_mean_std
from signwriting.visualizer.visualize import signwriting_to_image
from transformers import CLIPProcessor
from PIL import Image

class DynamicPosePredictionDataset(Dataset):
    """
    A PyTorch Dataset for dynamic sampling of normalized pose sequences,
    conditioned on SignWriting images and optional scalar metadata.
    Each sample includes past and future pose segments, associated masks,
    and a CLIP-ready rendering of the SignWriting annotation.
    """
    def __init__(self,data_dir,csv_path,num_past_frames= 40,num_future_frames= 20, with_metadata= True,
                 clip_model_name: str = "openai/clip-vit-base-patch32",
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.with_metadata = with_metadata
        self.records = pd.read_csv(csv_path).to_dict(orient="records")

        # Initialize CLIP processor for signwriting images
        self.clip_processor = CLIPProcessor.from_pretrained(clip_model_name)

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        start = rec.get("start", 0)
        total = rec.get("end", start + self.num_past_frames + self.num_future_frames)
        end = min(start + self.num_past_frames + self.num_future_frames, total)

        # Load & normalize pose bytes
        pose_path = os.path.join(self.data_dir, rec["pose"])
        with open(pose_path, "rb") as f:
            raw = f.read()
        pose = Pose.read(raw)
        pose = normalize_mean_std(pose)

        # Slice the PoseBody, then convert to TorchPose and extract data+mask
        input_pose_torch = pose.body[max(start, 0): start + self.num_past_frames].torch()
        target_pose_torch = pose.body[start + self.num_past_frames: start + self.num_past_frames + self.num_future_frames].torch()

        input_data = input_pose_torch.data.zero_filled().float()
        input_mask = torch.logical_not(input_pose_torch.data.mask)
        target_data = target_pose_torch.data.zero_filled().float()
        target_mask = torch.logical_not(target_pose_torch.data.mask)
        
        # Render SignWriting to PIL 
        sw_text = rec.get("text")
        if not isinstance(sw_text, str) or not sw_text.strip():
            pil_img = Image.new("RGB", (224, 224), color=(0, 0, 0))
        else:
            pil_img = signwriting_to_image(sw_text) 
                
        # Ensure valid RGB PIL Image
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        pil_img = pil_img.resize((224, 224))  # Resize to CLIP expected size

        # CLIP preprocessing → [1,3,224,224]
        clip_inputs = self.clip_processor(images=[pil_img], return_tensors="pt")
        sign_img = clip_inputs.pixel_values.squeeze(0)

        sample = {
            "data": target_data,
            "conditions": {
                "input_pose": input_data,
                "input_mask": input_mask,
                "target_mask": target_mask,
                "sign_image": sign_img,
            },
            "id": rec.get("id", os.path.basename(pose_path)),
        }

        if self.with_metadata:
            meta = {
                "total_frames": total,
                "sample_start": start,
                "sample_end": end,
                "orig_start": rec.get("start", 0),
                "orig_end": rec.get("end", total),
            }
            sample["metadata"] = {
                k: torch.tensor([v], dtype=torch.long) for k, v in meta.items()
            }

        return sample

def main():
    data_dir = "/scratch/yayun/pose_data/raw_poses"
    csv_path = os.path.join(os.path.dirname(data_dir), "data.csv")

    # create dataset & loader
    dataset = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0,
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

