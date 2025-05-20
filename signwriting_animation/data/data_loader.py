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
    Dataset for dynamic on-the-fly sampling of normalized pose data
    with SignWriting condition images and scalar metadata padded as length-1 tensors.
    """
    def __init__(self,data_dir,csv_path,past_frames=40,future_frames=20,with_metadata=True):
        super().__init__()
        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.with_metadata = with_metadata

        df = pd.read_csv(csv_path)
        self.records = df.to_dict(orient='records')

        # Initialize CLIP processor for signwriting images
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        rec = self.records[idx]
        start = rec.get("start", 0)
        total = rec.get(
            "end", start + self.num_past_frames + self.num_future_frames
        )
        end = min(start + self.num_past_frames + self.num_future_frames, total)

        # Load & normalize pose bytes
        pose_path = os.path.join(self.data_dir, rec["pose"])
        with open(pose_path, "rb") as f:
            raw = f.read()
        pose = Pose.read(raw)
        pose = normalize_mean_std(pose)

        data = pose.body.data.data           
        mask = torch.tensor(pose.body.data.mask, dtype=torch.bool)

        # Slice sequences
        input_pose = data[max(start, 0): start + self.num_past_frames]
        target_pose = data[start + self.num_past_frames : start + self.num_past_frames + self.num_future_frames]

        # Invert masks (True=visible)
        input_mask = torch.logical_not(mask[max(start, 0): start + self.num_past_frames])
        target_mask = torch.logical_not(mask[start + self.num_past_frames : start + self.num_past_frames + self.num_future_frames])

        # Render SignWriting to PIL
        sw_text = rec.get("text", "")
        if sw_text:
            pil_img = signwriting_to_image(sw_text)
        else:
            pil_img = Image.new("RGB", (224, 224), color=(0, 0, 0))

        # CLIP preprocessing → tensor [1,3,224,224]
        clip_inputs = self.clip_processor(images=pil_img, return_tensors="pt")
        sign_img = clip_inputs.pixel_values.squeeze(0)

        sample = {
            "data": torch.tensor(target_pose, dtype=torch.float32),
            "conditions": {
                "input_pose": torch.tensor(input_pose, dtype=torch.float32),
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

if __name__ == '__main__':
    data_dir = "/scratch/yayun/pose_data/raw_poses"
    csv_path = os.path.join(os.path.dirname(data_dir), "data.csv")
    
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
        num_workers=4,
        pin_memory=False,
    )
    batch = next(iter(loader))
    print('Batch:', batch['data'].shape)
    print('Input pose:', batch['conditions']['input_pose'].shape)
    print("Input mask:", batch['conditions']['input_mask'].shape)
    print("Target mask:", batch['conditions']['target_mask'].shape)
    print("Sign image:", batch['conditions']['sign_image'].shape)
    if 'metadata' in batch:
        for k, v in batch['metadata'].items():
            print(f"Metadata {k}:", v.shape)
