#dataloader
import os
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from pose_format import Pose

class PosePredictionDataset(Dataset):
    def __init__(self, data_dir, segmentation_csv, past_frames=40, future_frames=20, with_metadata=False):
        self.data_dir = data_dir
        self.with_metadata = with_metadata
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.samples = []

        df = pd.read_csv(segmentation_csv)
        for _, row in df.iterrows():
            pose_file = row["pose"]
            sample_path = os.path.join(data_dir, "raw_poses", pose_file)
            if not os.path.exists(sample_path):
                continue
            self.samples.append({
                "path": sample_path,
                "id": pose_file,
                "start": int(row["start"]),
                "end": int(row["end"])
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        with open(item["path"], "rb") as f:
            pose = Pose.read(f.read())

        data = pose.body.data.data
        mask = pose.body.data.mask

        clip = data[item["start"]:item["end"]]
        clip_mask = mask[item["start"]:item["end"]]

        input_pose = clip[:self.past_frames]
        target_pose = clip[self.past_frames:]
        input_mask = clip_mask[:self.past_frames]

        result = {
            "input_pose": torch.tensor(input_pose, dtype=torch.float32),
            "target_pose": torch.tensor(target_pose, dtype=torch.float32),
            "mask": torch.tensor(input_mask, dtype=torch.bool),
            "id": item["id"]
        }

        if self.with_metadata:
            result["metadata"] = {
                "total_frames": item["end"] - item["start"]
            }

        return result

def collate_pose_fn(batch):
    input_poses = [item["input_pose"] for item in batch]
    target_poses = [item["target_pose"] for item in batch]
    masks = [item["mask"] for item in batch]

    input_padded = pad_sequence(input_poses, batch_first=True)
    target_padded = pad_sequence(target_poses, batch_first=True)
    mask_padded = pad_sequence(masks, batch_first=True)

    return {
        "input_pose": input_padded,
        "target_pose": target_padded,
        "mask": mask_padded,
        "id": [item["id"] for item in batch],
        "metadata": [item.get("metadata", {}) for item in batch]
    }

if __name__ == "__main__":
    data_dir = "/scratch/yayun/pose_data"
    segmentation_csv = os.path.join(data_dir, "data_segmentation.csv")

    dataset = PosePredictionDataset(data_dir, segmentation_csv, past_frames=40, future_frames=20, with_metadata=True)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=collate_pose_fn, num_workers=0)

    for batch in loader:
        print("Loaded batch")
        print("Input Pose:", batch["input_pose"].shape)
        print("Target Pose:", batch["target_pose"].shape)
        print("Mask:", batch["mask"].shape)
        print("IDs:", batch["id"])
        break
