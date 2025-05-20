import os
# limit BLAS threads to avoid pthread_create failures
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pose_format.pose import Pose
from pose_anonymization.data.normalization import normalize_mean_std
from pose_format.torch.masked.collator import zero_pad_collator

# Use official signwriting_to_image only
from signwriting_evaluation.metrics.clip import signwriting_to_image
# define CLIP preprocess locally
from PIL import Image
import torchvision.transforms as T

_clip_transform = T.Compose([
    T.Resize(256),
    T.CenterCrop(224),
    T.ToTensor(),
    T.Normalize(
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711]
    )
])

def preprocess_for_clip(pil_img: Image.Image) -> torch.Tensor:
    """
    Convert PIL image to CLIP-ready tensor [1, 3, 224, 224].
    """
    return _clip_transform(pil_img.convert("RGB")).unsqueeze(0)

class DynamicPosePredictionDataset(Dataset):
    """
    Dataset for dynamic on-the-fly sampling of normalized pose data
    with SignWriting condition images and scalar metadata padded as length-1 tensors.
    """
    def __init__(
        self,
        data_dir: str,
        csv_path: str,
        num_past_frames: int = 40,
        num_future_frames: int = 20,
        with_metadata: bool = True,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.num_past_frames = num_past_frames
        self.num_future_frames = num_future_frames
        self.with_metadata = with_metadata

        df = pd.read_csv(csv_path)
        self.records = df.to_dict(orient='records')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        path = os.path.join(self.data_dir, 'raw_poses', rec['pose'])
        fname = os.path.basename(path)

        # load & normalize pose with raw bytes
        with open(path, 'rb') as f:
            pose = Pose.read(f.read())
        pose = normalize_mean_std(pose)

        data = pose.body.data.data    # [T, V, C]
        mask = pose.body.data.mask    # same shape
        total = len(data)
        window_size = self.num_past_frames + self.num_future_frames

        # dynamic start 
        start = random.randint(-self.num_past_frames, total - self.num_future_frames)

        input_pose = data[max(start, 0): start + self.num_past_frames]
        target_pose = data[start + self.num_past_frames: start + self.num_past_frames + self.num_future_frames]

        # invert masks
        input_mask = torch.tensor(mask[ max(start,0): start + self.num_past_frames ], dtype=torch.bool)
        target_mask = torch.tensor(mask[ start + self.num_past_frames:
                                         start + self.num_past_frames + self.num_future_frames ], dtype=torch.bool)
        input_mask = torch.logical_not(input_mask)
        target_mask = torch.logical_not(target_mask)

        # SignWriting → image → CLIP tensor
        sw_text = rec.get('text', '')
        if sw_text:
            pil_img = signwriting_to_image(sw_text)
            sign_img = preprocess_for_clip(pil_img)
        else:
            sign_img = torch.zeros(1, 3, 224, 224, dtype=torch.float32)

        sample = {
            'data': torch.tensor(target_pose, dtype=torch.float32),
            'conditions': {
                'input_pose': torch.tensor(input_pose, dtype=torch.float32),
                'input_mask': input_mask,
                'target_mask': target_mask,
                'sign_image': sign_img,
            },
            'id': fname,
        }

        if self.with_metadata:
            meta = {
                'total_frames': total,
                'sample_start': start,
                'sample_end': start + window_size,
                'orig_start': rec.get('start', 0),
                'orig_end': rec.get('end', total),
            }
            sample['metadata'] = { k: torch.tensor([v], dtype=torch.long)
                                   for k, v in meta.items() }

        return sample

if __name__ == '__main__':
    data_dir = '/scratch/yayun/pose_data'
    csv_path = os.path.join(data_dir, 'data.csv')

    dataset = DynamicPosePredictionDataset(
        data_dir,
        csv_path,
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
    print('Batch:', batch['data'].shape)
    print('Input pose:', batch['conditions']['input_pose'].shape)
    print('Input mask:', batch['conditions']['input_mask'].shape)
    print('Target mask:', batch['conditions']['target_mask'].shape)
    print('Sign image:', batch['conditions']['sign_image'].shape)
    if 'metadata' in batch:
        for k, v in batch['metadata'].items():
            print(f"Metadata {k}:", v.shape)




