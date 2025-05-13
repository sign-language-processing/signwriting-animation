import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import random
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from pose_format.pose import Pose
from pose_anonymization.data.normalization import normalize_mean_std
from signwriting.visualizer.visualize import signwriting_to_image

class DynamicPosePredictionDataset(Dataset):
    """
    Dataset for dynamic on-the-fly sampling of normalized pose data
    with SignWriting condition images and scalar metadata padded as length-1 tensors.
    """
    def __init__(self,data_dir,segmentation_csv,past_frames=40,future_frames=20,with_metadata=True):
        super().__init__()
        self.data_dir = data_dir
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.with_metadata = with_metadata

        df = pd.read_csv(segmentation_csv)
        self.records = df.to_dict(orient='records')

        raw_dir = os.path.join(data_dir, 'raw_poses')
        self.pose_paths = []
        for rec in self.records:
            fname = rec['pose']
            if not fname.endswith('.pose'):
                fname += '.pose'
            self.pose_paths.append(os.path.join(raw_dir, fname))

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        path = self.pose_paths[idx]
        rec = self.records[idx]
        fname = os.path.basename(path)

        with open(path, 'rb') as pf:
            raw = pf.read()
        pose = Pose.read(raw)

        pose = normalize_mean_std(pose)

        arr = pose.body.data.data  # shape [T, V, C]
        msk = pose.body.data.mask  # same shape
        total = arr.shape[0]
        window = self.past_frames + self.future_frames

        min_start = 1 - self.past_frames
        max_start = total - self.future_frames
        start = random.randint(min_start, max_start)
        end = start + window

        # slice with zero padding for negative start
        def slice_and_pad(x, start, length):
            x = torch.tensor(x)
            if start < 0:
                pad = torch.zeros((-start,) + x.shape[1:], dtype=x.dtype)
                seq = x[0:start+length]
                return torch.cat([pad, seq], dim=0)
            else:
                return x[start:start+length]

        input_pose = slice_and_pad(arr, start, self.past_frames)
        target_pose = slice_and_pad(arr, start + self.past_frames, self.future_frames)

        # invert masks (0=visible, 1=masked)
        input_mask = slice_and_pad(msk, start, self.past_frames).to(torch.bool)
        input_mask = ~input_mask
        target_mask = slice_and_pad(msk, start + self.past_frames, self.future_frames).to(torch.bool)
        target_mask = ~target_mask

        sw_text = rec.get('text', '')
        if sw_text:
            pil = signwriting_to_image(sw_text)
            img = torch.tensor(pil.convert('L'), dtype=torch.float32).unsqueeze(0)
        else:
            img = torch.zeros(1, 128, 128, dtype=torch.float32)

        meta = {}
        if self.with_metadata:
            for k,v in {
                'total_frames': total,
                'sample_start': start,
                'sample_end': end,
                'orig_start': rec.get('start', 0),
                'orig_end':   rec.get('end', total)
            }.items():
                meta[k] = torch.tensor([v], dtype=torch.long)

        sample = {
            'data': target_pose,
            'conditions': {
                'input_pose': input_pose,
                'input_mask': input_mask,
                'target_mask': target_mask,
                'sign_image': img
            },
            'id': fname
        }
        if meta:
            sample['metadata'] = meta
        return sample

if __name__ == '__main__':
    data_dir = '/scratch/yayun/pose_data'
    csv = os.path.join(data_dir, 'data.csv')
    dataset = DynamicPosePredictionDataset(data_dir, csv)
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0
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
