import os
import random
from torch.utils.data import Dataset, DataLoader
from pose_format import Pose
from pose_format.torch.masked.collator import zero_pad_collator
import pandas as pd
import torch  # added for tensor conversion

class DynamicPosePredictionDataset(Dataset):
    """
    Dataset for dynamic on-the-fly sampling of MaskedTorchTensor pose data.

    Each sample returns a dict:
      - 'target_pose': torch.Tensor of the future frames to predict
      - 'conditions': dict containing:
          'input_pose': torch.Tensor of past frames as input
          'input_mask': torch.BoolTensor mask for input frames
          'target_mask': torch.BoolTensor mask for target frames
      - 'id': filename identifier
      - optional 'metadata': dict with keys 'total_frames', 'sample_start', 'sample_end', and original CSV 'start','end'
    """
    def __init__(self,
                 data_dir,
                 segmentation_csv=None,
                 past_frames=40,
                 future_frames=20,
                 with_metadata=False):
        super().__init__()
        self.data_dir = data_dir
        self.past_frames = past_frames
        self.future_frames = future_frames
        self.with_metadata = with_metadata
        self.segmentation = {}

        if with_metadata and segmentation_csv:
            df = pd.read_csv(segmentation_csv)
            for _, row in df.iterrows():
                self.segmentation[row['pose']] = {
                    'start': int(row['start']),
                    'end':   int(row['end'])
                }
        raw_dir = os.path.join(data_dir, 'raw_poses')
        self.pose_paths = [os.path.join(raw_dir, f)
                           for f in os.listdir(raw_dir)
                           if os.path.isfile(os.path.join(raw_dir, f))]

    def __len__(self):
        return len(self.pose_paths)

    def __getitem__(self, idx):
        path = self.pose_paths[idx]
        fname = os.path.basename(path)
        with open(path, 'rb') as f:
            pose = Pose.read(f.read())

        # Extract raw numpy arrays
        arr = pose.body.data.data  # shape [T, V, D]
        msk = pose.body.data.mask  # shape [T, V]
        total = arr.shape[0]
        window = self.past_frames + self.future_frames
        if total < window:
            return self.__getitem__(random.randint(0, total-1))

        start = random.randint(0, total-window)
        end = start + window
        # Slices
        input_seq = torch.tensor(arr[start:start+self.past_frames], dtype=torch.float32)
        input_m = torch.tensor(msk[start:start+self.past_frames], dtype=torch.bool)
        future_seq = torch.tensor(arr[start+self.past_frames:end], dtype=torch.float32)

        sample = {
            'target_pose': future_seq,  # normalized target clip
            'conditions': {
                'input_pose': input_seq,
                'input_mask': input_m,
                'target_mask': torch.tensor(msk[start+self.past_frames:end], dtype=torch.bool)
            },
            'id': fname
        }
        if self.with_metadata:
            meta = {'total_frames': total,
                    'sample_start': start,
                    'sample_end': end}
            if fname in self.segmentation:
                meta.update(self.segmentation[fname])
            sample['metadata'] = meta
        return sample

if __name__ == '__main__':
    data_dir = '/scratch/yayun/pose_data'
    segmentation_csv = os.path.join(data_dir, 'data.csv')

    dataset = DynamicPosePredictionDataset(
        data_dir,
        segmentation_csv=segmentation_csv,
        past_frames=40,
        future_frames=20,
        with_metadata=True
    )

    # Use zero_pad_collator
    dataloader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=zero_pad_collator,
        num_workers=0
    )

    batch = next(iter(dataloader))
    print("Batch size (target poses):", batch['target_pose'].shape[0])
    print("Target pose sequence:", batch['target_pose'].shape)
    print("Input pose sequence:", batch['conditions']['input_pose'].shape)
    print("Input mask:", batch['conditions']['input_mask'].shape)
    print("Target mask:", batch['conditions']['target_mask'].shape)
    if 'metadata' in batch:
        print("Metadata:")
        for key, val in batch['metadata'].items():
            print(f"  {key}:", val)
