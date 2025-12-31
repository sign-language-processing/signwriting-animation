import os
import torch
import torch.nn.functional as F
import numpy as np
import lightning as pl
from torch.utils.data import DataLoader

from pose_format import Pose
from pose_format.numpy.pose_body import NumPyPoseBody
from pose_format.utils.generic import reduce_holistic
from pose_format.torch.masked.collator import zero_pad_collator
from pose_anonymization.data.normalization import unshift_hands

from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_animation.diffusion.lightning_module import LitDiffusion, sanitize_btjc, mean_frame_disp


def tensor_to_pose(t_btjc: torch.Tensor,
                   header,
                   ref_pose: Pose,
                   scale_to_ref: bool = True) -> Pose:
    """
    Convert tensor prediction to Pose format for visualization.

    Args:
        t_btjc: Tensor [B, T, J, C] in normalized space
        header: Pose header with component definitions
        ref_pose: Reference pose for scaling and alignment
        scale_to_ref: Whether to scale to reference variance

    Returns:
        Pose object ready for visualization
    """
    if t_btjc.dim() == 4:
        t = t_btjc[0]
    else:
        t = t_btjc

    t_np = t.detach().cpu().numpy().astype(np.float32)

    # Create pose object
    arr = t_np[:, None, :, :]
    T = arr.shape[0]
    conf = np.ones((T, 1, arr.shape[2]), dtype=np.float32)
    fps = ref_pose.body.fps

    body = NumPyPoseBody(fps=fps, data=arr, confidence=conf)
    pose_obj = Pose(header=header, body=body)

    # Fix hand positions (must be before scaling!)
    unshift_hands(pose_obj)

    if scale_to_ref:
        # Get reference data
        T_pred = t_np.shape[0]
        T_ref_total = ref_pose.body.data.shape[0]
        future_start = max(0, T_ref_total - T_pred)
        ref_arr = np.asarray(
            ref_pose.body.data[future_start:future_start+T_pred, 0], 
            dtype=np.float32
        )

        # Scale to reference variance
        def _var(a):
            center = a.mean(axis=(0, 1), keepdims=True)
            return float(((a - center) ** 2).mean())

        pose_data = pose_obj.body.data[:, 0, :, :]
        var_input = _var(pose_data)
        var_ref = _var(ref_arr)

        if var_input > 1e-8:
            scale = np.sqrt(var_ref / var_input)
            pose_obj.body.data = pose_obj.body.data * scale

        # Align center
        pose_data = pose_obj.body.data[:, 0, :, :].reshape(-1, 3)
        input_center = pose_data.mean(axis=0)
        ref_center = ref_arr.reshape(-1, 3).mean(axis=0)
        pose_obj.body.data = pose_obj.body.data + (ref_center - input_center)

    return pose_obj


def train_minimal():
    """
    Train model on 4 samples for quick validation.
    """
    pl.seed_everything(42)

    # === Configuration ===
    data_dir = "/home/yayun/data/pose_data/"
    csv_path = "/home/yayun/data/signwriting-animation/data_fixed.csv"
    stats_path = f"{data_dir}/mean_std_178_with_preprocess.pt"
    out_dir = "logs/minimal"

    NUM_SAMPLES = 4
    MAX_EPOCHS = 500
    BATCH_SIZE = 4

    os.makedirs(out_dir, exist_ok=True)

    print("=" * 70)
    print("4 SAMPLE MINIMAL TRAINING")
    print("=" * 70)

    # === Dataset ===
    base_ds = DynamicPosePredictionDataset(
        data_dir=data_dir,
        csv_path=csv_path,
        num_past_frames=40,
        num_future_frames=20,
        with_metadata=True,
        split="train",
    )

    class SubsetDataset(torch.utils.data.Dataset):
        def __init__(self, base, indices):
            self.samples = [base[i] for i in indices]
        def __len__(self):
            return len(self.samples)
        def __getitem__(self, idx):
            return self.samples[idx]

    train_ds = SubsetDataset(base_ds, list(range(NUM_SAMPLES)))
    train_loader = DataLoader(
        train_ds, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=zero_pad_collator
    )

    # Get dimensions
    sample = train_ds[0]["data"]
    if hasattr(sample, 'zero_filled'):
        sample = sample.zero_filled()
    if hasattr(sample, 'tensor'):
        sample = sample.tensor
    num_joints = sample.shape[-2]
    num_dims = sample.shape[-1]
    future_len = sample.shape[0]

    print(f"\nDataset: {NUM_SAMPLES} samples")
    print(f"Dimensions: J={num_joints}, D={num_dims}, T={future_len}")
    print(f"Epochs: {MAX_EPOCHS}")

    # === Create Model ===
    model = SignWritingToPoseDiffusion(
        num_keypoints=num_joints,
        num_dims_per_keypoint=num_dims,
        t_past=40,
        t_future=future_len,
    )

    lit_model = LitDiffusion(
        num_keypoints=num_joints,
        num_dims=num_dims,
        stats_path=stats_path,
        lr=1e-3,
        diffusion_steps=8,
        vel_weight=1.0,
        t_past=40,
        t_future=future_len,
    )
    lit_model.model = model

    # === Train ===
    trainer = pl.Trainer(
        max_epochs=MAX_EPOCHS,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        enable_checkpointing=False,
        logger=False,
        enable_progress_bar=False,
    )
    trainer.fit(lit_model, train_loader)

    # === Inference ===
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    lit_model = lit_model.to(device)
    lit_model.eval()

    test_batch = zero_pad_collator([train_ds[0]])
    cond = test_batch["conditions"]

    past_raw = sanitize_btjc(cond["input_pose"][:1]).to(device)
    sign = cond["sign_image"][:1].float().to(device)
    gt_raw = sanitize_btjc(test_batch["data"][:1]).to(device)

    past_norm = lit_model.normalize(past_raw)
    gt_norm = lit_model.normalize(gt_raw)

    # DDPM sampling
    with torch.no_grad():
        past_bjct = lit_model.btjc_to_bjct(past_norm)
        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        class _Wrapper(torch.nn.Module):
            def __init__(self, model, past, sign):
                super().__init__()
                self.model, self.past, self.sign = model, past, sign
            def forward(self, x, t, **kwargs):
                return self.model(x, t, self.past, self.sign)

        wrapped = _Wrapper(lit_model.model, past_bjct, sign)

        pred_bjct = lit_model.diffusion.p_sample_loop(
            model=wrapped,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )
        pred_norm = lit_model.bjct_to_btjc(pred_bjct)

    # === Metrics ===
    mse = F.mse_loss(pred_norm, gt_norm).item()
    disp_pred = mean_frame_disp(pred_norm)
    disp_gt = mean_frame_disp(gt_norm)
    disp_ratio = disp_pred / (disp_gt + 1e-8)

    pred_np = pred_norm[0].cpu().numpy()
    gt_np = gt_norm[0].cpu().numpy()
    per_joint_err = np.sqrt(((pred_np - gt_np) ** 2).sum(-1))
    mpjpe = per_joint_err.mean()
    pck_01 = (per_joint_err < 0.1).mean() * 100

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"  Disp Ratio: {disp_ratio:.4f} (ideal = 1.0)")
    print(f"  MPJPE: {mpjpe:.6f}")
    print(f"  PCK@0.1: {pck_01:.1f}%")
    print(f"  MSE: {mse:.6f}")

    # === Save Poses ===
    ref_path = base_ds.records[0]["pose"]
    if not os.path.isabs(ref_path):
        ref_path = os.path.join(data_dir, ref_path)

    with open(ref_path, "rb") as f:
        ref_pose = Pose.read(f)
    ref_pose = reduce_holistic(ref_pose)
    if "POSE_WORLD_LANDMARKS" in [c.name for c in ref_pose.header.components]:
        ref_pose = ref_pose.remove_components(["POSE_WORLD_LANDMARKS"])

    gt_unnorm = lit_model.unnormalize(gt_norm)
    pred_unnorm = lit_model.unnormalize(pred_norm)

    gt_pose = tensor_to_pose(gt_unnorm, ref_pose.header, ref_pose)
    pred_pose = tensor_to_pose(pred_unnorm, ref_pose.header, ref_pose)

    with open(f"{out_dir}/gt.pose", "wb") as f:
        gt_pose.write(f)
    with open(f"{out_dir}/pred.pose", "wb") as f:
        pred_pose.write(f)

    print(f"\nPose files saved to: {out_dir}/")
    print("=" * 70)
    print("✅ TRAINING COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    train_minimal()
