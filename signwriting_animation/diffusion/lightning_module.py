# pylint: disable=too-many-locals,too-many-instance-attributes,invalid-name
import os
import torch
from torch import nn
import torch.nn.functional as F
import lightning as pl
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW
from CAMDM.diffusion.gaussian_diffusion import (
    GaussianDiffusion, ModelMeanType, ModelVarType, LossType
)

from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


def sanitize_btjc(x: torch.Tensor) -> torch.Tensor:
    """
    Sanitize pose tensor to ensure BTJC format [Batch, Time, Joints, Coords].
    
    Handles various input formats:
    - MaskedTensor with zero_filled() method
    - Tensor with extra dimensions
    - Tensors with swapped J/C dimensions
    
    Args:
        x: Input tensor in various formats
        
    Returns:
        Tensor in [B, T, J, C] format with C=3
        
    Raises:
        ValueError: If tensor cannot be converted to valid BTJC format
    """
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if hasattr(x, "tensor"):
        x = x.tensor
    if x.dim() == 5:
        x = x[:, :, 0]
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    if x.shape[-1] != 3 and x.shape[-2] == 3:
        x = x.permute(0, 1, 3, 2)
    if x.shape[-1] != 3:
        raise ValueError(f"sanitize_btjc: last dim must be C=3, got {x.shape}")
    return x.contiguous().float()


def _btjc_to_tjc_list(x_btjc: torch.Tensor, mask_bt: torch.Tensor) -> list:
    """
    Convert batched BTJC tensor to list of variable-length TJC tensors.
    
    Uses mask to determine actual sequence length for each batch item.
    
    Args:
        x_btjc: Batched pose tensor [B, T, J, C]
        mask_bt: Binary mask [B, T] indicating valid frames
        
    Returns:
        List of tensors, each [T_i, J, C] where T_i is the valid length
    """
    x_btjc = sanitize_btjc(x_btjc)
    batch_size, seq_len, _, _ = x_btjc.shape
    mask_bt = (mask_bt > 0.5).float()
    seqs = []
    for b in range(batch_size):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, seq_len))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs


@torch.no_grad()
def masked_dtw(pred_btjc: torch.Tensor, tgt_btjc: torch.Tensor, mask_bt: torch.Tensor) -> torch.Tensor:
    """
    Compute Dynamic Time Warping distance with masking.
    
    Uses pose_evaluation's DTW implementation for accurate distance measurement.
    Falls back to MSE if DTW is unavailable.
    
    Args:
        pred_btjc: Predicted poses [B, T, J, C]
        tgt_btjc: Target poses [B, T, J, C]
        mask_bt: Valid frame mask [B, T]
        
    Returns:
        Mean DTW distance across batch
    """
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts = _btjc_to_tjc_list(tgt_btjc, mask_bt)

    try:
        dtw_metric = PE_DTW()
    except (ImportError, RuntimeError):
        # Fallback to MSE if DTW unavailable
        pred = sanitize_btjc(pred_btjc)
        tgt = sanitize_btjc(tgt_btjc)
        t_max = min(pred.size(1), tgt.size(1))
        return torch.mean((pred[:, :t_max] - tgt[:, :t_max]) ** 2)

    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")[:, None, :, :]
        gv = g.detach().cpu().numpy().astype("float32")[:, None, :, :]
        vals.append(float(dtw_metric.get_distance(pv, gv)))

    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


def mean_frame_disp(x_btjc: torch.Tensor) -> float:
    """
    Compute mean per-frame displacement (motion magnitude).
    
    This is a critical metric for detecting motion collapse.
    Healthy models should have pred_disp / gt_disp ≈ 1.0.
    
    Args:
        x_btjc: Pose tensor [B, T, J, C]
        
    Returns:
        Mean absolute displacement between consecutive frames
    """
    x = sanitize_btjc(x_btjc)
    if x.size(1) < 2:
        return 0.0
    v = x[:, 1:] - x[:, :-1]
    return v.abs().mean().item()


def cosine_beta_schedule(timesteps: int, s: float = 0.008) -> torch.Tensor:
    """
    Create cosine beta schedule for diffusion process.
    
    Cosine schedule provides better training dynamics than linear schedule,
    with more gradual noise addition at the start and end.
    
    Args:
        timesteps: Number of diffusion steps
        s: Small offset for numerical stability
        
    Returns:
        Beta values for each timestep
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0.0001, 0.9999)


class _ConditionalWrapper(nn.Module):
    """
    Wrapper to adapt model interface for GaussianDiffusion.
    
    GaussianDiffusion expects: model(x, t) -> output
    model expects: model(x, t, past, sign_img) -> output
    
    This wrapper fixes the conditions and provides the expected interface.
    """

    def __init__(self, base_model: nn.Module, past_bjct: torch.Tensor, sign_img: torch.Tensor):
        super().__init__()
        self.base_model = base_model
        self.past_bjct = past_bjct
        self.sign_img = sign_img

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """Forward with fixed conditions."""
        return self.base_model(x, t, self.past_bjct, self.sign_img)


class LitDiffusion(pl.LightningModule):
    """
    PyTorch Lightning module for diffusion training.
    
    Args:
        num_keypoints: Number of pose keypoints (default: 178 for holistic)
        num_dims: Dimensions per keypoint (default: 3 for x,y,z)
        lr: Learning rate
        stats_path: Path to normalization statistics file
        diffusion_steps: Number of diffusion timesteps (default: 8)
        vel_weight: Weight for velocity loss term
        acc_weight: Weight for acceleration loss term
        t_past: Number of past frames for conditioning
        t_future: Number of future frames to predict
    """

    def __init__(
        self,
        num_keypoints: int = 178,
        num_dims: int = 3,
        lr: float = 1e-4,
        stats_path: str = "/home/yayun/data/pose_data/mean_std_178_with_preprocess.pt",
        diffusion_steps: int = 8,
        vel_weight: float = 1.0,
        acc_weight: float = 0.5,
        t_past: int = 40,
        t_future: int = 20,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.diffusion_steps = diffusion_steps
        self.vel_weight = vel_weight
        self.acc_weight = acc_weight
        self._step_count = 0

        # Load normalization statistics
        stats = torch.load(stats_path, map_location="cpu")
        mean = stats["mean"].float().view(1, 1, -1, 3)
        std = stats["std"].float().view(1, 1, -1, 3)
        self.register_buffer("mean_pose", mean.clone())
        self.register_buffer("std_pose", std.clone())

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints,
            num_dims_per_keypoint=num_dims,
            t_past=t_past,
            t_future=t_future,
        )

        # Create Gaussian diffusion process with cosine schedule
        betas = cosine_beta_schedule(diffusion_steps).numpy()
        self.diffusion = GaussianDiffusion(
            betas=betas,
            model_mean_type=ModelMeanType.START_X,  # Predict x0 directly
            model_var_type=ModelVarType.FIXED_SMALL,
            loss_type=LossType.MSE,
            rescale_timesteps=False,
        )

        self.lr = lr
        self.train_logs = {"loss": [], "mse": [], "vel": [], "acc": [], "disp_ratio": []}

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalize pose to approximately zero mean and unit variance."""
        return (x - self.mean_pose) / (self.std_pose + 1e-6)

    def unnormalize(self, x: torch.Tensor) -> torch.Tensor:
        """Denormalize pose back to original scale."""
        return x * self.std_pose + self.mean_pose

    @staticmethod
    def btjc_to_bjct(x: torch.Tensor) -> torch.Tensor:
        """Convert [B,T,J,C] to [B,J,C,T] format for diffusion."""
        return x.permute(0, 2, 3, 1).contiguous()

    @staticmethod
    def bjct_to_btjc(x: torch.Tensor) -> torch.Tensor:
        """Convert [B,J,C,T] to [B,T,J,C] format for output."""
        return x.permute(0, 3, 1, 2).contiguous()

    # pylint: disable=arguments-differ
    def training_step(self, batch: dict, _batch_idx: int) -> torch.Tensor:
        """
        Single training step for diffusion model.
        
        Implements the diffusion training objective:
        1. Sample random timestep t
        2. Add noise to get x_t from x_0
        3. Predict x_0 from x_t
        4. Compute loss (MSE + velocity + acceleration)
        
        Args:
            batch: Dict with 'data' (target) and 'conditions' (past, sign)
            batch_idx: Batch index (unused)
            
        Returns:
            Total loss value
        """
        debug = self._step_count == 0 or self._step_count % 100 == 0

        # Extract data from batch
        cond_raw = batch["conditions"]
        gt_btjc = sanitize_btjc(batch["data"])
        past_btjc = sanitize_btjc(cond_raw["input_pose"])
        sign_img = cond_raw["sign_image"].float()

        # Normalize to standard space
        gt_norm = self.normalize(gt_btjc)
        past_norm = self.normalize(past_btjc)

        batch_size = gt_norm.shape[0]
        device = gt_norm.device

        # Convert to BJCT format for diffusion
        gt_bjct = self.btjc_to_bjct(gt_norm)
        past_bjct = self.btjc_to_bjct(past_norm)

        # Sample random diffusion timestep
        timestep = torch.randint(0, self.diffusion_steps, (batch_size,), device=device, dtype=torch.long)

        # Forward diffusion: add noise to get x_t
        noise = torch.randn_like(gt_bjct)
        x_noisy = self.diffusion.q_sample(gt_bjct, timestep, noise=noise)

        # Model prediction: predict x_0 from x_t
        pred_x0_bjct = self.model(x_noisy, timestep, past_bjct, sign_img)

        # === Loss Computation ===
        loss_mse = F.mse_loss(pred_x0_bjct, gt_bjct)

        # Velocity loss for motion smoothness
        pred_vel = pred_x0_bjct[..., 1:] - pred_x0_bjct[..., :-1]
        gt_vel = gt_bjct[..., 1:] - gt_bjct[..., :-1]
        loss_vel = F.mse_loss(pred_vel, gt_vel)

        # Acceleration loss for motion naturalness
        loss_acc = torch.tensor(0.0, device=device)
        if pred_vel.size(-1) > 1:
            pred_acc = pred_vel[..., 1:] - pred_vel[..., :-1]
            gt_acc = gt_vel[..., 1:] - gt_vel[..., :-1]
            loss_acc = F.mse_loss(pred_acc, gt_acc)

        loss = loss_mse + self.vel_weight * loss_vel + self.acc_weight * loss_acc

        # Displacement ratio monitoring
        with torch.no_grad():
            pred_disp = pred_vel.abs().mean().item()
            gt_disp = gt_vel.abs().mean().item()
            disp_ratio = pred_disp / (gt_disp + 1e-8)

        # Debug logging
        if debug:
            print("\n" + "=" * 70)
            print(f"TRAINING STEP {self._step_count} (Frame-Independent)")
            print("=" * 70)
            print(f"  t range: [{timestep.min().item()}, {timestep.max().item()}]")
            print(f"  loss_mse: {loss_mse.item():.6f}")
            print(f"  loss_vel: {loss_vel.item():.6f}")
            print(f"  loss_acc: {loss_acc.item():.6f}")
            print(f"  disp_ratio: {disp_ratio:.4f} (ideal=1.0)")
            print(f"  TOTAL: {loss.item():.6f}")
            print("=" * 70)

        # Log metrics
        self.log_dict({
            "train/loss": loss,
            "train/mse": loss_mse,
            "train/vel": loss_vel,
            "train/acc": loss_acc,
            "train/disp_ratio": disp_ratio,
        }, prog_bar=True)

        # Store for plotting
        self.train_logs["loss"].append(loss.item())
        self.train_logs["mse"].append(loss_mse.item())
        self.train_logs["vel"].append(loss_vel.item())
        self.train_logs["acc"].append(loss_acc.item())
        self.train_logs["disp_ratio"].append(disp_ratio)

        self._step_count += 1
        return loss

    @torch.no_grad()
    def sample(self, past_btjc: torch.Tensor, sign_img: torch.Tensor, future_len: int = 20) -> torch.Tensor:
        """
        Generate pose sequence using DDPM sampling.
        
        Iteratively denoises from pure noise to generate the predicted
        future motion conditioned on past motion and SignWriting image.
        
        Args:
            past_btjc: Past motion context [B, T_past, J, C]
            sign_img: SignWriting condition images [B, 3, H, W]
            future_len: Number of frames to generate
            
        Returns:
            Predicted poses [B, T_future, J, C] in original (unnormalized) scale
        """
        self.eval()
        device = self.device

        # Prepare conditions
        past_raw = sanitize_btjc(past_btjc.to(device))
        past_norm = self.normalize(past_raw)
        past_bjct = self.btjc_to_bjct(past_norm)

        B, J, C, _ = past_bjct.shape
        target_shape = (B, J, C, future_len)

        # Wrap model for GaussianDiffusion interface
        wrapped_model = _ConditionalWrapper(self.model, past_bjct, sign_img)

        # DDPM sampling loop
        pred_bjct = self.diffusion.p_sample_loop(
            model=wrapped_model,
            shape=target_shape,
            clip_denoised=False,
            model_kwargs={"y": {}},
            progress=False,
        )

        pred_btjc = self.bjct_to_btjc(pred_bjct)
        return self.unnormalize(pred_btjc)


    def configure_optimizers(self):
        """Configure AdamW optimizer."""
        return torch.optim.AdamW(self.parameters(), lr=self.lr)

    def on_train_end(self):
        """Save training curves after training completes."""
        out_dir = "logs/diffusion"
        os.makedirs(out_dir, exist_ok=True)

        _, axes = plt.subplots(2, 2, figsize=(10, 8))

        axes[0, 0].plot(self.train_logs["loss"])
        axes[0, 0].set_title("Total Loss")

        axes[0, 1].plot(self.train_logs["mse"])
        axes[0, 1].set_title("MSE Loss")

        axes[1, 0].plot(self.train_logs["vel"])
        axes[1, 0].set_title("Velocity Loss")

        axes[1, 1].plot(self.train_logs["disp_ratio"])
        axes[1, 1].set_title("Displacement Ratio (ideal=1.0)")
        axes[1, 1].axhline(y=1.0, color='r', linestyle='--', alpha=0.5)

        plt.tight_layout()
        plt.savefig(f"{out_dir}/train_curve.png")
        print(f"[TRAIN CURVE] saved to {out_dir}/train_curve.png")
