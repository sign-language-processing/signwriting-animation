import os
import csv
import torch
import lightning as pl
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from pose_evaluation.metrics.dtw_metric import DTWDTAIImplementationDistanceMeasure as PE_DTW


def _to_dense(x):
    """
    Convert a potentially sparse or masked tensor to a dense, contiguous float32 tensor.
        - pose-format MaskedTensor (via .zero_filled())
        - sparse tensors (.to_dense())
        - dtype casting to float32
    Args:
        x (torch.Tensor or MaskedTensor): Input tensor of arbitrary type.
    Returns:
        torch.Tensor: Dense float32 tensor with contiguous memory layout.
    """
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    if x.dtype != torch.float32:
        x = x.float()
    return x.contiguous()

def sanitize_btjc(x):
    """Ensure tensor is [B,T,J,C]. Handle sparse or [B,T,P,J,C] inputs."""
    x = _to_dense(x)
    if x.dim() == 5:  # [B,T,P,J,C]
        x = x[:, :, 0, ...]
    if x.dim() != 4:
        raise ValueError(f"Expected [B,T,J,C], got {tuple(x.shape)}")
    return x

def btjc_to_bjct(x):  # [B,T,J,C] -> [B,J,C,T]
    """Permute tensor from [B, T, J, C] → [B, J, C, T]."""
    return x.permute(0, 2, 3, 1).contiguous()

def bjct_to_btjc(x):  # [B,J,C,T] -> [B,T,J,C]
    """Permute tensor from [B, J, C, T] → [B, T, J, C]."""
    return x.permute(0, 3, 1, 2).contiguous()

def masked_mse(pred_btjc, tgt_btjc, mask_bt):
    """
    Compute mean squared error over valid (masked) frames.
    Args:
        pred_btjc (torch.Tensor): Predicted poses [B, T, J, C].
        tgt_btjc (torch.Tensor): Target poses [B, T, J, C].
        mask_bt (torch.Tensor): Binary mask [B, T] where 1 indicates valid frames.
    Returns:
        torch.Tensor: Scalar loss value (float).
    """
    pred = sanitize_btjc(pred_btjc)
    tgt  = sanitize_btjc(tgt_btjc)

    Tm = min(pred.size(1), tgt.size(1), mask_bt.size(1))
    pred = pred[:, :Tm]
    tgt  = tgt[:,  :Tm]
    m4 = mask_bt[:, :Tm].float()[:, :, None, None]   # [B,T,1,1]

    diff2 = (pred - tgt) ** 2                        # [B,T,J,C]
    num = (diff2 * m4).sum()
    den = (m4.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den

def _btjc_to_tjc_list(x_btjc, mask_bt):
    """
    Convert batched [B,T,J,C] tensor into list of variable-length [T,J,C] sequences.
    Uses mask to trim valid frames for each sample.
    Args:
        x_btjc (torch.Tensor): Pose tensor [B, T, J, C].
        mask_bt (torch.Tensor): Frame validity mask [B, T].
    Returns:
        list[torch.Tensor]: List of [T, J, C] tensors (one per batch sample).
    """
    x_btjc = sanitize_btjc(x_btjc)
    B, T, J, C = x_btjc.shape
    seqs = []
    mask_bt = (mask_bt > 0.5).float()
    for b in range(B):
        t = int(mask_bt[b].sum().item())
        t = max(0, min(t, T))
        seqs.append(x_btjc[b, :t].contiguous())
    return seqs

@torch.no_grad()
def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    preds = _btjc_to_tjc_list(pred_btjc, mask_bt)
    tgts  = _btjc_to_tjc_list(tgt_btjc,  mask_bt)
    dtw_metric = PE_DTW()

    vals = []
    for p, g in zip(preds, tgts):
        if p.size(0) < 2 or g.size(0) < 2:
            continue
        pv = p.detach().cpu().numpy().astype("float32")  # [T,J,C]
        gv = g.detach().cpu().numpy().astype("float32")  # [T,J,C]
        pv = pv[:, None, :, :]  # (T, 1, J, C)
        gv = gv[:, None, :, :]  # (T, 1, J, C)
        vals.append(float(dtw_metric.get_distance(pv, gv)))

    if not vals:
        return torch.tensor(0.0, device=pred_btjc.device)
    return torch.tensor(vals, device=pred_btjc.device).mean()


class LitMinimal(pl.LightningModule):
    """
    Minimal PyTorch Lightning module for SignWriting-to-Pose training.
    Features:
        - Dynamic-window training (no fixed segment length)
        - Masked MSE and velocity-based auxiliary loss
        - One-shot full-sequence prediction during inference
        - On-fit metric logging to CSV
    Args:
        num_keypoints (int): Number of body keypoints in pose representation.
        num_dims (int): Number of spatial dimensions per keypoint (e.g. 3 for x,y,z).
        lr (float): Learning rate for AdamW optimizer.
        log_dir (str): Directory to save training metrics.
    """
    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-3, log_dir="logs"):
        super().__init__()
        self.save_hyperparameters()

        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        import signwriting_animation.diffusion.core.models as M
        print("[WHERE MODELS.PY] ", M.__file__, flush=True)
        print("[HAS time_proj?] ", hasattr(self.model, "future_time_proj"), flush=True)
        print("[FUTURE_PROJ PARAMS] ",
            [n for n, _ in self.model.named_parameters() if "future_time_proj" in n],
            flush=True)

        self.lr = lr
        self.log_dir = log_dir
        self.train_losses, self.val_losses, self.val_dtws = [], [], []

        print("[LitMinimal] full-sequence training (zeros query) + velocity loss ✅")

    def forward(self, x_btjc, timesteps, past_btjc, sign_img):
        """Forward pass through diffusion model (batch-first format)."""
        x_bjct    = btjc_to_bjct(sanitize_btjc(x_btjc))
        past_bjct = btjc_to_bjct(sanitize_btjc(past_btjc))
        out_bjct  = self.model.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred_btjc = bjct_to_btjc(out_bjct)  # [B,T,J,C]
        return pred_btjc

    def _make_mask_bt(self, raw_mask):
        mask = raw_mask.float()
        if mask.dim() == 5:   # [B,T,P,J,C] -> [B,T]
            mask = (mask.abs().sum(dim=(2, 3, 4)) > 0).float()
        elif mask.dim() == 4: # [B,T,J,C] -> [B,T]
            mask = (mask.abs().sum(dim=(2, 3)) > 0).float()
        elif mask.dim() == 3: # [B,T,C]   -> [B,T]
            mask = (mask.abs().sum(dim=2) > 0).float()
        return mask
    def _time_ramp(self, T, device):
        return torch.linspace(0, 1, steps=T, device=device).view(1, T, 1, 1)

    def training_step(self, batch, _):
        if self.global_step == 0:
            import signwriting_animation.diffusion.lightning_module as lm
            print(f"[USING FILE] {lm.__file__}")

        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"])              # [B,Tf,J,C] (Tf dynamic)
        past = sanitize_btjc(cond["input_pose"])         # [B,Tp,J,C]
        mask = self._make_mask_bt(cond["target_mask"])   # [B,Tf]
        sign = cond["sign_image"].float()
        ts   = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)
        
        T = fut.size(1)
        in_seq = 0.20 * torch.randn_like(fut) + 0.20 * self._time_ramp(T, fut.device)
        pred = self.forward(in_seq, ts, past, sign)      # [B,Tf,J,C]

        loss_pos = masked_mse(pred, fut, mask)
        if fut.size(1) > 1:
            vel_mask = mask[:, 1:]
            loss_vel = masked_mse(pred[:,1:]-pred[:,:-1], fut[:,1:]-fut[:,:-1], vel_mask)
            loss = loss_pos + 0.5 * loss_vel
        else:
            loss = loss_pos

        if self.global_step == 0:
            with torch.no_grad():
                mv = (pred[:,1:]-pred[:,:-1]).abs().mean().item()
                print(f"[Sanity] mean |Δpred| (train) = {mv:.6f}")

        self.train_losses.append(loss.item())
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut  = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        mask = self._make_mask_bt(cond["target_mask"])
        sign = cond["sign_image"].float()
        ts   = torch.zeros(fut.size(0), dtype=torch.long, device=fut.device)
        
        T = fut.size(1)
        in_seq = 0.20 * torch.randn_like(fut) + 0.20 * self._time_ramp(T, fut.device)
        pred = self.forward(in_seq, ts, past, sign)

        loss_pos = masked_mse(pred, fut, mask)
        if fut.size(1) > 1:
            vel_mask = mask[:, 1:]
            loss_vel = masked_mse(pred[:,1:]-pred[:,:-1], fut[:,1:]-fut[:,:-1], vel_mask)
            loss = loss_pos + 1 * loss_vel
        else:
            loss = loss_pos

        dtw  = masked_dtw(pred, fut, mask)

        if self.global_step == 0:
            with torch.no_grad():
                mv = (pred[:,1:]-pred[:,:-1]).abs().mean().item()
                print(f"[Sanity] mean |Δpred| (val) = {mv:.6f}")

        self.val_losses.append(loss.item())
        self.val_dtws.append(dtw.item())
        self.log("val/loss", loss, prog_bar=True)
        self.log("val/dtw",  dtw,  prog_bar=False)

    @torch.no_grad()
    def generate_full_sequence(self, past_btjc, sign_img, target_mask=None, target_len=None):
        """
        Predict the entire future segment in one forward pass (per sample length).
        - If `target_len` is None, infer per-sample Tf from `target_mask`.
        - Supports dynamic window (different Tf per sample).
        """
        print("[GEN/full] ENTER generate_full_sequence", flush=True)
        self.eval()
        ctx  = sanitize_btjc(past_btjc).to(self.device)     # [B,Tp,J,C]
        sign = sign_img.to(self.device)
        B, _, J, C = ctx.shape

        if target_len is not None:
            if isinstance(target_len, (int, float)):
                tf_list = [int(target_len)] * B
            elif torch.is_tensor(target_len):
                tf_list = target_len.view(-1).to(torch.long).cpu().tolist()
            else:
                tf_list = [int(x) for x in target_len]
        else:
            assert target_mask is not None, "Need target_len or target_mask"
            mask_bt = self._make_mask_bt(target_mask).to(self.device)      # [B,T]
            tf_list = mask_bt.sum(dim=1).to(torch.long).view(-1).cpu().tolist()

        outs = []
        for b in range(B):
            Tf = max(1, int(tf_list[b]))
            t  = torch.linspace(0, 1, steps=Tf, device=self.device).view(1, Tf, 1, 1)
            x_query = 0.20 * torch.randn((1, Tf, J, C), device=self.device) + 0.20 * t
            ts = torch.zeros(1, dtype=torch.long, device=self.device)
            print(f"[DBG] Tf={Tf}, t.min={float(t.min()):.3f}, t.max={float(t.max()):.3f}", flush=True)
            print(f"[DBG] x_query.mean={float(x_query.mean()):.5f}, x_query.std={float(x_query.std()):.5f}", flush=True)
            pred = self.forward(x_query, ts, ctx[b:b+1], sign[b:b+1])  # [1,Tf,J,C]

            if Tf > 1:
                dv = (pred[:, 1:, :, :2] - pred[:, :-1, :, :2]).abs().mean().item()
                print(f"[GEN/full] sample {b}, Tf={Tf}, mean|Δpred| BEFORE-CPU = {dv:.6f}", flush=True)
                tv = pred[:, :, :, :2].std(dim=1).mean().item()
                print(f"[GEN/full] sample {b}, Tf={Tf}, mean|Δpred|={dv:.6f}, time-std={tv:.6f}", flush=True)
            outs.append(pred)

        return torch.cat(outs, dim=0)  # [B,Tf,J,C]

    def on_fit_end(self):
        os.makedirs(self.log_dir, exist_ok=True)
        csv_path = os.path.join(self.log_dir, "minimal_metrics.csv")
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["step", "train_loss", "val_loss", "val_dtw"])
            max_len = max(len(self.train_losses), len(self.val_losses), len(self.val_dtws))
            for i in range(max_len):
                tr  = self.train_losses[i] if i < len(self.train_losses) else ""
                vl  = self.val_losses[i]  if i < len(self.val_losses)  else ""
                dv  = self.val_dtws[i]    if i < len(self.val_dtws)    else ""
                w.writerow([i + 1, tr, vl, dv])
        print(f"[on_fit_end] metrics saved to {csv_path}")

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)
    