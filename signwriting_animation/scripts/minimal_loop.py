import os
import csv
import torch
import lightning as pl
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion


def _to_dense(x):
    """
    Safely convert a batch tensor to dense float32:
    - If it's a pose-format MaskedTensor, call .zero_filled() to obtain a dense tensor.
    - If it's a sparse tensor, densify it.
    - Cast to float32 and make memory contiguous.
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
        raise ValueError(f"Expected 4D tensor [B,T,J,C], got {tuple(x.shape)}")
    return x


def btjc_to_bjct(x):
    """Convert [B, T, J, C] → [B, J, C, T] (model forward format)."""
    return x.permute(0, 2, 3, 1).contiguous()


def bjct_to_btjc(x):
    """Convert [B, J, C, T] → [B, T, J, C] (loss/metrics format)."""
    return x.permute(0, 3, 1, 2).contiguous()


def masked_mse(pred, tgt, mask_bt):
    pred, tgt = sanitize_btjc(pred), sanitize_btjc(tgt)

    Tm = min(pred.size(1), tgt.size(1), mask_bt.size(1))
    pred = pred[:, :Tm]
    tgt  = tgt[:,  :Tm]

    m4 = mask_bt[:, :Tm].float()[:, :, None, None]   # [B,T,1,1]

    diff2 = (pred - tgt) ** 2                        # [B,T,J,C]
    num = (diff2 * m4).sum()
    den = (m4.sum() * pred.size(2) * pred.size(3)).clamp_min(1.0)
    return num / den

def masked_dtw(pred_btjc, tgt_btjc, mask_bt):
    pred = sanitize_btjc(pred_btjc)  # [B,T,J,C]
    tgt  = sanitize_btjc(tgt_btjc)
    B, T, J, C = pred.shape
    vals = []

    for b in range(B):
        t = min(int(mask_bt[b].sum().item()), pred.size(1), tgt.size(1))
        if t <= 1:
            continue

        x = pred[b, :t].reshape(t, -1).to(dtype=torch.float32)  # [t, J*C]
        y = tgt[b,  :t].reshape(t, -1).to(dtype=torch.float32)

        device = x.device
        D = torch.cdist(x, y)                      # [t, t]
        Cmat = torch.empty((t, t), device=device)
        Cmat[0, 0] = D[0, 0]
        for i in range(1, t):
            Cmat[i, 0] = D[i, 0] + Cmat[i-1, 0]
        for j in range(1, t):
            Cmat[0, j] = D[0, j] + Cmat[0, j-1]

        for i in range(1, t):
            for j in range(1, t):
                m = torch.minimum(
                        torch.minimum(Cmat[i-1, j], Cmat[i, j-1]),
                        Cmat[i-1, j-1]
                    )
                Cmat[i, j] = D[i, j] + m

        vals.append(Cmat[t-1, t-1] / (2 * t))

    if not vals:
        return torch.tensor(0.0, device=pred.device)
    return torch.stack(vals).mean()


class FilteredDataset(Dataset):
    """Only keep valid samples."""
    def __init__(self, base: Dataset, target_count=200, max_scan=5000):
        self.base = base
        self.idx = []
        N = len(base)
        for i in range(min(N, max_scan)):
            try:
                it = base[i]
                if isinstance(it, dict) and "data" in it and "conditions" in it:
                    self.idx.append(i)
                if len(self.idx) >= target_count:
                    break
            except Exception:
                continue
        if not self.idx:
            self.idx = [0]

    def __len__(self):
        return len(self.idx)

    def __getitem__(self, i):
        return self.base[self.idx[i]]

class LitMinimal(pl.LightningModule):
    """
    Minimal Lightning module:
    - Forward: SignWritingToPoseDiffusion (expects BJCT)
    - Loss: masked MSE; plus DTW in validation as a sanity metric
    - No checkpointing; meant for quick end-to-end checks.
    """
    def __init__(self, num_keypoints=586, num_dims=3, lr=1e-3, log_dir="logs"):
        super().__init__()
        self.model = SignWritingToPoseDiffusion(
            num_keypoints=num_keypoints, num_dims_per_keypoint=num_dims
        )
        self.lr = lr
        self.log_dir = log_dir
        self.train_losses, self.val_losses, self.val_dtws = [], [], []

    def forward(self, x_bjct, timesteps, past_bjct, sign_img):
        return self.model.forward(x_bjct, timesteps, past_bjct, sign_img)

    def training_step(self, batch, _):
        cond = batch["conditions"]
        fut = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        raw_mask = cond["target_mask"]
        mask = raw_mask.float()
        if mask.dim() == 5:
            mask = (mask.abs().sum(dim=(2, 3, 4)) > 0).float()
        elif mask.dim() == 4:
            mask = (mask.abs().sum(dim=(2, 3)) > 0).float()
        elif mask.dim() == 3:
            mask = (mask.abs().sum(dim=2) > 0).float()
        sign_img = cond["sign_image"].float()

        x_bjct = btjc_to_bjct(fut)
        past_bjct = btjc_to_bjct(past)
        timesteps = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)
        out = self.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred = bjct_to_btjc(out)
        loss = masked_mse(pred, fut, mask)

        self.train_losses.append(loss.item())
        self.log("train/loss", loss, prog_bar=True, on_step=True)
        return loss

    def validation_step(self, batch, _):
        cond = batch["conditions"]
        fut = sanitize_btjc(batch["data"])
        past = sanitize_btjc(cond["input_pose"])
        
        raw_mask = cond["target_mask"]
        mask = raw_mask.float()
        if mask.dim() == 5:
            mask = (mask.abs().sum(dim=(2, 3, 4)) > 0).float()
        elif mask.dim() == 4:
            mask = (mask.abs().sum(dim=(2, 3)) > 0).float()
        elif mask.dim() == 3:
            mask = (mask.abs().sum(dim=2) > 0).float()
        sign_img = cond["sign_image"].float()

        x_bjct = btjc_to_bjct(fut)
        past_bjct = btjc_to_bjct(past)
        timesteps = torch.zeros(x_bjct.size(0), dtype=torch.long, device=x_bjct.device)
        out = self.forward(x_bjct, timesteps, past_bjct, sign_img)
        pred = bjct_to_btjc(out)
        loss = masked_mse(pred, fut, mask)
        
        dtw_val = masked_dtw(pred, fut, mask)
        self.log("val/dtw", dtw_val, prog_bar=False)
        self.val_dtws.append(dtw_val.item())
        self.val_losses.append(loss.item())
        self.log("val/loss", loss, prog_bar=True)

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
                dtw = self.val_dtws[i]    if i < len(self.val_dtws)    else ""
                w.writerow([i + 1, tr, vl, dtw])

        import matplotlib.pyplot as plt
        plt.figure()
        if self.train_losses:
            plt.plot(self.train_losses, label="train_loss")
        if self.val_losses:
            plt.plot(self.val_losses, label="val_loss")
        if self.val_dtws:
            plt.plot(self.val_dtws, label="val_dtw")
        plt.xlabel("steps")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(self.log_dir, "minimal_curves.png"))
        plt.close()

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=self.lr)


def make_loader(data_dir, csv_path, split, bs, num_workers):
    """
    Build a DataLoader with:
    - Dataset returning MaskedTensor.
    - zero_pad_collator to align sequences along time dimension.
    - FilteredDataset to pick a small, valid subset for a minimal run.
    """
    base = DynamicPosePredictionDataset(data_dir=data_dir, csv_path=csv_path, with_metadata=True, split=split)
    ds = FilteredDataset(base, target_count=200, max_scan=3000)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(ds, batch_size=bs, shuffle=False, num_workers=num_workers, collate_fn=zero_pad_collator)


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"

    batch_size = 4
    num_workers = 2

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers)
    val_loader = train_loader

    model = LitMinimal(log_dir="logs")

    trainer = pl.Trainer(
        max_steps=1000,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        check_val_every_n_epoch=1,
        enable_checkpointing=False,
        deterministic=True,
        num_sanity_val_steps=0,
    )
    trainer.fit(model, train_loader, val_loader)
