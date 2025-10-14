import os
import csv
import torch
import numpy as np
import lightning as pl
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import animation
from torch.utils.data import DataLoader, Dataset
from pose_format.torch.masked.collator import zero_pad_collator
from signwriting_animation.data.data_loader import DynamicPosePredictionDataset
from signwriting_animation.diffusion.lightning_module import LitMinimal, masked_dtw


def _to_plain_tensor(x):
    """Convert MaskedTensor or custom tensor to plain CPU tensor."""
    if hasattr(x, "tensor"):
        x = x.tensor
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    return x.detach().cpu()


def visualize_pose_sequence(seq_btjc, save_path="logs/free_run_vis.png", step=5):
    """
    Visualize a pose sequence (e.g., model prediction).
    seq_btjc: [1,T,J,C]
    """
    seq = _to_plain_tensor(seq_btjc)[0]  # [T,J,C]
    T, J, C = seq.shape
    plt.figure(figsize=(5, 5))
    for t in range(0, T, step):
        pose = seq[t]
        plt.scatter(pose[:, 0], -pose[:, 1], s=8)
    plt.title("Predicted Pose Trajectory (sampled frames)")
    plt.axis("equal")
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    plt.close()


class FilteredDataset(Dataset):
    """Subset of valid samples for minimal overfit test."""
    def __init__(self, base: Dataset, target_count=4, max_scan=500):
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

def _as_dense_cpu_btjc(x):
    if hasattr(x, "zero_filled"):
        x = x.zero_filled()
    if getattr(x, "is_sparse", False):
        x = x.to_dense()
    x = x.detach().float().cpu().contiguous()
    if x.dim() == 5:
        x = x[:, :, 0, ...]
    return x

def make_loader(data_dir, csv_path, split, bs, num_workers):
    base = DynamicPosePredictionDataset(
        data_dir=data_dir, csv_path=csv_path, with_metadata=True, split=split
    )
    ds = FilteredDataset(base, target_count=4, max_scan=1000)
    print(f"[DEBUG] split={split} | batch_size={bs} | len(ds)={len(ds)}")
    return DataLoader(
        ds, batch_size=bs, shuffle=True, num_workers=num_workers, collate_fn=zero_pad_collator
    )


if __name__ == "__main__":
    pl.seed_everything(42, workers=True)
    torch.set_default_dtype(torch.float32)

    data_dir = "/data/yayun/pose_data"
    csv_path = "/data/yayun/signwriting-animation/data.csv"

    batch_size = 2
    num_workers = 2

    train_loader = make_loader(data_dir, csv_path, "train", bs=batch_size, num_workers=num_workers)
    val_loader = train_loader  # same small subset for validation

    model = LitMinimal(log_dir="logs")

    trainer = pl.Trainer(
        max_steps=500,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        log_every_n_steps=5,
        limit_train_batches=10,
        limit_val_batches=5,
        check_val_every_n_epoch=1,
        deterministic=True,
        num_sanity_val_steps=0,
        enable_checkpointing=False,
    )

    trainer.fit(model, train_loader, val_loader)

    try:
        model.on_fit_end()
    except Exception as e:
        print("[WARN] on_fit_end() failed:", e)

    # Inference on 1 example for visual check
    model.eval()
    with torch.no_grad():
        batch = next(iter(train_loader))
        cond  = batch["conditions"]

        past_btjc = cond["input_pose"][:1].to(model.device)
        sign_img  = cond["sign_image"][:1].to(model.device)
        fut_gt    = batch["data"][:1].to(model.device)
        mask_bt   = cond["target_mask"][:1].to(model.device) 

        print("[GEN] using generate_full_sequence", flush=True)
        gen_btjc = model.generate_full_sequence(
            past_btjc=past_btjc,
            sign_img=sign_img,
            target_mask=mask_bt,
        )

        gen_btjc_cpu = _as_dense_cpu_btjc(gen_btjc)[0]  # [T,J,C]
        fut_gt_cpu   = _as_dense_cpu_btjc(fut_gt)[0]    # [T,J,C]

        def frame_disp_cpu(x):  # x: [T,J,C] dense
            if x.size(0) < 2: return 0.0
            dx = x[1:, :, :2] - x[:-1, :, :2]
            return dx.abs().mean().item()

        Tf = gen_btjc_cpu.size(0)
        mv_pred = frame_disp_cpu(gen_btjc_cpu)
        mv_gt   = frame_disp_cpu(fut_gt_cpu)
        print(f"[GEN] Tf={Tf}, mean|Δpred|={mv_pred:.4f}, mean|Δgt|={mv_gt:.4f}", flush=True)
        if Tf < 2:
            print("[WARN] predicted sequence length < 2; animation may look static.", flush=True)

        try:
            dtw_free = masked_dtw(gen_btjc, fut_gt, mask_bt).item()
            print(f"[Free-run] DTW: {dtw_free:.4f}", flush=True)
        except Exception as e:
            print("[Free-run] DTW eval skipped:", e)

        os.makedirs("logs", exist_ok=True)
        torch.save({"gen": gen_btjc_cpu, "gt": fut_gt_cpu}, "logs/free_run.pt")
        print("Saved free-run sequences to logs/free_run.pt", flush=True)

        fig, ax = plt.subplots(figsize=(5, 5))
        sc_pred = ax.scatter([], [], s=15, c="red",  label="Predicted", animated=True)
        sc_gt   = ax.scatter([], [], s=15, c="blue", label="GT",        alpha=0.35, animated=True)
        ax.legend(loc="upper right", frameon=False)
        ax.axis("equal")

        xy = torch.cat([
            gen_btjc_cpu[..., :2].reshape(-1, 2),
            fut_gt_cpu[...,  :2].reshape(-1, 2)
        ], dim=0).numpy()
        x_min, y_min = xy.min(axis=0); x_max, y_max = xy.max(axis=0)
        pad = 0.05 * max(x_max - x_min, y_max - y_min, 1e-6)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)

        def _init():
            sc_pred.set_offsets(np.empty((0, 2)))
            sc_gt.set_offsets(np.empty((0, 2)))
            return sc_pred, sc_gt

        def _update(f):
            ax.set_title(f"Free-run prediction  |  frame {f+1}/{len(gen_btjc_cpu)}")
            sc_pred.set_offsets(gen_btjc_cpu[f, :, :2].numpy())
            sc_gt.set_offsets(  fut_gt_cpu[f,   :, :2].numpy())
            return sc_pred, sc_gt

        ani = animation.FuncAnimation(
            fig, _update, frames= max(1, len(gen_btjc_cpu)),
            init_func=_init, interval=100, blit=True
        )
        ani.save("logs/free_run_anim.gif", writer="pillow", fps=10)
        plt.close(fig)
        print("Saved free-run animation to logs/free_run_anim.gif", flush=True)
