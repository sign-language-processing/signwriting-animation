import os
import pytest
import torch
from torch.utils.data import DataLoader
from pose_format.torch.masked.collator import zero_pad_collator
from lightning.pytorch import Callback
import lightning as pl
import matplotlib.pyplot as plt
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion

class LengthPredictionModule(pl.LightningModule):
    """
    Lightning module to test length prediction with loss logging.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def test_step(self, batch, batch_idx):
        noisy_x = batch["data"].float()
        input_pose = batch["conditions"]["input_pose"].float()
        sign_image = batch["conditions"]["sign_image"].float()
        target_lengths = batch["length_target"].float().squeeze(-1)

        if noisy_x.dim() == 5:
            noisy_x = noisy_x.squeeze(2).permute(0, 2, 3, 1).contiguous()
        if input_pose.dim() == 5:
            input_pose = input_pose.squeeze(2).permute(0, 2, 3, 1).contiguous()

        timesteps = torch.zeros(noisy_x.shape[0], dtype=torch.long, device=noisy_x.device)

        _, length_dist = self.model(noisy_x, timesteps, input_pose, sign_image)

        pred_mean = length_dist.mean.squeeze(-1)
        pred_std = length_dist.stddev.squeeze(-1)

        loss = self.loss_fn(pred_mean, target_lengths)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True, batch_size=target_lengths.shape[0])

        return pred_mean, pred_std, target_lengths

class LengthPredictionPlotCallback(Callback):
    """
    Callback to collect predictions and generate a plot of predicted means with standard deviation error bars.
    """
    def __init__(self, output_dir="outputs"):
        super().__init__()
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        self.test_preds = []
        self.test_stds = []
        self.test_targets = []

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        pred_mean, pred_std, targets = outputs
        self.test_preds.append(pred_mean.cpu())
        self.test_stds.append(pred_std.cpu())
        self.test_targets.append(targets.cpu())

    def on_test_epoch_end(self, trainer, pl_module):
        preds = torch.cat(self.test_preds)
        stds = torch.cat(self.test_stds)
        targets = torch.cat(self.test_targets)

        plt.figure(figsize=(6,6))
        plt.errorbar(targets, preds, yerr=stds, fmt='o', ecolor='gray', capsize=3)
        max_len = max(targets.max().item(), preds.max().item())
        plt.plot([0, max_len], [0, max_len], 'r--')
        plt.xlabel("Target Length")
        plt.ylabel("Predicted Length")
        plt.title("Predicted Mean with StdDev")
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "length_pred_with_std.png"))
        plt.close()

@pytest.mark.parametrize("batch_size", [4])
def test_length_prediction(pose_dataset, batch_size):
    """
    Test the length predictor on a batch of data and log the loss + plot predictions.
    """
    dataset, *_ = pose_dataset
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=zero_pad_collator,
        num_workers=0,
        pin_memory=False,
    )

    model = SignWritingToPoseDiffusion(
        num_keypoints=586,
        num_dims_per_keypoint=3,
    )
    lightning_model = LengthPredictionModule(model)
    plot_callback = LengthPredictionPlotCallback()

    trainer = pl.Trainer(
        max_epochs=1,
        log_every_n_steps=1,
        accelerator="cpu",
        callbacks=[plot_callback]
    )

    trainer.test(lightning_model, dataloader)
