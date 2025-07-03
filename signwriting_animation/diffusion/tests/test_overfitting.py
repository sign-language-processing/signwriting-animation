import torch
import lightning as pl
import unittest
from torch.utils.data import DataLoader
from transformers import CLIPProcessor
from signwriting_animation.diffusion.core.models import SignWritingToPoseDiffusion
from signwriting_evaluation.metrics.clip import signwriting_to_clip_image

CLIP_MODEL_NAME = "openai/clip-vit-base-patch32"

def make_sample(signwriting_str, pose_val, past_motion_val, clip_processor,
                num_past_frames=10, num_keypoints=21, num_dims=3):
    """
    Create a toy sample for overfitting tests: 
    pose and past_motion are all set to the given value, signwriting is rendered to image.
    Returns (pose, timestep, past_motion, sw_img, target_val_tensor).
    """
    x = torch.full((num_keypoints, num_dims, num_past_frames), pose_val, dtype=torch.float32)
    timesteps = torch.tensor(0, dtype=torch.long)
    past_motion = torch.full((num_keypoints, num_dims, num_past_frames), past_motion_val, dtype=torch.float32)
    pil_img = signwriting_to_clip_image(signwriting_str)
    sw_img = clip_processor(images=pil_img, return_tensors="pt").pixel_values.squeeze(0)
    pose_val_tensor = torch.tensor(pose_val, dtype=torch.float32)
    return x, timesteps, past_motion, sw_img, pose_val_tensor

class LightningOverfitModel(pl.LightningModule):
    """
    PyTorch Lightning wrapper for overfit sanity check on a SignWriting-to-Pose diffusion model.
    """
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.loss_fn = torch.nn.MSELoss()

    def forward(self, x, timesteps, past_motion, sw_img):
        return self.model(x, timesteps, past_motion, sw_img)

    def training_step(self, batch, batch_idx):
        x, timesteps, past_motion, sw_img, val = batch
        output = self(x, timesteps, past_motion, sw_img)
        target = val.view(-1, 1, 1, 1).expand_as(output)
        loss = self.loss_fn(output, target)
        if batch_idx == 0 and self.current_epoch % 10 == 0:
            self.print(f"Output min/max: {output.min().item()}, {output.max().item()}")
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=1e-4)

class TestOverfitSanity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        pl.seed_everything(42)
        cls.clip_processor = CLIPProcessor.from_pretrained(CLIP_MODEL_NAME)
        sample_configs = [
            ("M518x529S14c20481x471S27106503x489", 0, -1),
            ("M518x529S14c20481x471S27106503x489", 1, 1),
            ("M518x533S1870a489x515S18701482x490", 0, 0),
            ("M518x533S1870a489x515S18701482x490", 1, 2),
        ]

        cls.samples = [
            make_sample(sw, pose_val, past_motion_val, cls.clip_processor)
            for sw, pose_val, past_motion_val in sample_configs
        ]

    def test_overfit(self):
        dataloader = DataLoader(self.samples, batch_size=4, shuffle=False)
        model = SignWritingToPoseDiffusion(
            num_keypoints=21,
            num_dims_per_keypoint=3,
            embedding_arch=CLIP_MODEL_NAME,
            num_latent_dims=32,
            ff_size=64,
            num_layers=2,
            num_heads=2,
            dropout=0.0,
            cond_mask_prob=0
        )
        lightning_model = LightningOverfitModel(model)
        trainer = pl.Trainer(
            max_epochs=60,
            log_every_n_steps=1,
            accelerator="cpu",
        )
        trainer.fit(lightning_model, dataloader)

        print("\nEvaluating overfit sanity...")
        lightning_model.eval()
        test_dataloader = DataLoader(self.samples, batch_size=1, shuffle=False)
        with torch.no_grad():
            for idx, (x, timesteps, past_motion, sw_img, val) in enumerate(test_dataloader):
                output = lightning_model(x, timesteps, past_motion, sw_img)
                rounded = torch.round(output)
                target = val.view(-1, 1, 1, 1).expand_as(output)
                print(f"[EVAL] Sample {idx+1} (target={val.item()})")
                print("Output min/max:", output.min().item(), output.max().item())
                print("Rounded unique:", rounded.unique())
                print("Prediction after round:\n", rounded.cpu().numpy())
                print("Target:\n", target.cpu().numpy())
                assert torch.allclose(rounded, target, atol=1e-1), f"Sample {idx+1} did not overfit!"
        print("All overfit sanity checks passed!")

if __name__ == "__main__":
    unittest.main()
