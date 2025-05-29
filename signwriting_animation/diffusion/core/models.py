from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class SignWritingToPoseDiffusion(nn.Module):
    def __init__(self,
                 num_keypoints: int,
                 num_dims: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 activation: str = "gelu",
                 arch: str = "trans_enc",
                 cond_mask_prob: float = 0):
        """
        Generates pose sequences conditioned on SignWriting images

        Args:
            num_keypoints: Number of keypoints.
            num_dims: Number of dimensions per keypoint.
            embedding_arch: CLIP embedding model architecture
            num_latent_dims: Dimension of the latent space.
            ff_size: Feed-forward network size.
            num_layers: Number of Transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            activation: Activation function.
            arch: Architecture type: "trans_enc", "trans_dec", or "gru".
            cond_mask_prob: Condition mask probability for classifier-free guidance (CFG).
        """
        super().__init__()

        self.cond_mask_prob = cond_mask_prob

        # local conditions
        input_feats = num_keypoints * num_dims
        self.future_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.past_motion_process = MotionProcess(input_feats, num_latent_dims)
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)

        # global conditions
        self.embed_signwriting = EmbedSignWriting(num_latent_dims, embedding_arch)
        self.embed_timestep = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        self.seqEncoder = seq_encoder_factory(arch=arch,
                                              latent_dim=num_latent_dims,
                                              ff_size=ff_size,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              activation=activation)

        self.pose_projection = OutputProcessMLP(num_latent_dims, num_keypoints, num_dims)

    def forward(self, x, timesteps, past_motion, signwriting_im_batch):
        bs, keypoints, dims, nframes = x.shape

        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)  # [1, bs, L]
        past_motion_emb = self.past_motion_process(past_motion)  # [past_frames, bs, L]
        future_motion_emb = self.future_motion_process(x)

        xseq = torch.cat((time_emb,
                          signwriting_emb,
                          past_motion_emb,
                          future_motion_emb), axis=0)

        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:]
        output = self.pose_projection(output)
        return output

    def interface(self, x, timesteps, y=None):
        """
        Performs classifier-free guidance: runs a forward pass of the diffusion model using either conditional
        or unconditional mode.

        Args:
            x: [batch_size, frames, keypoints, dims], denoted x_t in the CAMDM paper
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        batch_size, num_keypoints, num_dims, num_frames = x.shape

        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # CFG on past motion
        keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model: project to pose space.

    Obtained module from https://github.com/sign-language-processing/fluent-pose-synthesis
    """
    def __init__(self,
                 num_latent_dims: int,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
                 hidden_dim: int = 512):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(num_latent_dims, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, num_keypoints * num_dims_per_keypoint)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (input) [num_frames, batch_size, num_keypoints*num_dims_per_keypoint]
            x: (output) [batch_size, num_keypoints, num_dims_per_keypoint, num_frames]
        """
        num_frames, batch_size, d = x.shape
        x = self.mlp(x)  # use MLP instead of single linear layer
        x = x.reshape(num_frames, batch_size, self.num_keypoints, self.num_dims_per_keypoint)
        x = x.permute(1, 2, 3, 0)
        return x


class OutputProcess(nn.Module):
    def __init__(self, input_feats, latent_dim, num_keypoints, num_dims_per_keypoint):
        super().__init__()
        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.poseFinal = nn.Linear(latent_dim, input_feats)

    def forward(self, output):
        """
        Args:
            output: (input) [num_frames, batch_size, latent_dim]
            output: (output) [batch_size, num_keypoints, num_dims_per_keypoint, num_frames]
        """
        num_frames, batch_size, d = output.shape
        output = self.poseFinal(output)
        output = output.reshape(num_frames, batch_size, self.num_keypoints, self.num_dims_per_keypoint)
        output = output.permute(1, 2, 3, 0)
        return output


class EmbedSignWriting(nn.Module):
    def __init__(self, num_latent_dims: int, embedding_arch='openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None

        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        # image_batch should be in the format [B, 3, H, W], where H=W=224.
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)

        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)

        return embeddings_batch
