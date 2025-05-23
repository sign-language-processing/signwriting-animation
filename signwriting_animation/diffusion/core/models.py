from typing import Optional
import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class SignWritingToPoseDiffusion(nn.Module):
    def __init__(self,
                 input_feats: int,
                 keypoints: int,
                 dims: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 activation: str = "gelu",
                 arch: str = "trans_enc",
                 cond_mask_prob: float = 0,
                 device: Optional[torch.device] = None):
        """
        Generates pose sequences conditioned on SignWriting images

        Args:
            input_feats: Number of input features (keypoints * dimensions).
            keypoints: Number of keypoints.
            dims: Number of dimensions per keypoint.
            embedding_arch: CLIP embedding model architecture
            latent_dim: Dimension of the latent space.
            ff_size: Feed-forward network size.
            num_layers: Number of Transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            activation: Activation function.
            arch: Architecture type: "trans_enc", "trans_dec", or "gru".
            cond_mask_prob: Condition mask probability for classifier-free guidance (CFG).
            device: Device to run the model.
        """
        super().__init__()

        self.cond_mask_prob = cond_mask_prob

        # local conditions
        self.future_motion_process = MotionProcess(input_feats, latent_dim)
        self.past_motion_process = MotionProcess(input_feats, latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(latent_dim, dropout)

        # global conditions
        self.embed_signwriting = EmbedSignWriting(latent_dim, embedding_arch)
        self.embed_timestep = TimestepEmbedder(latent_dim, self.sequence_pos_encoder)

        self.seqEncoder = seq_encoder_factory(arch=arch,
                                              latent_dim=latent_dim,
                                              ff_size=ff_size,
                                              num_layers=num_layers,
                                              num_heads=num_heads,
                                              dropout=dropout,
                                              activation=activation)

        self.pose_projection = OutputProcessMLP(input_feats, latent_dim, keypoints, dims)

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
            x: [batch_size, frames, keypoints, dims], denoted x_t in the paper
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        bs, keypoints, dims, nframes = x.shape

        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        # CFG on past motion
        keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1 - self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((bs, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model: project to pose space.

    Obtained module from https://github.com/sign-language-processing/fluent-pose-synthesis
    """
    def __init__(self, input_feats: int,
                 latent_dim: int,
                 keypoints: int,
                 dims: int,
                 hidden_dim=512):
        super().__init__()
        self.keypoints = keypoints
        self.dims = dims

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(hidden_dim // 2, input_feats)
        )

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.mlp(output)  # use MLP instead of single linear layer
        output = output.reshape(nframes, bs, self.keypoints, self.dims)
        output = output.permute(1, 2, 3, 0)
        return output


class EmbedSignWriting(nn.Module):
    def __init__(self, latent_dim: int, embedding_arch='openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None

        if (embedding_dim := self.model.visual_projection.out_features) != latent_dim:
            self.proj = nn.Linear(embedding_dim, latent_dim)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        # image_batch should be in the format [B, 3, H, W], where H=W=224.
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)

        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)

        return embeddings_batch
