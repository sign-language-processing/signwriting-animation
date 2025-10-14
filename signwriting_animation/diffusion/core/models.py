import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, seq_encoder_factory


class SignWritingToPoseDiffusion(nn.Module):
    def __init__(self,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
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
        Generates pose sequences conditioned on SignWriting images and past motion using a diffusion model.

        Args:
            num_keypoints (int):
                Number of keypoints in the pose representation.

            num_dims_per_keypoint (int):
                Number of spatial dimensions per keypoint (e.g., 2 for 2D, 3 for 3D).

            embedding_arch (str):
                Architecture used for extracting image embeddings (e.g., CLIP variants).

            num_latent_dims (int):
                Dimensionality of the latent representation used by the model.

            ff_size (int):
                Size of the feed-forward network in the Transformer blocks.

            num_layers (int):
                Number of Transformer encoder/decoder layers.

            num_heads (int):
                Number of attention heads in each multi-head attention block.

            dropout (float):
                Dropout rate applied during training.

            activation (str):
                Activation function used in the Transformer (e.g., "gelu", "relu").

            arch (str):
                Architecture type used in the diffusion model. Options: "trans_enc", "trans_dec", or "gru".

            cond_mask_prob (float):
                Probability of masking conditional inputs for classifier-free guidance (CFG).
        """
        super().__init__()

        self.cond_mask_prob = cond_mask_prob

        # local conditions
        input_feats = num_keypoints * num_dims_per_keypoint
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

        self.pose_projection = OutputProcessMLP(num_latent_dims, num_keypoints, num_dims_per_keypoint)

        self.future_time_proj = nn.Sequential(
            nn.Linear(1, num_latent_dims),
            nn.SiLU(),
            nn.Linear(num_latent_dims, num_latent_dims)
        )

        self.future_after_time_ln = nn.LayerNorm(num_latent_dims)

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor):
        """
        Performs classifier-free guidance by running a forward pass of the diffusion model in either
        conditional or unconditional mode.

        Args:
            x (Tensor):
                The noisy input tensor at the current diffusion step, denoted as x_t in the CAMDM paper.
                Shape: [batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint].

            timesteps (Tensor):
                Diffusion timesteps for each sample in the batch.
                Shape: [batch_size], dtype: int.

            past_motion (Tensor):
                Tensor containing motion history information.
                Shape: [batch_size, num_keypoints, num_dims_per_keypoint, num_past_frames].

            signwriting_im_batch (Tensor):
                Batch of rendered SignWriting images.
                Shape: [batch_size, 3, 224, 224].

        Returns:
            Tensor:
                The predicted denoised motion at the current timestep.
                Shape: [batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint].
        """

        batch_size, num_keypoints, num_dims_per_keypoint, num_frames = x.shape

        time_emb = self.embed_timestep(timesteps)  # [1, batch_size, num_latent_dims]
        signwriting_emb = self.embed_signwriting(signwriting_im_batch)  # [1, batch_size, num_latent_dims]
        past_motion_emb = self.past_motion_process(past_motion)  # [past_frames, batch_size, num_latent_dims]
        future_motion_emb = self.future_motion_process(x)  # [future_frames, batch_size, num_latent_dims]

        Tf = future_motion_emb.size(0)
        B  = future_motion_emb.size(1)
        t  = torch.linspace(0, 1, steps=Tf, device=future_motion_emb.device).view(Tf, 1, 1)  # [Tf,1,1]
        t_latent = self.future_time_proj(t)                  # [Tf,1,D]
        t_latent = t_latent.expand(-1, B, -1)                # [Tf,B,D]
        future_motion_emb = future_motion_emb + 0.5* t_latent
        #future_motion_emb = self.future_after_time_ln(future_motion_emb)

        with torch.no_grad():
            fut_time_std = future_motion_emb.float().std(dim=0).mean().item()  # time=dim 0
            print(f"[DBG/model] future_emb time-std={fut_time_std:.6f}", flush=True)
        xseq = torch.cat((time_emb,
                          signwriting_emb,
                          past_motion_emb,
                          future_motion_emb), axis=0)

        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-num_frames:]
        with torch.no_grad():
            enc_time_std = output.float().std(dim=0).mean().item()
            print(f"[DBG/model] encoder_out time-std={enc_time_std:.6f}", flush=True)
        output = self.pose_projection(output)
        return output

    def interface(self,
                  x: torch.Tensor,
                  timesteps: torch.Tensor,
                  y: dict):
        """
        Performs classifier-free guidance by running a forward pass of the diffusion model
        in either conditional or unconditional mode. Extracts conditioning inputs from `y` and
        applies random masking to simulate unconditional sampling.

        Args:
            x (Tensor):
                The noisy input tensor at the current diffusion step, denoted as x_t in the CAMDM paper.
                Shape: [batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint].

            timesteps (Tensor):
                Diffusion timesteps for each sample in the batch.
                Shape: [batch_size], dtype: int.

            y (dict):
                Dictionary of conditioning inputs. Must contain:
                    - 'sign_image': Tensor of shape [batch_size, 3, 224, 224]
                    - 'input_pose': Tensor of shape [batch_size, num_keypoints, num_dims_per_keypoint, num_past_frames]

        Returns:
            Tensor:
                The predicted denoised motion at the current timestep.
                Shape: [batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint].
        """
        batch_size, num_past_frames, num_keypoints, num_dims_per_keypoint = x.shape

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
        Decodes a sequence of latent vectors into keypoint motion data using a multi-layer perceptron (MLP).

        Args:
            x (Tensor):
                Input latent tensor.
                Shape: [num_frames, batch_size, num_latent_dims].

        Returns:
            Tensor:
                Decoded keypoint motion.
                Shape: [batch_size, num_keypoints, num_dims_per_keypoint, num_frames].
        """
        num_frames, batch_size, num_latent_dims = x.shape
        x = self.mlp(x)  # use MLP instead of single linear layer
        x = x.reshape(num_frames, batch_size, self.num_keypoints, self.num_dims_per_keypoint)
        x = x.permute(1, 2, 3, 0)
        return x


class EmbedSignWriting(nn.Module):
    def __init__(self, num_latent_dims: int, embedding_arch: str = 'openai/clip-vit-base-patch32'):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)
        self.proj = None

        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Args:
            image_batch: [batch_size, 3, 224, 224]
        Returns:
            embeddings_batch: [1, batch_size, num_latent_dims]
        """
        # image_batch should be in the format [B, 3, H, W], where H=W=224.
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)

        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)

        return embeddings_batch[None, ...]
