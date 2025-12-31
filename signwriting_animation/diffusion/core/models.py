import torch
import torch.nn as nn
from transformers import CLIPModel

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder


class ContextEncoder(nn.Module):
    """
    Past motion context encoder with PositionalEncoding.
    
    Encodes the past motion sequence into a fixed-size context vector
    that conditions the diffusion model. Uses Transformer encoder with
    CAMDM PositionalEncoding for temporal awareness.
    
    Args:
        input_feats: Input feature dimension (num_joints * num_dims)
        latent_dim: Latent/output dimension
        num_layers: Number of Transformer encoder layers
        num_heads: Number of attention heads
        dropout: Dropout probability
    """
    
    def __init__(self, 
                 input_feats: int, 
                 latent_dim: int, 
                 num_layers: int = 2, 
                 num_heads: int = 4, 
                 dropout: float = 0.1):
        super().__init__()
        
        # Project pose features to latent space
        self.pose_encoder = nn.Linear(input_feats, latent_dim)
        
        # CAMDM PositionalEncoding for temporal awareness
        self.pos_encoding = PositionalEncoding(latent_dim, dropout)
        
        # Transformer encoder for sequence modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=latent_dim,
            nhead=num_heads,
            dim_feedforward=latent_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode past motion sequence to context vector.
        
        Args:
            x: Past poses [B, T, J, C] or [B, T, J*C]

        Returns:
            Context vector [B, D] (mean-pooled over time)
        """
        if x.dim() == 4:
            B, T, J, C = x.shape
            x = x.reshape(B, T, J * C)

        x_emb = self.pose_encoder(x)      # [B, T, D]
        x_emb = x_emb.permute(1, 0, 2)    # [T, B, D]
        x_emb = self.pos_encoding(x_emb)  # Add positional info
        x_enc = self.encoder(x_emb)       # [T, B, D]
        x_enc = x_enc.permute(1, 0, 2)    # [B, T, D]
        context = x_enc.mean(dim=1)       # [B, D]
        return context


class SignWritingToPoseDiffusion(nn.Module):
    """
    Diffusion Model: Frame-Independent Decoder with CAMDM Components.
    
    Architecture:
    - ContextEncoder: Encodes past motion with PositionalEncoding
    - EmbedSignWriting: CLIP-based SignWriting image encoder
    - TimestepEmbedder: Encodes diffusion timestep
    - Frame decoder: MLP that decodes each frame independently
    
    Args:
        num_keypoints: Number of pose keypoints (e.g., 178 for holistic)
        num_dims_per_keypoint: Dimensions per keypoint (typically 3 for x,y,z)
        embedding_arch: CLIP model architecture
        num_latent_dims: Latent dimension for all encoders
        num_heads: Number of attention heads in ContextEncoder
        dropout: Dropout probability
        cond_mask_prob: Condition masking probability for classifier-free guidance
        t_past: Number of past frames for context
        t_future: Number of future frames to predict
        freeze_clip: Whether to freeze CLIP parameters (default: True)
    """
    
    def __init__(self,
                 num_keypoints: int,
                 num_dims_per_keypoint: int,
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 num_latent_dims: int = 256,
                 num_heads: int = 4,
                 dropout: float = 0.1,
                 cond_mask_prob: float = 0,
                 t_past: int = 40,
                 t_future: int = 20,
                 freeze_clip: bool = True):
        super().__init__()

        self.num_keypoints = num_keypoints
        self.num_dims_per_keypoint = num_dims_per_keypoint
        self.cond_mask_prob = cond_mask_prob
        self.t_past = t_past
        self.t_future = t_future
        self._forward_count = 0

        input_feats = num_keypoints * num_dims_per_keypoint

        # === Condition Encoders ===

        # Past motion encoder with PositionalEncoding
        self.past_context_encoder = ContextEncoder(
            input_feats, num_latent_dims,
            num_layers=2, num_heads=num_heads, dropout=dropout,
        )

        # SignWriting image encoder (CLIP-based)
        self.embed_signwriting = EmbedSignWriting(
            num_latent_dims, embedding_arch, freeze_clip=freeze_clip
        )

        # Timestep encoder using CAMDM TimestepEmbedder
        self.sequence_pos_encoder = PositionalEncoding(num_latent_dims, dropout)
        self.time_embed = TimestepEmbedder(num_latent_dims, self.sequence_pos_encoder)

        # === Noisy Frame Encoder ===
        self.xt_frame_encoder = nn.Sequential(
            nn.Linear(input_feats, num_latent_dims),
            nn.GELU(),
            nn.Linear(num_latent_dims, num_latent_dims),
        )

        # === Output Positional Embeddings ===
        self.output_pos_embed = nn.Embedding(512, num_latent_dims)

        # === Frame Decoder ===
        decoder_input_dim = num_latent_dims * 3  # context + xt_emb + pos_emb
        self.decoder = nn.Sequential(
            nn.Linear(decoder_input_dim, 512),
            nn.GELU(),
            nn.Linear(512, 512),
            nn.GELU(),
            nn.Linear(512, input_feats),
        )

    def forward(self,
                x: torch.Tensor,
                timesteps: torch.Tensor,
                past_motion: torch.Tensor,
                signwriting_im_batch: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: Predict clean x0 from noisy input.
        
        Args:
            x: Noisy motion [B, J, C, T_future] in BJCT format
            timesteps: Diffusion timestep [B]
            past_motion: Historical frames [B, J, C, T_past] in BJCT format
            signwriting_im_batch: SignWriting images [B, 3, H, W]
        
        Returns:
            Predicted x0 [B, J, C, T_future] in BJCT format
        """
        B, J, C, T_future = x.shape
        device = x.device

        debug = self._forward_count == 0

        # === Convert past_motion to BTJC format for ContextEncoder ===
        if past_motion.dim() == 4:
            if past_motion.shape[1] == J and past_motion.shape[2] == C:
                past_btjc = past_motion.permute(0, 3, 1, 2).contiguous()
            else:
                past_btjc = past_motion

        # === Encode Conditions ===
        past_ctx = self.past_context_encoder(past_btjc)  # [B, D]
        sign_emb = self.embed_signwriting(signwriting_im_batch)  # [B, D]
        time_emb = self.time_embed(timesteps).squeeze(0)  # [B, D]

        # Fuse all conditions
        context = past_ctx + sign_emb + time_emb  # [B, D]

        # === Frame-Independent Decoding ===
        outputs = []
        for t in range(T_future):
            xt_frame = x[:, :, :, t].reshape(B, -1)
            xt_emb = self.xt_frame_encoder(xt_frame)

            pos_idx = torch.tensor([t], device=device)
            pos_emb = self.output_pos_embed(pos_idx).expand(B, -1)

            dec_input = torch.cat([context, xt_emb, pos_emb], dim=-1)
            out = self.decoder(dec_input)
            outputs.append(out)

        # Stack and reshape to BJCT
        result = torch.stack(outputs, dim=0)
        result = result.permute(1, 0, 2)
        result = result.reshape(B, T_future, J, C)
        result = result.permute(0, 2, 3, 1).contiguous()

        if debug:
            disp = (result[:, :, :, 1:] - result[:, :, :, :-1]).abs().mean().item()
            print(f"[FORWARD] V2: disp={disp:.6f}")

        self._forward_count += 1
        return result

    def interface(self, x, timesteps, y):
        """Diffusion training interface."""
        batch_size = x.shape[0]
        signwriting_image = y['sign_image']
        past_motion = y['input_pose']

        if self.cond_mask_prob > 0:
            keep_batch_idx = torch.rand(batch_size, device=past_motion.device) < (1 - self.cond_mask_prob)
            past_motion = past_motion * keep_batch_idx.view((batch_size, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, signwriting_image)


class EmbedSignWriting(nn.Module):
    """
    SignWriting image encoder using CLIP vision model.
    
    Encodes SignWriting symbol images into latent embeddings that
    condition the diffusion model on the target sign to generate.
    
    Args:
        num_latent_dims: Output embedding dimension
        embedding_arch: CLIP model architecture to use
        freeze_clip: Whether to freeze CLIP parameters (default: True for backward compatibility)
                     Set to False to allow CLIP to learn SignWriting-specific features
    """

    def __init__(self, 
                 num_latent_dims: int, 
                 embedding_arch: str = 'openai/clip-vit-base-patch32',
                 freeze_clip: bool = True):
        super().__init__()
        self.model = CLIPModel.from_pretrained(embedding_arch)

        if freeze_clip:
            for param in self.model.parameters():
                param.requires_grad = False

        self.proj = None
        # Project to target dimension if needed
        if (num_embedding_dims := self.model.visual_projection.out_features) != num_latent_dims:
            self.proj = nn.Linear(num_embedding_dims, num_latent_dims)

    def forward(self, image_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode SignWriting images to embeddings.
        
        Args:
            image_batch: Input images [B, 3, H, W]
            
        Returns:
            Embeddings [B, D]
        """
        embeddings_batch = self.model.get_image_features(pixel_values=image_batch)
        if self.proj is not None:
            embeddings_batch = self.proj(embeddings_batch)
        return embeddings_batch
