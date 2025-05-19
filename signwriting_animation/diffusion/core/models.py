from typing import Optional
import torch
import torch.nn as nn

from CAMDM.network.models import PositionalEncoding, TimestepEmbedder, MotionProcess, TrajProcess


class SignWritingToPoseDiffusion(nn.Module):
    def __init__(self,
                 input_feats: int,
                 nstyles: int,
                 njoints: int,
                 nfeats: int,
                 rot_req,
                 clip_len: int,
                 latent_dim: int = 256,
                 ff_size: int = 1024,
                 num_layers: int = 8,
                 num_heads: int = 4,
                 dropout: float = 0.2,
                 ablation: Optional[str] = None,
                 activation: str = "gelu",
                 legacy: bool = False,
                 arch: str = "trans_enc",
                 cond_mask_prob: float = 0,
                 device: Optional[torch.device] = None):
        """
        Args:
            input_feats: Number of input features (keypoints * dimensions).
            nstyles:
            njoints:
            nfeats:
            rot_req:
            clip_len:
            latent_dim: Dimension of the latent space.
            ff_size: Feed-forward network size.
            num_layers: Number of Transformer layers.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
            ablation: Ablation study parameter.
            activation: Activation function.
            legacy: Legacy flag.
            arch: Architecture type: "trans_enc", "trans_dec", or "gru".
            cond_mask_prob: Condition mask probability for classifier-free guidance (CFG).
            device: Device to run the model.
        """
        super().__init__()

        self.legacy = legacy
        self.training = True

        self.rot_req = rot_req
        self.nfeats = nfeats
        self.njoints = njoints
        self.clip_len = clip_len
        self.input_feats = input_feats

        self.latent_dim = latent_dim
        self.ff_size = ff_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout
        self.ablation = ablation
        self.activation = activation
        self.cond_mask_prob = cond_mask_prob
        self.arch = arch

        # local conditions
        self.future_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.past_motion_process = MotionProcess(self.input_feats, self.latent_dim)
        self.traj_trans_process = TrajProcess(2, self.latent_dim)
        self.traj_pose_process = TrajProcess(6, self.latent_dim)
        self.sequence_pos_encoder = PositionalEncoding(self.latent_dim, self.dropout)

        # global conditions
        self.embed_style = EmbedStyle(nstyles, self.latent_dim)
        self.embed_timestep = TimestepEmbedder(self.latent_dim, self.sequence_pos_encoder)

        if self.arch == 'trans_enc':
            print("TRANS_ENC init")

            seqTransEncoderLayer = nn.TransformerEncoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=self.activation)

            self.seqEncoder = nn.TransformerEncoder(seqTransEncoderLayer,
                                                    num_layers=self.num_layers)
        elif self.arch == 'trans_dec':
            print("TRANS_DEC init")
            seqTransDecoderLayer = nn.TransformerDecoderLayer(d_model=self.latent_dim,
                                                              nhead=self.num_heads,
                                                              dim_feedforward=self.ff_size,
                                                              dropout=self.dropout,
                                                              activation=activation)
            self.seqEncoder = nn.TransformerDecoder(seqTransDecoderLayer,
                                                    num_layers=self.num_layers)

        elif self.arch == 'gru':
            print("GRU init")
            self.seqEncoder = nn.GRU(self.latent_dim, self.latent_dim, num_layers=self.num_layers, batch_first=True)
        else:
            raise ValueError('Please choose correct architecture [trans_enc, trans_dec, gru]')

        self.output_process = OutputProcessMLP(self.input_feats, self.latent_dim, self.njoints, self.nfeats)

    def forward(self, x, timesteps, past_motion, traj_pose, traj_trans, style_idx):
        bs, njoints, nfeats, nframes = x.shape

        time_emb = self.embed_timestep(timesteps)  # [1, bs, L]
        style_emb = self.embed_style(style_idx).unsqueeze(0)  # [1, bs, L]
        traj_trans_emb = self.traj_trans_process(traj_trans)  # [N/2, bs, L]
        traj_pose_emb = self.traj_pose_process(traj_pose)  # [N/2, bs, L]
        past_motion_emb = self.past_motion_process(past_motion)  # [past_frames, bs, L]

        future_motion_emb = self.future_motion_process(x)

        xseq = torch.cat((time_emb,
                          style_emb,
                          traj_trans_emb,
                          traj_pose_emb,
                          past_motion_emb,
                          future_motion_emb), axis=0)

        xseq = self.sequence_pos_encoder(xseq)
        output = self.seqEncoder(xseq)[-nframes:]
        output = self.output_process(output)
        return output

    def interface(self, x, timesteps, y=None):
        """
            x: [batch_size, frames, njoints, nfeats], denoted x_t in the paper
            timesteps: [batch_size] (int)
            y: a dictionary containing conditions
        """
        bs, njoints, nfeats, nframes = x.shape

        style_idx = y['style_idx']
        past_motion = y['past_motion']
        traj_pose = y['traj_pose']
        traj_trans = y['traj_trans']

        # CFG on past motion
        keep_batch_idx = torch.rand(bs, device=past_motion.device) < (1 - self.cond_mask_prob)
        past_motion = past_motion * keep_batch_idx.view((bs, 1, 1, 1))

        return self.forward(x, timesteps, past_motion, traj_pose, traj_trans, style_idx)


class OutputProcessMLP(nn.Module):
    """
    Output process for the Sign Language Pose Diffusion model.

    Obtained module from https://github.com/sign-language-processing/fluent-pose-synthesis
    """
    def __init__(self, input_feats, latent_dim, njoints, nfeats, hidden_dim=512): # add hidden_dim as parameter
        super().__init__()
        self.input_feats = input_feats
        self.latent_dim = latent_dim
        self.njoints = njoints
        self.nfeats = nfeats
        self.hidden_dim = hidden_dim # store hidden dimension

        # MLP layers
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.SiLU(),
            nn.Linear(self.hidden_dim // 2, self.input_feats)
        )

    def forward(self, output):
        nframes, bs, d = output.shape
        output = self.mlp(output)  # use MLP instead of single linear layer
        output = output.reshape(nframes, bs, self.njoints, self.nfeats)
        output = output.permute(1, 2, 3, 0)
        return output


class EmbedStyle(nn.Module):
    def __init__(self, num_actions, latent_dim):
        super().__init__()
        self.action_embedding = nn.Parameter(torch.randn(num_actions, latent_dim))

    def forward(self, input):
        idx = input.to(torch.long)
        output = self.action_embedding[idx]
        return output
