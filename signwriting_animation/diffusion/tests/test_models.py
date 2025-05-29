import torch
import pytest

from signwriting_animation.diffusion.core import models


def test_output_process_model():
    num_keypoints = 586
    num_dims_per_keypoint = 3
    num_frames = 10
    batch_size = 4
    latent_dim = 12

    model = models.OutputProcess(
        input_feats=num_keypoints * num_dims_per_keypoint,
        latent_dim=latent_dim,
        num_keypoints=num_keypoints,
        num_dims_per_keypoint=num_dims_per_keypoint)

    # num_frames, batch_size, d
    x = torch.ones(num_frames, batch_size, latent_dim)

    assert model(x).shape == (batch_size, num_keypoints, num_dims_per_keypoint, num_frames)


@pytest.mark.parametrize("latent_dim", [32, 512])
def test_embed_signwriting_model_fwd(latent_dim):
    model = models.EmbedSignWriting(latent_dim=latent_dim,
                                    embedding_arch='openai/clip-vit-base-patch32')
    image_batch = torch.ones(4, 3, 224, 224)

    assert model(image_batch).shape == (4, latent_dim)
