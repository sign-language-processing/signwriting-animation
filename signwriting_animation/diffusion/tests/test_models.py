import torch
import pytest

from signwriting_animation.diffusion.core import models


def test_output_process_mlp_model():
    num_keypoints = 586
    num_dims_per_keypoint = 3
    num_frames = 10
    batch_size = 4
    num_latent_dims = 12

    model = models.OutputProcessMLP(
        num_latent_dims=num_latent_dims,
        num_keypoints=num_keypoints,
        num_dims_per_keypoint=num_dims_per_keypoint,
        hidden_dim=512)

    x = torch.ones(num_frames, batch_size, num_latent_dims)

    assert model(x).shape == (batch_size, num_keypoints, num_dims_per_keypoint, num_frames)


def test_output_process_model():
    num_keypoints = 586
    num_dims_per_keypoint = 3
    num_frames = 10
    batch_size = 4
    num_latent_dims = 12

    model = models.OutputProcess(
        latent_dim=num_latent_dims,
        num_keypoints=num_keypoints,
        num_dims_per_keypoint=num_dims_per_keypoint)

    x = torch.ones(num_frames, batch_size, num_latent_dims)

    assert model(x).shape == (batch_size, num_keypoints, num_dims_per_keypoint, num_frames)


@pytest.mark.parametrize("latent_dim", [32, 512])
def test_embed_signwriting_model_output_shape(latent_dim):
    model = models.EmbedSignWriting(num_latent_dims=latent_dim,
                                    embedding_arch='openai/clip-vit-base-patch32')
    batch_size = 4
    image_batch = torch.ones(batch_size, 3, 224, 224)

    assert model(image_batch).shape == (1, batch_size, latent_dim)

