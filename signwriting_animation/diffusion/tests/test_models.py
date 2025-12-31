import torch
import pytest

from signwriting_animation.diffusion.core import models


@pytest.mark.parametrize("latent_dim", [32, 512])
def test_embed_signwriting_model_output_shape(latent_dim):
    model = models.EmbedSignWriting(num_latent_dims=latent_dim,
                                    embedding_arch='openai/clip-vit-base-patch32')
    batch_size = 4
    image_batch = torch.ones(batch_size, 3, 224, 224)

    # Output shape is [B, D] (was [1, B, D] in old API)
    assert model(image_batch).shape == (batch_size, latent_dim)


def test_signwriting_to_pose_diffusion_fwd():
    batch_size = 4
    num_past_frames = 40
    num_future_frames = 20
    num_keypoints = 178  # After reduce_holistic (was 586)
    num_dims_per_keypoint = 3

    model = models.SignWritingToPoseDiffusion(num_keypoints=num_keypoints,
                                              num_dims_per_keypoint=num_dims_per_keypoint)

    sign_image = torch.ones(batch_size, 3, 224, 224)    # CLIP input image resolution
    input_pose_seq = torch.ones(batch_size, num_keypoints, num_dims_per_keypoint, num_past_frames)

    x_t = torch.ones(batch_size, num_keypoints, num_dims_per_keypoint, num_future_frames)

    timesteps = torch.tensor([0, 45, 12, 999])

    assert timesteps.shape == (batch_size, )
    assert timesteps.dtype == torch.int64
    assert timesteps.max() >= 0     # min timestep value

    output = model(x=x_t,
                   timesteps=timesteps,
                   past_motion=input_pose_seq,
                   signwriting_im_batch=sign_image)
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape == x_t.shape


def test_signwriting_to_pose_diffusion_interface_with_conditional_input():
    batch_size = 4
    num_past_frames = 40
    num_future_frames = 20
    num_keypoints = 178  # After reduce_holistic (was 586)
    num_dims_per_keypoint = 3

    model = models.SignWritingToPoseDiffusion(num_keypoints=num_keypoints,
                                              num_dims_per_keypoint=num_dims_per_keypoint)

    sign_image = torch.ones(batch_size, 3, 224, 224)
    input_pose_seq = torch.ones(batch_size, num_keypoints, num_dims_per_keypoint, num_past_frames)
    conditions = dict(sign_image=sign_image, input_pose=input_pose_seq)

    x_t = torch.ones(batch_size, num_keypoints, num_dims_per_keypoint, num_future_frames)

    timesteps = torch.tensor([0, 45, 12, 999])

    output = model.interface(x=x_t,
                             timesteps=timesteps,
                             y=conditions)
    if isinstance(output, tuple):
        output = output[0]
    assert output.shape == x_t.shape
