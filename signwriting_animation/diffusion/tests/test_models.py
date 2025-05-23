import torch
import pytest

from signwriting_animation.diffusion.core import models


@pytest.mark.parametrize("latent_dim", [32, 512])
def test_embed_signwriting_model_fwd(latent_dim):
    model = models.EmbedSignWriting(latent_dim=latent_dim,
                                    embedding_arch='openai/clip-vit-base-patch32')
    image_batch = torch.ones(4, 3, 224, 224)

    assert model(image_batch).shape == (4, latent_dim)
