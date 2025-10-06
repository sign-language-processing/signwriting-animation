from pathlib import Path
import sys, types, importlib
import torch
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

class DummyModel:
    def __init__(self, *args, **kwargs):
        pass
    def forward(self, x_bjct, timesteps, past_bjct, sign_img):
        return x_bjct  # identity, correct BJCT shape

stub_core_models = types.ModuleType("signwriting_animation.diffusion.core.models")
stub_core_models.SignWritingToPoseDiffusion = DummyModel
sys.modules["signwriting_animation.diffusion.core.models"] = stub_core_models

stub_mpl = types.ModuleType("matplotlib")
stub_plt = types.ModuleType("matplotlib.pyplot")
def _noop(*a, **k): pass
# minimal set of functions used in on_fit_end
stub_plt.figure = _noop
stub_plt.plot = _noop
stub_plt.xlabel = _noop
stub_plt.legend = _noop
stub_plt.tight_layout = _noop
stub_plt.savefig = _noop
stub_plt.close = _noop
sys.modules.setdefault("matplotlib", stub_mpl)
sys.modules["matplotlib.pyplot"] = stub_plt

ml = importlib.import_module("signwriting_animation.scripts.minimal_loop")

@pytest.fixture
def dummy_batch():
    """Tiny fake batch matching the minimal loop expected keys/shapes."""
    B, T, J, C = 2, 5, 3, 2
    return {
        "data": torch.zeros(B, T, J, C),
        "conditions": {
            "input_pose": torch.zeros(B, T, J, C),
            "target_mask": torch.ones(B, T),
            "sign_image": torch.zeros(B, 3, 224, 224),
        },
    }


def test_btjc_bjct_roundtrip():
    x = torch.randn(2, 5, 3, 2)      # [B,T,J,C]
    y = ml.btjc_to_bjct(x)           # [B,J,C,T]
    z = ml.bjct_to_btjc(y)           # [B,T,J,C]
    assert z.shape == x.shape and torch.allclose(z, x)


def test_training_validation_step_smoke(dummy_batch):
    model = ml.LitMinimal()          # uses DummyModel via stub
    loss = model.training_step(dummy_batch, 0)
    assert loss.ndim == 0 and torch.isfinite(loss)

    model.validation_step(dummy_batch, 0)
    assert len(model.val_losses) == 1
    assert len(model.val_dtws) == 1