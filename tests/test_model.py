import pytest
import torch

from fallsense.features import N_FEATURES
from fallsense.models import MODELS, build_model


@pytest.mark.parametrize("name", sorted(MODELS))
def test_output_shape(name: str) -> None:
    model = build_model(name).eval()
    assert model(torch.randn(8, 30, N_FEATURES)).shape == (8, 4)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_overfits_small_batch(name: str) -> None:
    torch.manual_seed(0)
    model = build_model(name)
    x = torch.randn(64, 30, N_FEATURES)
    y = torch.randint(0, 4, (64,))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = torch.nn.CrossEntropyLoss()
    for _ in range(50):
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        opt.step()
    assert loss.item() < 0.1
