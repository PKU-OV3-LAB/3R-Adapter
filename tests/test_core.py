import pytest
import torch

from adapter3r import Adapter3R


def test_grid_layouts_match_documented_coordinates():
    legacy = Adapter3R(4, 2, grid_layout="legacy")
    fixed = Adapter3R(4, 2, grid_layout="pixel_center")
    assert torch.allclose(
        legacy.create_grid(1, 2, 2, torch.device("cpu")),
        torch.tensor([[[[-1.0, -1.0], [1.0, -1.0]], [[-1.0, 1.0], [1.0, 1.0]]]]),
    )
    assert torch.allclose(
        fixed.create_grid(1, 2, 2, torch.device("cpu")),
        torch.tensor([[[[-0.5, -0.5], [0.5, -0.5]], [[-0.5, 0.5], [0.5, 0.5]]]]),
    )


def test_forward_shape_and_checkpoint_compatibility():
    torch.manual_seed(1)
    model = Adapter3R(8, 4, K_sample=2, grid_layout="pixel_center")
    x = torch.randn(2, 6, 8)
    refs = torch.randn(2, 3, 6, 8)
    out = model(x, refs, (4, 6), 2)
    assert out.shape == x.shape
    clone = Adapter3R(8, 4, K_sample=2, grid_layout="legacy")
    assert clone.load_state_dict(model.state_dict(), strict=True).missing_keys == []


def test_forward_rejects_grid_mismatch():
    model = Adapter3R(8, 4)
    with pytest.raises(ValueError, match="patch grid"):
        model(
            torch.randn(1, 5, 8),
            torch.randn(1, 2, 5, 8),
            (4, 4),
            2,
        )


def test_sampling_layout_reference_major_expands_sample_before_query(
    monkeypatch,
):
    captured = {}

    def fake_grid_sample(inputs, grid, **kwargs):
        captured["grid"] = grid.detach().clone()
        return inputs.new_zeros(
            inputs.shape[0], inputs.shape[1], grid.shape[1], grid.shape[2]
        )

    monkeypatch.setattr("adapter3r.core.F.grid_sample", fake_grid_sample)
    torch.manual_seed(12)
    model = Adapter3R(
        4, 2, K_sample=2, sampling_layout="reference_major"
    ).eval()
    with torch.no_grad():
        model.offset_head.weight.zero_()
        model.offset_head.bias.zero_()
    x, refs = torch.randn(1, 4, 4), torch.randn(1, 2, 4, 4)
    model(x, refs, (4, 4), 2)
    locations = model.create_grid(1, 2, 2, x.device).reshape(1, 4, 1, 2).expand(-1, -1, 2, -1)
    expected = (
        locations.unsqueeze(1)
        .expand(-1, 2, -1, -1, -1)
        .permute(0, 1, 3, 2, 4)
        .reshape(4, 4, 1, 2)
    )
    torch.testing.assert_close(captured["grid"], expected)
