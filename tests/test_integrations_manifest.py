import torch

from adapter3r import (
    Adapter3R,
    AdapterManifest,
    adapter_state_dict,
    unwrap_checkpoint,
)
from adapter3r.integrations import (
    apply_human3r_adapter,
    apply_pi3_adapter,
    make_pi3_adapters,
)


def test_manifest_round_trip(tmp_path):
    path = tmp_path / "manifest.json"
    expected = AdapterManifest(
        grid_layout="pixel_center", sampling_layout="reference_major"
    )
    expected.dump(path)
    assert AdapterManifest.load(path) == expected


def test_checkpoint_unwrap_and_filter():
    tensor = torch.tensor(1.0)
    assert unwrap_checkpoint({"model": {"x": tensor}})["x"] is tensor
    state = {"backbone.x": tensor, "dec_adapters.0.gate": tensor}
    assert list(adapter_state_dict({"state_dict": state})) == [
        "dec_adapters.0.gate"
    ]


def test_pi3_integration_preserves_special_tokens():
    torch.manual_seed(2)
    adapter = Adapter3R(4, 2, K_sample=1, grid_layout="pixel_center")
    hidden = torch.randn(2, 6, 4)
    special = hidden[:, :2].clone()
    out = apply_pi3_adapter(
        hidden,
        adapter,
        1,
        2,
        (4, 4),
        2,
        patch_start_idx=2,
        radius=1,
    )
    assert out.shape == hidden.shape
    assert torch.equal(out[:, :2], special)


def test_pi3_public_default_matches_released_decoder_layers():
    adapters = make_pi3_adapters(embed_dim=4, reduction_dim=2)
    assert list(adapters) == [str(index) for index in range(17, 25)]


def test_human3r_integration_preserves_layout():
    adapter = Adapter3R(4, 2, K_sample=1, grid_layout="pixel_center")
    tokens = torch.randn(1, 3, 4, 4)
    out = apply_human3r_adapter(
        tokens, adapter, (4, 4), 2, radius=1
    )
    assert out.shape == tokens.shape
