"""Model B + C (pose_mlp) contract tests (TRAIN-04/TRAIN-05, Plan 03-02).

Torch-only, weight-free, deterministic: torch.manual_seed(0) + synthetic
tensors, no npz, no real training. Pinned contracts:
- Width fail-closed trips (T-03-06 mitigation): Model B's train entry
  raises ValueError NAMING THE OBSERVED WIDTH on anything other than 51
  columns; Model C's on anything other than 59 — and the correct widths
  are accepted end-to-end on synthetic tensors.
- compute_class_weights is PROGRAMMATIC inverse-frequency N/(C * n_c)
  (RESEARCH A7 — never hard-coded) and rejects a zero-support class.
- Loss wiring (locked ablation design): B's loss carries
  label_smoothing=0.05 and NO weight tensor; C's carries the computed
  weight tensor of shape (6,) — only Model C is weighted.
- ResBlock residual forward (x + drop(act(bn(fc(x))))) + shapes;
  ResidualMLP (2,59) -> (2,6); PoseMLP (2,51) -> (2,6).
- The pose_mlp.pt checkpoint payload is primitives-only and loads under
  torch 2.15's DEFAULT weights_only=True loader (Pitfall 1 inverted —
  any UnpicklingError is a contract violation).
- The B optimizer stack: AdamW (param groups match the model) +
  CosineAnnealingLR(T_max = the passed max_epochs) stepping max_epochs
  times drives lr to eta_min.
"""

import numpy as np
import pytest
import torch

from src.models.data import canonical_classes, fit_label_encoder
from src.models.pose_mlp import (
    PoseMLP,
    ResBlock,
    ResidualMLP,
    compute_class_weights,
    load_pose_mlp_checkpoint,
    make_b_loss,
    make_b_optimizers,
    make_c_loss,
    save_pose_mlp_checkpoint,
    train_pose_mlp,
    train_residual_mlp,
)

LE = fit_label_encoder(canonical_classes())


def _synthetic(width: int, rows: int = 36, n_classes: int = 6):
    """Deterministic synthetic (X, y): a torch tensor (rows, width) plus
    int labels with every class represented (rows % n_classes == 0, 6 per
    class — stratifiable by StratifiedKFold(5))."""
    torch.manual_seed(0)
    X = torch.randn(rows, width)
    y = np.arange(rows) % n_classes
    return X, y


# ---------------------------------------------------------------------------
# Test 1: width contracts (fail-closed trips + accepted widths)
# ---------------------------------------------------------------------------


def test_train_pose_mlp_rejects_width_50_naming_observed_width():
    X, y = _synthetic(50)
    with pytest.raises(ValueError) as excinfo:
        train_pose_mlp(X, y, LE)
    assert "observed width 50" in str(excinfo.value)
    assert "51" in str(excinfo.value)  # the contract width is named too


def test_train_pose_mlp_accepts_width_51():
    X, y = _synthetic(51)
    model, meta = train_pose_mlp(X, y, LE, max_epochs=2)
    assert isinstance(model, PoseMLP)
    assert meta["classes"] == list(LE.classes_)
    assert "AdamW" in meta["optimizer"] and "CosineAnnealingLR" in meta["optimizer"]
    assert "label_smoothing" in meta["loss"] and "NO class" in meta["loss"]


def test_train_residual_mlp_rejects_width_58_naming_observed_width():
    X, y = _synthetic(58)
    with pytest.raises(ValueError) as excinfo:
        train_residual_mlp(X, y, LE)
    assert "observed width 58" in str(excinfo.value)
    assert "59" in str(excinfo.value)


def test_train_residual_mlp_accepts_width_59():
    X, y = _synthetic(59)
    model, meta = train_residual_mlp(X, y, LE, max_epochs=2)
    assert isinstance(model, ResidualMLP)
    assert meta["classes"] == list(LE.classes_)
    assert "AdamW" in meta["optimizer"] and "no scheduler" in meta["optimizer"]
    assert "inverse-frequency" in meta["loss"]


# ---------------------------------------------------------------------------
# Test 2: class weights — programmatic inverse frequency, zero-support trip
# ---------------------------------------------------------------------------


def test_compute_class_weights_inverse_frequency_hand_computed():
    # counts {0: 4, 1: 8, 2: 2}, N=14, C=3 -> [N/(3*4), N/(3*8), N/(3*2)]
    # = [14/12, 14/24, 14/6] = [7/6, 7/12, 7/3]. (The plan's "= [1.0, 0.5,
    # 2.0]" expansion was arithmetically impossible for these counts —
    # N/(C*n_c) with N=14 is the locked formula authority, TRAIN-05 /
    # RESEARCH A7; Rule 1 plan-bug fix, 03-02.)
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2])
    w = compute_class_weights(y, 3)
    assert w.dtype == torch.float32
    expected = torch.tensor([14 / 12, 14 / 24, 14 / 6])
    assert torch.allclose(w, expected, atol=1e-6)


def test_compute_class_weights_rejects_zero_support_class():
    y = np.array([0, 0, 1, 1, 1, 1])  # class 2 has zero train support
    with pytest.raises(ValueError, match=r"zero train support"):
        compute_class_weights(y, 3)


# ---------------------------------------------------------------------------
# Test 3: loss wiring — B smoothed/unweighted, C weighted
# ---------------------------------------------------------------------------


def test_b_loss_label_smoothing_only_no_weight_tensor():
    loss = make_b_loss()
    assert loss.weight is None
    assert loss.label_smoothing == 0.05


def test_c_loss_carries_computed_weight_tensor():
    X, y = _synthetic(59)
    loss = make_c_loss(y)
    assert loss.weight is not None
    assert tuple(loss.weight.shape) == (6,)
    assert torch.allclose(loss.weight, compute_class_weights(y, 6))


# ---------------------------------------------------------------------------
# Test 4: ResBlock / ResidualMLP forward shapes + residual identity
# ---------------------------------------------------------------------------


def test_resblock_forward_shape():
    torch.manual_seed(0)
    block = ResBlock(dim=8, p=0.25)
    out = block(torch.randn(4, 8))
    assert out.shape == (4, 8)


def test_resblock_residual_path_carries_input():
    # Zero the block transform (fc -> 0, bn -> identity, no dropout): the
    # residual addition must return the input EXACTLY (x + 0) — proof the
    # skip connection carries x.
    torch.manual_seed(0)
    block = ResBlock(dim=8, p=0.0)
    with torch.no_grad():
        block.fc.weight.zero_()
        block.fc.bias.zero_()
        block.bn.weight.fill_(1.0)
        block.bn.bias.zero_()
    block.eval()
    x = torch.randn(4, 8)
    assert torch.allclose(block(x), x, atol=1e-7)


def test_residual_mlp_forward_shape():
    torch.manual_seed(0)
    model = ResidualMLP()
    out = model(torch.randn(2, 59))
    assert out.shape == (2, 6)


# ---------------------------------------------------------------------------
# Test 5: Model B forward shape
# ---------------------------------------------------------------------------


def test_pose_mlp_forward_shape():
    torch.manual_seed(0)
    model = PoseMLP()
    out = model(torch.randn(2, 51))
    assert out.shape == (2, 6)


# ---------------------------------------------------------------------------
# Test 6: checkpoint roundtrip under the DEFAULT weights_only mode
# ---------------------------------------------------------------------------


def test_pose_mlp_checkpoint_roundtrip_under_default_weights_only(tmp_path):
    torch.manual_seed(0)
    model = PoseMLP()
    path = tmp_path / "pose_mlp.pt"
    save_pose_mlp_checkpoint(path, model, LE, code_version="03-02-test")
    # DEFAULT weights_only mode — NO weights_only argument. The payload is
    # primitives-only (str/list/int/tensor); any UnpicklingError here is a
    # contract violation (RESEARCH Pattern 4 / Pitfall 1 inverted).
    ck = torch.load(path)
    assert ck["arch"] == "pose_mlp"
    assert ck["input_dim"] == 51
    assert ck["n_classes"] == 6
    assert ck["classes"] == list(LE.classes_)
    assert ck["code_version"] == "03-02-test"
    assert "state_dict" in ck
    rebuilt = PoseMLP()
    rebuilt.load_state_dict(ck["state_dict"])
    assert rebuilt(torch.randn(3, 51)).shape == (3, 6)
    # The load-side helper uses the same default mode.
    ck2 = load_pose_mlp_checkpoint(path)
    assert ck2["arch"] == "pose_mlp"


# ---------------------------------------------------------------------------
# Test 7: B optimizer stack — AdamW + CosineAnnealingLR(T_max=max_epochs)
# ---------------------------------------------------------------------------


def test_b_optimizer_stack_adamw_with_cosine_schedule():
    torch.manual_seed(0)
    model = PoseMLP()
    optimizer, scheduler = make_b_optimizers(model, 60)
    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == 1e-3
    assert optimizer.param_groups[0]["weight_decay"] == 1e-4
    assert list(optimizer.param_groups[0]["params"]) == list(model.parameters())
    assert isinstance(scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
    assert scheduler.T_max == 60
    for _ in range(60):
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-5, rel=1e-6)
