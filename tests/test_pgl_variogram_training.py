from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tcv_diagnostics.pgl_variogram import IndexedPairBank
from tcv_diagnostics.pgl_variogram_training import (
    PGL_VARIOGRAM_CONTROL_STARTS,
    VariogramControlMagnitudes,
    VariogramScreenConfig,
    VariogramTerms,
    arm_objective,
    fractional_periodic_roll,
    load_pair_banks,
    load_training_transport_truth,
    save_pair_banks,
    save_training_transport_truth,
    training_transport_window,
)


ROOT = Path(__file__).resolve().parents[1]


def _bank(name: str) -> IndexedPairBank:
    return IndexedPairBank(
        left=np.asarray([0, 2], dtype=np.int64),
        right=np.asarray([1, 3], dtype=np.int64),
        weight=np.asarray([0.5, 0.5], dtype=np.float64),
        group=np.asarray([0, 1], dtype=np.int64),
        group_name=name,
        group_values=(1.0, 2.0),
        metadata={"seed": 856040828, "kind": name},
    )


def _controls() -> VariogramControlMagnitudes:
    return VariogramControlMagnitudes(
        edm=10.0,
        field_spatial=2.0,
        field_temporal=4.0,
        transport_spatial=(1.0, 2.0, 4.0, 8.0),
        transport_temporal=(2.0, 4.0, 8.0, 16.0),
    )


def _terms() -> VariogramTerms:
    one = torch.tensor(1.0, requires_grad=True)
    controls = _controls()
    return VariogramTerms(
        edm=torch.tensor(10.0, requires_grad=True),
        field_spatial=torch.tensor(controls.field_spatial, requires_grad=True),
        field_temporal=torch.tensor(controls.field_temporal, requires_grad=True),
        transport_spatial=tuple(torch.tensor(value, requires_grad=True) for value in controls.transport_spatial),
        transport_temporal=tuple(torch.tensor(value, requires_grad=True) for value in controls.transport_temporal),
        ordinary={"sentinel": one},
    )


def test_screen_config_is_exact_and_truthfully_labels_transport_loss() -> None:
    expected = {
        "A": (False, False, False),
        "B": (True, False, False),
        "C": (False, True, True),
        "D": (True, True, True),
    }
    for arm, values in expected.items():
        config = VariogramScreenConfig(mode="screen", arm=arm)
        assert (
            config.field_variogram_enabled,
            config.transport_variogram_enabled,
            config.physics_derived_training_loss_used,
        ) == values
        assert config.training_windows == 428
        assert config.optimizer_updates == 214
        assert config.to_record()["fixed_final_ema_no_checkpoint_selection"] is True
    smoke = VariogramScreenConfig(mode="smoke", arm="D")
    assert smoke.training_windows == 2
    assert smoke.optimizer_updates == 1


def test_arm_objectives_have_equal_initial_ten_percent_budget() -> None:
    controls = _controls()
    expected = {"A": 10.0, "B": 11.0, "C": 11.0, "D": 11.0}
    for arm, value in expected.items():
        terms = _terms()
        objective, field, transport = arm_objective(arm, terms, controls)
        assert float(field.detach()) == pytest.approx(1.0)
        assert float(transport.detach()) == pytest.approx(1.0)
        assert float(objective.detach()) == pytest.approx(value)
        objective.backward()
        assert terms.edm.grad is not None


def test_control_record_requires_exact_population_and_all_positive() -> None:
    controls = _controls()
    record = controls.to_record()
    assert record["current_frames"] == list(PGL_VARIOGRAM_CONTROL_STARTS)
    assert VariogramControlMagnitudes.from_record(record) == controls
    bad = json.loads(json.dumps(record))
    bad["transport_spatial"]["particle"] = 0.0
    with pytest.raises(ValueError, match="positive"):
        VariogramControlMagnitudes.from_record(bad)


def test_pair_bank_round_trip_is_hash_closed(tmp_path: Path) -> None:
    banks = {
        "field_spatial": _bank("field spatial"),
        "field_temporal": _bank("field temporal"),
        "transport_spatial": _bank("transport spatial"),
        "transport_temporal": _bank("transport temporal"),
    }
    path = tmp_path / "pairs.npz"
    record = save_pair_banks(path, banks)
    restored = load_pair_banks(path, expected_sha256=record["sha256"])
    assert {name: bank.sha256 for name, bank in restored.items()} == {
        name: bank.sha256 for name, bank in banks.items()
    }
    with pytest.raises(ValueError, match="SHA-256"):
        load_pair_banks(path, expected_sha256="0" * 64)


def test_fractional_roll_matches_integer_numpy_roll_and_round_trip() -> None:
    rng = np.random.default_rng(7)
    values = rng.normal(size=(3, 81)).astype(np.float64)
    assert np.allclose(fractional_periodic_roll(values, 9.0), np.roll(values, 9, -1), atol=2e-14)
    fractional = fractional_periodic_roll(values, 9.25)
    restored = fractional_periodic_roll(fractional, -9.25)
    assert np.allclose(restored, values, rtol=1e-12, atol=1e-12)


def test_transport_window_uses_physical_fractional_shift() -> None:
    truth = np.zeros((432, 4, 16, 81), dtype=np.float32)
    truth[:, :, :, 0] = 1.0
    observed = training_transport_window(truth, current_frame=10, model_roll=0)
    assert observed.shape == (5, 4, 16, 81)
    assert torch.equal(observed[..., 0], torch.ones_like(observed[..., 0]))
    rolled = training_transport_window(truth, current_frame=10, model_roll=8)
    expected = fractional_periodic_roll(truth[10:15], 8.0 * 81.0 / 88.0)
    assert np.allclose(rolled.numpy(), expected)


def test_native_transport_truth_round_trip_is_hash_closed(tmp_path: Path) -> None:
    values = np.zeros((432, 4, 16, 81), dtype=np.float64)
    values[:, 0] = 0.25
    record = save_training_transport_truth(tmp_path / "truth.npz", values)
    restored = load_training_transport_truth(
        Path(record["path"]), expected_sha256=record["sha256"]
    )
    assert np.array_equal(restored, values)


def test_launchers_are_small_gpu_jobs_and_freeze_the_exact_sequence() -> None:
    preflight = (ROOT / "cluster/post_ecrd_old_85604_pgl_variogram_preflight.sbatch").read_text()
    smoke = (ROOT / "cluster/post_ecrd_old_85604_pgl_variogram_smoke.sbatch").read_text()
    screen = (ROOT / "cluster/post_ecrd_old_85604_pgl_variogram_screen.sbatch").read_text()
    for source in (preflight, smoke, screen):
        assert "#SBATCH --gres=gpu:1" in source
        assert "#SBATCH --cpus-per-task=4" in source
        assert "#SBATCH --mem=32G" in source
        assert "PAPER0_EXPECTED_COMMIT" in source
        assert "status --porcelain --untracked-files=all" in source
    assert "#SBATCH --array=0-3%4" in smoke
    assert "--mode smoke" in smoke
    assert "PAPER0_PGL_VARIOGRAM_PREFLIGHT_JOB_ID" in smoke
    assert "#SBATCH --array=0-3%4" in screen
    assert "--mode screen" in screen
    assert "PAPER0_PGL_VARIOGRAM_SMOKE_JOB_ID" in screen
    assert '.completed_optimizer_updates == 214' in screen
    assert "WANDB_MODE=online" in smoke and "WANDB_MODE=online" in screen
