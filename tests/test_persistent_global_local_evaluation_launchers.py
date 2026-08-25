from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FORECAST = ROOT / "cluster/post_ecrd_old_85604_persistent_global_local_forecast.sbatch"
SCORE = ROOT / "cluster/post_ecrd_old_85604_persistent_global_local_score.sbatch"
MANIFEST = ROOT / "paper0/manifests/post_ecrd_old_85604_persistent_global_local_physics_evaluation.json"
GENERATOR = ROOT / "paper0/tools/generate_persistent_global_local_forecast.py"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_forecast_launcher_is_one_unconstrained_preemptible_gpu_and_truth_free():
    source = FORECAST.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=" not in source
    assert "generate_persistent_global_local_forecast.py" in source
    assert "score_persistent_global_local_physics.py" not in source
    assert "target_truth_read == false" in source
    assert "physics_diagnostics_scored == false" in source
    assert "/85606" not in source


def test_score_launcher_is_cpu_only_and_consumes_closed_generation_dependency():
    source = SCORE.read_text(encoding="utf-8")
    assert "#SBATCH --partition=preempt" in source
    assert "#SBATCH --gres=gpu" not in source
    assert "PGL_GENERATION_JOB_ID" in source
    assert "forecast_M32_four_frame.h5" in source
    assert "sha256sum \"${FORECAST}\"" in source
    assert "score_persistent_global_local_physics.py" in source
    assert "target_truth_used_during_generation == false" in source
    assert "/85606" not in source


def test_frozen_evaluation_manifest_locks_all_local_sources_before_sampling():
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == "frozen_after_seed1702_state_gate_before_forecast"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["new_nersc_data_access_allowed"] is False
    assert manifest["training_allowed"] is False
    assert manifest["confirmation_seed_training_allowed"] is False
    assert manifest["forecast_population"]["start_count"] == 36
    assert manifest["forecast_population"]["ensemble_members"] == 32
    assert manifest["forecast_population"]["future_frames"] == 4
    assert manifest["sampler"]["steps"] == 18
    assert manifest["physics_gates"]["all_families_required"] is True
    protocol = manifest["protocol"]
    assert _sha256(ROOT / protocol["path"]) == protocol["sha256"]
    for relative, record in manifest["evaluation_code"].items():
        assert _sha256(ROOT / relative) == record["sha256"], relative
    for record in manifest["evidence_locks"].values():
        if "path" in record and not Path(record["path"]).is_absolute():
            assert _sha256(ROOT / record["path"]) == record["sha256"]


def test_checkpoint_identity_uses_canonical_path_and_frozen_sha256():
    source = GENERATOR.read_text(encoding="utf-8")
    assert "result_checkpoint" in source
    assert ").resolve(strict=True)" in source
    assert "PGL_SELECTED_CHECKPOINT_SHA256" in source
