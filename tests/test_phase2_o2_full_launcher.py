"""Prospective locks for the six full C5P O2 training runs."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase2_o2_train_full.sbatch"
MANIFEST = ROOT / "paper0/manifests/phase2_c5p_o2_full_runs_85604.json"
SMOKE = ROOT / "paper0/results/phase2_o2_gpu_smoke_6894971.json"
LOCKED = (
    MANIFEST,
    ROOT / "paper0/manifests/phase2_c5p_o2_continuation_85604.json",
    ROOT / "paper0/protocol/PHASE2_C5P_O2_CONTINUATION_PROTOCOL.md",
    SMOKE,
    ROOT / "paper0/results/phase2_matched_o1_finalize_r2_6894863.json",
    ROOT / "paper0/results/phase2_model_dataset_6893525.json",
    ROOT / "paper0/results/phase2_model_dataset_normalization_6893525.json",
    ROOT / "src/tcv_diagnostics/models/layers.py",
    ROOT / "src/tcv_diagnostics/models/dcae.py",
    ROOT / "src/tcv_diagnostics/models/__init__.py",
    ROOT / "src/tcv_diagnostics/models/vit.py",
    ROOT / "src/tcv_diagnostics/models/o2.py",
    ROOT / "src/tcv_diagnostics/model_training_data.py",
    ROOT / "src/tcv_diagnostics/o2_training_data.py",
    ROOT / "src/tcv_diagnostics/o2_training.py",
    ROOT / "src/tcv_diagnostics/codec_training.py",
    ROOT / "src/tcv_diagnostics/wandb_tracking.py",
    ROOT / "paper0/tools/train_o2.py",
)


def test_full_launcher_is_nonpreemptible_four_h200_rocky9_bash() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        "#SBATCH --partition=gpuxl",
        "#SBATCH --qos=gen",
        "#SBATCH --gres=gpu:h200:4",
        "#SBATCH --constraint=h200",
        "#SBATCH --time=12:00:00",
        "#SBATCH --no-requeue",
        '"${VERSION_ID%%.*}" != "9"',
        "PAPER0_EXPECTED_COMMIT",
        "status --porcelain --untracked-files=all",
        "Refusing to overwrite existing result directory",
    ):
        assert required in text
    assert "#SBATCH --partition=gpupreempt" not in text


def test_full_manifest_is_frozen_six_run_c5p_matrix() -> None:
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        "1f49699396a8529a1ee42fd5b0d746c75c98a750c3e376f3720c23dfae4ed203"
    )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == (
        "frozen_execution_revision_after_zero_compute_H100_hold_before_full_O2_training"
    )
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["data"]["zperiod"] == 5
    assert manifest["data"]["mode_mapping"] == "n=5k"
    assert manifest["data"]["absolute_time_input_allowed"] is False
    assert manifest["model"]["representation"] == "C5P-dcae_l10"
    assert manifest["model"]["codec_trainable"] is False
    assert manifest["model"]["physics_derived_loss_allowed"] is False
    assert manifest["training"]["epochs"] == 200
    assert manifest["training"]["total_optimizer_steps"] == 5400
    assert manifest["training"]["wandb_online_required"] is True
    assert [
        (task["run_index"], task["arm"], task["context_frames"], task["seed"])
        for task in manifest["tasks"]
    ] == [
        (0, "C5P-H1", 1, 1701),
        (1, "C5P-H1", 1, 1702),
        (2, "C5P-H1", 1, 1703),
        (3, "C5P-H2", 2, 1701),
        (4, "C5P-H2", 2, 1702),
        (5, "C5P-H2", 2, 1703),
    ]
    assert manifest["execution"]["gpu_wave_0"] == [0, 3, 1, 4]
    assert manifest["execution"]["gpu_wave_1"] == [2, 5]
    revision = manifest["execution"]["execution_only_revision"]
    assert revision["scientific_protocol_changed"] is False
    assert revision["superseded_job_id"] == "6894979"
    assert revision["superseded_job_runtime"] == "00:00:00"


def test_full_launcher_matches_frozen_task_order_and_budget() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert (
        'arms=("C5P-H1" "C5P-H1" "C5P-H1" "C5P-H2" "C5P-H2" "C5P-H2")'
        in text
    )
    assert "seeds=(1701 1702 1703 1701 1702 1703)" in text
    assert "contexts=(1 1 1 2 2 2)" in text
    assert "wait_wave 0 3 1 4" in text
    assert "wait_wave 2 5" in text
    assert text.count("--mode full") == 2
    assert 'result["completed_epochs"] != 200' in text
    assert 'result["completed_optimizer_steps"] != 5400' in text
    assert 'config["train_targets"] != [2, 432]' in text
    assert 'config["validation_targets"] != [498, 624]' in text
    assert 'latent["fit_frames"] != [0, 432]' in text
    assert "E6B" not in text
    assert "diffusion" not in text.lower()


def test_full_launcher_requires_exact_smoke_code_codecs_and_data() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for path in LOCKED:
        assert hashlib.sha256(path.read_bytes()).hexdigest() in text
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    for task in manifest["tasks"]:
        assert task["codec_checkpoint"] in text
        assert task["codec_sha256"] in text
    smoke = json.loads(SMOKE.read_text(encoding="utf-8"))
    assert smoke["status"] == "passed"
    assert smoke["scientific_result"] is False
    assert smoke["full_O2_training_allowed"] is True


def test_full_launcher_requires_tests_verified_85604_and_online_wandb() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    for required in (
        '"${PYTHON}" -m pytest -p no:cacheprovider -q',
        "MODEL_DATA_STAGED",
        "verified_shards=$((verified_shards + 1))",
        '"${verified_shards}" -ne 8',
        "WANDB_MODE=online",
        "WANDB_REQUIRE_SERVICE=true",
        'WANDB_GROUP="o2-c5p-l10-full"',
        'tracking["remote_state_after_finish"] != "finished"',
        'tracking["epochs_logged"] != 200',
    ):
        assert required in text
    assert "85606/data" not in text


def test_training_job_cannot_claim_o2_acceptance_or_open_o3() -> None:
    text = LAUNCHER.read_text(encoding="utf-8")
    assert 'result["held_out_85606_read"] is not False' in text
    assert 'result["physics_derived_loss_used"] is not False' in text
    assert 'result["target_truth_used_as_model_input"] is not False' in text
    assert '"training_summary_is_scientific_acceptance": False' in text
    assert '"O2_scientific_gate_evaluated": False' in text
    assert '"O3_launch_allowed": False' in text
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["post_training"]["training_summary_is_scientific_acceptance"] is False
    assert manifest["post_training"]["O2_gate_evaluated_by_training_job"] is False
    assert manifest["post_training"]["O3_launch_allowed"] is False
