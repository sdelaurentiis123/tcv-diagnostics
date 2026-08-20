"""Frozen-manifest and launcher tests for the four-arm ECRD smoke."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess

from tcv_diagnostics.codec_training import sha256_path
from tcv_diagnostics.ecrd_training import ECRD_ARMS, frozen_parameter_counts
from tcv_diagnostics.models.ecrd import MultiscaleNoiseConfig


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/ecrd_engineering_smoke_85604.json"
LAUNCHER = ROOT / "cluster/ecrd_engineering_smoke.sbatch"
FINALIZER = ROOT / "cluster/ecrd_engineering_smoke_finalize.sbatch"
ENTRYPOINT = ROOT / "paper0/tools/train_ecrd.py"
SPEC = importlib.util.spec_from_file_location("train_ecrd_for_smoke", ENTRYPOINT)
assert SPEC is not None and SPEC.loader is not None
TRAIN = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(TRAIN)
EXPECTED_MANIFEST_SHA256 = (
    "6204e7bfef02c449f464ea7647b8d6fc14c0bba22135c0b56ea47d77f530ac45"
)


def _manifest() -> dict[str, object]:
    return json.loads(MANIFEST.read_text(encoding="utf-8"))


def test_smoke_manifest_bytes_and_exact_budget_are_frozen() -> None:
    manifest = _manifest()
    assert sha256_path(MANIFEST) == EXPECTED_MANIFEST_SHA256
    assert manifest["status"] == "frozen_before_ECRD_engineering_smoke"
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert tuple(manifest["authorized_arms"]) == ECRD_ARMS
    assert manifest["authorized_seeds"] == [1701]
    assert manifest["full_training_authorized"] is False
    assert manifest["smoke"] == {
        "training_targets": [2, 6],
        "validation_targets": [498, 502],
        "epochs": 1,
        "optimizer_steps": 2,
        "target_presentations": 4,
        "ensemble_members": 2,
        "EDM_steps": 18,
        "network_evaluations_per_member": 35,
        "full_volume_sampling": True,
        "checkpoint_reload_bitwise_exact_required": True,
        "finished_online_wandb_required": True,
        "scientific_result": False,
    }
    exact = manifest["exact_implementation"]
    assert exact["parameter_counts"] == frozen_parameter_counts()
    assert exact["multiscale_noise"] == MultiscaleNoiseConfig().to_record()


def test_real_smoke_manifest_authorizes_only_seed1701_smoke() -> None:
    manifest = _manifest()
    locks = manifest["evidence_locks"]
    input_hashes = {
        name: locks[name]["sha256"]
        for name in (
            "H1_training_parent",
            "H1_validation_parent",
            "sym_H1_training_parent",
            "sym_H1_validation_parent",
        )
    }
    for arm in ECRD_ARMS:
        authority = TRAIN.authorize_manifest(
            manifest,
            manifest_path=MANIFEST,
            manifest_sha256=EXPECTED_MANIFEST_SHA256,
            mode="smoke",
            arm=arm,
            seed=1701,
            input_hashes=input_hashes,
        )
        assert authority["scope"] == f"ECRD_smoke_{arm}_seed1701_85604"
        assert authority["held_out_85606_read"] is False


def test_cpu_parent_is_hash_locked_and_cannot_authorize_full_training() -> None:
    manifest = _manifest()
    parent_use = manifest["symmetrized_parent_use"]
    assert parent_use == {
        "artifact_authority": "bounded_non_scientific_engineering_smoke_only",
        "execution_device": "cpu-smoke",
        "authorized_modes": ["smoke"],
        "H100_comparison_required_before_full_training": True,
    }
    generation = manifest["evidence_locks"]["sym_H1_parent_generation"]
    assert generation["slurm_job_id"] == "6912481"
    assert generation["result_sha256"] == (
        "ba8710e1e0813652fbbddc5c0ce7de20ddcdc39a3d74cf478ed344d8f3bb4037"
    )
    assert generation["remote_state_after_finish"] == "finished"
    assert generation["full_training_authorized"] is False


def test_smoke_array_launcher_is_fail_closed_and_h100_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    assert "__ECRD" not in source and "__SYM" not in source
    for required in (
        "#SBATCH --partition=gpupreempt",
        "#SBATCH --qos=gpupreempt",
        "#SBATCH --gres=gpu:1",
        "#SBATCH --constraint=h100",
        "#SBATCH --array=0-3%1",
        "PAPER0_EXPECTED_COMMIT",
        "WANDB_MODE=online",
        "--mode smoke",
        "--seed 1701",
        "--manifest-sha256",
        "-m pytest -p no:cacheprovider -q",
        EXPECTED_MANIFEST_SHA256,
        "d238d055c3f1da9e3096a81cac67176f90365c99dfb423a1a0629f85b61f9532",
        "9acaf190507d2cc0216a4f137e7e63717370e52484e99351d6ded903e30cc2d3",
        '"scientific_result": False',
        '"full_training_authorized": False',
    ):
        assert required in source
    assert "--mode full" not in source
    assert "from tcv_diagnostics.transport" not in source
    assert "from tcv_diagnostics.assimilat" not in source


def test_finalizer_requires_all_four_runs_and_is_non_scientific() -> None:
    source = FINALIZER.read_text(encoding="utf-8")
    subprocess.run(["bash", "-n", str(FINALIZER)], check=True)
    assert "__ECRD" not in source
    assert "ECRD_SMOKE_ARRAY_JOB_ID" in source
    assert "--dependency=afterok:<array-job-id>" in source
    assert EXPECTED_MANIFEST_SHA256 in source
    assert "e00521421bdeaab56bdac7899257f9e0521d736a4f191f4c19d053f2a4077ccd" in source
    run_arguments = [
        'B5=${JOB_ROOT}/task_0_b5_seed_1701/model',
        'B5-Context=${JOB_ROOT}/task_1_b5_context_seed_1701/model',
        'ECRD=${JOB_ROOT}/task_2_ecrd_seed_1701/model',
        'ECRD-History=${JOB_ROOT}/task_3_ecrd_history_seed_1701/model',
    ]
    assert all(value in source for value in run_arguments)
    assert '"scientific_result": False' in source
    assert '"full_training_authorized": False' in source
    assert "from tcv_diagnostics.transport" not in source
    assert "from tcv_diagnostics.assimilat" not in source
