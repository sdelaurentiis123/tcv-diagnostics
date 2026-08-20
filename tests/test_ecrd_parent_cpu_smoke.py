"""Fail-closed gates for the provisional CPU parent used by ECRD smoke."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper0/manifests/ecrd_sym_h1_parent_cpu_smoke_85604.json"
H100_MANIFEST = ROOT / "paper0/manifests/ecrd_sym_h1_parent_85604.json"
AMENDMENT = ROOT / "paper0/protocol/ECRD_PARENT_CPU_SMOKE_AMENDMENT_2026-08-20.md"
LAUNCHER = ROOT / "cluster/ecrd_sym_h1_parent_cpu_smoke.sbatch"
ENTRYPOINT = ROOT / "paper0/tools/build_ecrd_sym_h1_parent.py"
SPEC = importlib.util.spec_from_file_location("build_ecrd_parent", ENTRYPOINT)
assert SPEC is not None and SPEC.loader is not None
BUILDER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BUILDER)


def test_cpu_smoke_manifest_and_amendment_are_hash_locked() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert hashlib.sha256(MANIFEST.read_bytes()).hexdigest() == (
        BUILDER.EXPECTED_CPU_SMOKE_MANIFEST_SHA256
    )
    amendment_sha256 = hashlib.sha256(AMENDMENT.read_bytes()).hexdigest()
    assert amendment_sha256 == manifest["execution_amendment"]["sha256"]
    assert amendment_sha256 == BUILDER.EXPECTED_CPU_SMOKE_AMENDMENT_SHA256
    assert manifest["development_run"] == "85604"
    assert manifest["held_out_85606_access_allowed"] is False
    assert manifest["execution"]["artifact_authority"] == (
        "bounded_non_scientific_engineering_smoke_only"
    )
    assert manifest["execution"]["full_training_authorized"] is False
    assert (
        manifest["execution"]["H100_parent_comparison_required_before_full_training"]
        is True
    )


def test_cpu_manifest_authorizes_only_bounded_smoke_parent() -> None:
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    authorization = BUILDER.authorize_manifest(
        manifest,
        manifest_path=MANIFEST,
        artifact_root=Path(manifest["evidence_locks"]["model_dataset"]["root"]),
        h1_checkpoint=Path(manifest["evidence_locks"]["H1_checkpoint"]["path"]),
        h1_sha256=manifest["evidence_locks"]["H1_checkpoint"]["sha256"],
        codec_checkpoint=Path(manifest["evidence_locks"]["codec_checkpoint"]["path"]),
        codec_sha256=manifest["evidence_locks"]["codec_checkpoint"]["sha256"],
        h1_training_commit=manifest["evidence_locks"]["H1_checkpoint"][
            "training_commit"
        ],
        execution_device="cpu-smoke",
    )
    assert authorization["execution_device"] == "cpu-smoke"
    assert authorization["artifact_authority"] == (
        "bounded_non_scientific_engineering_smoke_only"
    )
    assert authorization["held_out_85606_read"] is False


def test_original_h100_parent_authorization_is_preserved() -> None:
    manifest = json.loads(H100_MANIFEST.read_text(encoding="utf-8"))
    authorization = BUILDER.authorize_manifest(
        manifest,
        manifest_path=H100_MANIFEST,
        artifact_root=Path(manifest["evidence_locks"]["model_dataset"]["root"]),
        h1_checkpoint=Path(manifest["evidence_locks"]["H1_checkpoint"]["path"]),
        h1_sha256=manifest["evidence_locks"]["H1_checkpoint"]["sha256"],
        codec_checkpoint=Path(manifest["evidence_locks"]["codec_checkpoint"]["path"]),
        codec_sha256=manifest["evidence_locks"]["codec_checkpoint"]["sha256"],
        h1_training_commit=manifest["evidence_locks"]["H1_checkpoint"][
            "training_commit"
        ],
        execution_device="h100",
    )
    assert authorization["execution_device"] == "h100"
    assert authorization["artifact_authority"] == "scientific_H100_parent"


def test_launcher_is_cpu_only_clean_commit_locked_and_online_tracked() -> None:
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --qos=gen" in source
    assert "#SBATCH --gres" not in source
    assert "#SBATCH --time=02:00:00" in source
    assert 'export CUDA_VISIBLE_DEVICES=""' in source
    assert "PAPER0_EXPECTED_COMMIT" in source
    assert "status --porcelain --untracked-files=all" in source
    assert "WANDB_MODE=online" in source
    assert "WANDB_REQUIRE_SERVICE=true" in source
    assert "--execution-device cpu-smoke" in source


def test_launcher_validates_non_scientific_authority_and_data_scope() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    for statement in (
        '"artifact_authority": "bounded_non_scientific_engineering_smoke_only"',
        '"full_training_authorized": False',
        '"target_truth_read": False',
        '"guard_frames_read": False',
        '"held_out_85606_read": False',
        '"training_performed": False',
        '"physics_metric_evaluated": False',
        '"assimilation_performed": False',
    ):
        assert statement in source
    assert '["standardized_parent_mean"].shape != (count, 5, 64, 32, 88)' in source
    assert "-m pytest -p no:cacheprovider -q" in source


def test_parent_generator_device_branch_cannot_silently_make_cpu_authority() -> None:
    source = (ROOT / "src/tcv_diagnostics/ecrd_data.py").read_text(encoding="utf-8")
    assert 'execution_device != "cpu-smoke"' in source
    assert '!= "bounded_non_scientific_engineering_smoke_only"' in source
    assert 'device.type not in ("cpu", "cuda")' in source
    assert 'if device.type == "cuda"' in source
