"""Static locks for the bounded Rocky 9 B3 FGN smoke launcher."""

from __future__ import annotations

import hashlib
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/phase3_b3_fgn_gpu_smoke.sbatch"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_b3_launcher_is_one_gpu_bounded_smoke_only() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gpupreempt" in source
    assert "#SBATCH --qos=gpupreempt" in source
    assert "#SBATCH --gres=gpu:1" in source
    assert "#SBATCH --constraint=h100|h200" in source
    assert "#SBATCH --time=02:00:00" in source
    assert "#SBATCH --no-requeue" in source
    assert "--mode smoke" in source
    assert "--seed 1701" in source
    assert "train_b3_fgn.py" in source
    assert "train_fgn_full" not in source
    assert "--array" not in source
    assert "rm -" not in source
    assert "git reset" not in source
    assert "git checkout" not in source
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def test_b3_launcher_requires_online_wandb_and_full_preflight_suite() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "export WANDB_MODE=online" in source
    assert "export WANDB_REQUIRE_SERVICE=true" in source
    assert '"${PYTHON}" -m pytest -p no:cacheprovider -q' in source
    assert 'tracking["remote_state_after_finish"] != "finished"' in source
    assert 'tracking["epochs_logged"] != 2' in source


def test_b3_launcher_checks_parent_identity_diversity_codec_and_scope() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    required = (
        'result["preoptimization_parent_identity"]["bitwise_exact"]',
        'result["deterministic_parent_load_audit"]["passed"]',
        'result["checkpoint_reload_bitwise_exact"]',
        'result["codec_bitwise_unchanged"]',
        'result["common_parameter_gradient_seen"]',
        'result["new_parameter_gradient_seen"]',
        'probe["nonzero_latent_diversity"]',
        'probe["nonzero_field_diversity"]',
        '[1, 2, 1, 5, 64, 32, 88]',
        '"held_out_85606_read"',
        '"scientific_result"',
        '"full_B3_training_authorized"',
        '"assimilation_allowed"',
        '"diagnostic_ranking_allowed"',
    )
    for text in required:
        assert text in source


def test_b3_launcher_embedded_local_source_hashes_match() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    mapping = {
        "src/tcv_diagnostics/models/layers.py": "models/layers.py",
        "src/tcv_diagnostics/models/dcae.py": "models/dcae.py",
        "src/tcv_diagnostics/models/o2.py": "models/o2.py",
        "src/tcv_diagnostics/models/vit.py": "models/vit.py",
        "src/tcv_diagnostics/models/functional_noise.py": "models/functional_noise.py",
        "src/tcv_diagnostics/models/FGN_LICENSE.txt": "models/FGN_LICENSE.txt",
        "src/tcv_diagnostics/model_training_data.py": "model_training_data.py",
        "src/tcv_diagnostics/o2_training_data.py": "o2_training_data.py",
        "src/tcv_diagnostics/o2_training.py": "o2_training.py",
        "src/tcv_diagnostics/codec_training.py": "codec_training.py",
        "src/tcv_diagnostics/fgn_training.py": "fgn_training.py",
        "src/tcv_diagnostics/wandb_tracking.py": "wandb_tracking.py",
        "src/tcv_diagnostics/fgn_wandb_tracking.py": "fgn_wandb_tracking.py",
    }
    for relative, suffix in mapping.items():
        pattern = re.compile(
            rf'check_sha256 "([0-9a-f]{{64}})" "\$\{{PAPER0_ROOT\}}/src/tcv_diagnostics/{re.escape(suffix)}"'
        )
        match = pattern.search(source)
        assert match is not None, relative
        assert match.group(1) == _sha256(ROOT / relative), relative

    entry_match = re.search(
        r'check_sha256 "([0-9a-f]{64})" "\$\{ENTRYPOINT\}"', source
    )
    assert entry_match is not None
    assert entry_match.group(1) == _sha256(ROOT / "paper0/tools/train_b3_fgn.py")


def test_b3_launcher_locks_protocol_manifest_and_external_parent_hashes() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert _sha256(ROOT / "paper0/protocol/PHASE3_B3_FGN_PROTOCOL.md") in source
    assert _sha256(ROOT / "paper0/manifests/phase3_b3_fgn_85604.json") in source
    assert (
        'readonly PARENT_SHA="5562095a9316077fb16299b08fa9527b33c768efdde48a066e3e71f94150671e"'
        in source
    )
    assert (
        'readonly C5P_CODEC_SHA="9fc7fbd684d660bd9f33e9db32500aa2795fc354c90886d2e893cd470ea6bc9d"'
        in source
    )
    assert (
        'readonly LATENT_NORMALIZATION_SHA="afcb0eda5d611d58f6eb2340aa55cfecd1a231b83a6912d9db398be706296738"'
        in source
    )
