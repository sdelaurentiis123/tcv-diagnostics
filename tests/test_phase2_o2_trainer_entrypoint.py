import importlib.util
import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]
ENTRYPOINT = ROOT / "paper0/tools/train_o2.py"
MANIFEST = ROOT / "paper0/manifests/phase2_c5p_o2_continuation_85604.json"


def load_entrypoint():
    specification = importlib.util.spec_from_file_location("paper0_train_o2", ENTRYPOINT)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    return module


def test_cli_exposes_only_frozen_arms_modes_and_seeds():
    completed = subprocess.run(
        [sys.executable, str(ENTRYPOINT), "--help"],
        check=True,
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(ROOT / "src")},
    )
    assert "{smoke,full}" in completed.stdout
    assert "{C5P-H1,C5P-H2}" in completed.stdout
    assert "{1701,1702,1703}" in completed.stdout
    assert "E6B-H1" not in completed.stdout


def test_manifest_authorization_accepts_only_the_exact_seed_checkpoint():
    module = load_entrypoint()
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    selected = manifest["codec"]["selected_checkpoints"][1]
    result = module.authorize_from_manifest(
        manifest,
        arm="C5P-H2",
        seed=1702,
        codec_checkpoint=Path(selected["path"]),
        codec_sha256=selected["sha256"],
    )
    assert result["authorized"] is True
    assert result["held_out_85606_read"] is False
    assert result["codec"] == "C5P-dcae_l10"

    with pytest.raises(RuntimeError, match="hash differs"):
        module.authorize_from_manifest(
            manifest,
            arm="C5P-H2",
            seed=1702,
            codec_checkpoint=Path(selected["path"]),
            codec_sha256="0" * 64,
        )
    with pytest.raises(RuntimeError, match="not authorized"):
        module.authorize_from_manifest(
            manifest,
            arm="E6B-H1",
            seed=1702,
            codec_checkpoint=Path(selected["path"]),
            codec_sha256=selected["sha256"],
        )


def test_entrypoint_is_fail_closed_on_clean_checkout_cuda_and_online_wandb():
    text = ENTRYPOINT.read_text(encoding="utf-8")
    assert "verify_checkout(args.paper0_commit)" in text
    assert "if not torch.cuda.is_available()" in text
    assert "OnlineWandbTracker.start" in text
    assert '"mode": "online_required"' in text
    assert "held-out manifests are prohibited" in text
    assert "train_o2(" in text
