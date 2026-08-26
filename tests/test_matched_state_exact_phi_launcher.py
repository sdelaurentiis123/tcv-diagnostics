"""Static safety checks for causal exact-phi evaluation on Rusty."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = ROOT / "cluster/post_ecrd_old_85604_matched_state_exact_phi.sbatch"


def test_launcher_is_cpu_only_and_right_sized() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "#SBATCH --partition=gen" in source
    assert "#SBATCH --ntasks=4" in source
    assert "#SBATCH --cpus-per-task=2" in source
    assert "#SBATCH --mem=96G" in source
    assert "#SBATCH --time=04:00:00" in source
    assert "--gres=gpu" not in source
    assert "--array=" not in source


def test_launcher_hash_locks_validated_elliptic_stack() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    required = (
        "7c13a60c26238acbe0db0ed6eb0e3c18f8b18f570bb7f22913642bae9bce22e5",
        "d3d1addaa421ddc005cd4ecfedfe303370034c43de6137ef648e3b116ceb8ab4",
        "622841b559c3c4132444c0466915aecc35cbd6801242d94fafb0042a0acda5e4",
        "39a1404c3bdf2bf4b295c8531ef27771884e2d8df3d6ebc33153595bd7eb2bc2",
        "031ca578e7db5ef5a3cbd9d4688dadc244c0da542b66f82fd79e1c10d5712f27",
        "0eeffe4c550d71eacd4c2d09874280bf85f394c2e08558712e15dca0495e8bf8",
        "9e4ae1f46c01418711515cda63fd92513712705655c5623d932297e5d8c53333",
        "b5d86b8adedf11f3fab2783cbc75fd7b1064e87d91c7df596437fd7d26dd11e6",
    )
    assert all(digest in source for digest in required)


def test_launcher_prohibits_truth_phi_and_new_data() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "PAPER0_E6B_GENERATION_RESULT_SHA256" in source
    assert "status --porcelain --untracked-files=all" in source
    assert '"paper0:truth_layout=false"' in source
    assert "predicted_Bphi_no_truth_bypass" in source
    assert '.target_truth_phi_read == false' in source
    assert '.held_out_85606_read == false' in source
    assert '.new_nersc_data_read == false' in source
    assert "target_truth_phi_read\": False" in source
    assert "/85606/" not in source


def test_launcher_solves_exactly_the_frozen_candidate_set() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert 'if [[ "${#CANDIDATE_RECORDS[@]}" -ne 7 ]]' in source
    for name in (
        "h4_direct_predicted_e6b_native81.h5",
        "h4_autoregressive_lead1_predicted_e6b_native81.h5",
        "h4_autoregressive_lead2_predicted_e6b_native81.h5",
        "h8_direct_predicted_e6b_native81.h5",
        "h8_autoregressive_lead1_predicted_e6b_native81.h5",
        "h8_autoregressive_lead2_predicted_e6b_native81.h5",
        "h8_autoregressive_lead4_predicted_e6b_native81.h5",
    ):
        assert name in source
