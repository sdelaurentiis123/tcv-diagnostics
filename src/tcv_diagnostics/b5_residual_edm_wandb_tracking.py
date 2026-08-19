"""Strict online W&B mirror for the bounded B5 residual-EDM smoke."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

from .wandb_tracking import OnlineWandbTracker, WandbRunSpec


def b5_edm_wandb_step_metrics(
    record: Mapping[str, Any],
) -> dict[str, int | float]:
    """Map one authoritative optimization record to stable W&B names."""

    return {
        "optimizer/global_step": int(record["global_step"]),
        "data/target_frame": int(record["target_frame"]),
        "data/context_frame": int(record["context_frame"]),
        "noise/sigma": float(record["sigma"]),
        "train/EDM_loss": float(record["EDM_loss"]),
        "train/unweighted_MSE": float(record["unweighted_MSE"]),
        "optimizer/learning_rate": float(record["learning_rate"]),
        "optimizer/preclip_gradient_norm": float(
            record["preclip_gradient_norm"]
        ),
        "timing/step_wall_seconds": float(record["step_wall_seconds"]),
    }


def b5_edm_wandb_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Select bounded-smoke facts; no large artifact is uploaded."""

    sampler = result["sampler_probe"]
    return {
        "final/status_passed": result["status"] == "passed",
        "final/all_mechanical_gates_passed": bool(
            result["all_mechanical_gates_passed"]
        ),
        "final/completed_optimizer_steps": int(
            result["completed_optimizer_steps"]
        ),
        "final/initial_fixed_probe_EDM_loss": float(
            result["initial_fixed_probe"]["mean_EDM_loss"]
        ),
        "final/final_fixed_probe_EDM_loss": float(
            result["final_fixed_probe"]["mean_EDM_loss"]
        ),
        "final/fixed_probe_relative_change": float(
            result["fixed_probe_relative_change"]
        ),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
        ),
        "final/toroidal_equivariance_passed": bool(
            result["toroidal_equivariance"]["passed"]
        ),
        "ensemble/normalized_residual_member_RMS_difference": float(
            sampler["normalized_residual_member_RMS_difference"]
        ),
        "ensemble/standardized_field_member_RMS_difference": float(
            sampler["standardized_field_member_RMS_difference"]
        ),
        "ensemble/nonzero_member_diversity": bool(
            sampler["nonzero_member_diversity"]
        ),
        "compute/network_evaluations_per_member": int(
            sampler["network_evaluations_per_member"]
        ),
        "compute/parameter_count": int(result["parameter_count"]),
        "compute/peak_cuda_bytes": int(result["peak_cuda_bytes"]),
        "compute/peak_cuda_GiB": float(result["peak_cuda_GiB"]),
        "compute/wall_seconds": float(result["wall_seconds"]),
        "provenance/paper0_commit": str(result["paper0_commit"]),
        "provenance/checkpoint_sha256": str(
            result["artifacts"]["smoke_checkpoint"]["sha256"]
        ),
        "provenance/history_sha256": str(
            result["artifacts"]["history"]["sha256"]
        ),
        "scope/scientific_result": bool(result["scientific_result"]),
        "scope/full_B5_training_authorized": bool(
            result["full_B5_training_authorized"]
        ),
        "scope/validation_frames_read": bool(result["validation_frames_read"]),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
    }


class B5EDMOnlineWandbTracker(OnlineWandbTracker):
    """Required online B5 smoke run with remote completion verification."""

    def __init__(
        self,
        *,
        module: Any,
        run: Any,
        spec: WandbRunSpec,
        authenticated_username: str,
        tracking_directory: Path,
    ) -> None:
        super().__init__(
            module=module,
            run=run,
            spec=spec,
            authenticated_username=authenticated_username,
            tracking_directory=tracking_directory,
        )
        self.steps_logged = 0

    def log_step(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            b5_edm_wandb_step_metrics(record),
            step=int(record["global_step"]),
            commit=True,
        )
        self.steps_logged += 1

    def finish_success(self, result: Mapping[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        expected = int(result["completed_optimizer_steps"])
        if self.steps_logged != expected or expected != 64:
            raise RuntimeError(
                f"B5 W&B logged {self.steps_logged} steps, expected 64"
            )
        if result.get("all_mechanical_gates_passed") is not True:
            raise RuntimeError("cannot finish B5 W&B success for a failed smoke")
        self._run.summary.update(b5_edm_wandb_result_summary(result))
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True

        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = self._module.Api(timeout=30).run(remote_path)
        if str(remote.id) != self.spec.run_id:
            raise RuntimeError("remote B5 W&B run identity differs after finish")
        remote_state = str(remote.state)
        if remote_state != "finished":
            raise RuntimeError(
                f"remote B5 W&B state is {remote_state!r}, not 'finished'"
            )
        return {
            "schema_version": 1,
            "required": True,
            "mode": "online",
            "spec": self.spec.to_record(),
            "authenticated_username": self.authenticated_username,
            "wandb_version": str(self._module.__version__),
            "run_url": run_url,
            "remote_path": remote_path,
            "remote_presence_verified_after_finish": True,
            "remote_state_after_finish": remote_state,
            "steps_logged": self.steps_logged,
            "checkpoints_uploaded": False,
            "samples_uploaded": False,
            "local_artifacts_are_scientific_authority": True,
        }
