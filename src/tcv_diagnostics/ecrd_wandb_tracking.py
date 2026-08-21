"""Strict online W&B mirrors for ECRD parent generation and training.

Ceph artifacts, hashes, and JSON records remain the scientific authority.
W&B is required only as a live and remotely verified monitoring mirror.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Mapping, Sequence

from .wandb_tracking import OnlineWandbTracker


def verify_ecrd_remote_finished_run(
    *,
    module: Any,
    remote_path: str,
    expected_run_id: str,
    retry_delays_seconds: Sequence[float] = (2.0, 4.0, 8.0, 16.0, 30.0),
    sleep: Callable[[float], None] = time.sleep,
) -> Any:
    """Boundedly wait for W&B's read API to reflect ECRD completion."""

    delays = tuple(float(delay) for delay in retry_delays_seconds)
    if any(delay < 0 for delay in delays):
        raise ValueError("ECRD W&B verification delays must be nonnegative")
    last_state: str | None = None
    last_error: Exception | None = None
    for attempt in range(len(delays) + 1):
        if attempt:
            sleep(delays[attempt - 1])
        try:
            remote = module.Api(timeout=30).run(remote_path)
        except Exception as error:  # The read API can lag the completed upload.
            last_error = error
            continue
        remote_id = str(remote.id)
        if remote_id != expected_run_id:
            raise RuntimeError(
                f"remote ECRD W&B identity {remote_id!r} differs from "
                f"required {expected_run_id!r}"
            )
        last_state = str(remote.state)
        last_error = None
        if last_state == "finished":
            return remote
    if last_error is not None:
        raise RuntimeError(
            "remote ECRD W&B completion verification exhausted bounded retries; "
            f"last API error: {last_error!r}"
        ) from last_error
    raise RuntimeError(
        "remote ECRD W&B completion verification exhausted bounded retries; "
        f"last state was {last_state!r}, not 'finished'"
    )


def ecrd_epoch_metrics(record: Mapping[str, Any]) -> dict[str, int | float]:
    """Map one authoritative ECRD epoch while retaining block validation."""

    performed = bool(record["validation_candidate"])
    metrics: dict[str, int | float] = {
        "epoch/completed": int(record["completed_epoch"]),
        "optimizer/global_step": int(record["global_optimizer_step"]),
        "optimizer/first_learning_rate": float(record["first_learning_rate"]),
        "optimizer/last_learning_rate": float(record["last_learning_rate"]),
        "optimizer/mean_preclip_gradient_norm": float(
            record["mean_preclip_gradient_norm"]
        ),
        "optimizer/maximum_preclip_gradient_norm": float(
            record["maximum_preclip_gradient_norm"]
        ),
        "train/target_count": int(record["train_target_count"]),
        "train/mean_objective": float(record["train_mean_objective"]),
        "train/mean_EDM_loss": float(record["train_mean_edm_loss"]),
        "train/mean_unweighted_EDM_MSE": float(
            record["train_mean_unweighted_edm_mse"]
        ),
        "train/mean_mean_head_MSE": float(record["train_mean_mean_mse"]),
        "validation/performed": int(performed),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }
    validation = record.get("validation")
    if not performed:
        if validation is not None:
            raise ValueError("non-candidate ECRD epoch contains validation")
        return metrics
    if not isinstance(validation, Mapping):
        raise ValueError("ECRD candidate epoch lacks validation")
    metrics.update(
        {
            "validation/checkpoint_score": float(validation["checkpoint_score"]),
            "validation/aggregate_objective": float(
                validation["aggregate"]["objective"]
            ),
            "validation/aggregate_EDM_loss": float(
                validation["aggregate"]["edm_loss"]
            ),
            "validation/aggregate_mean_head_MSE": float(
                validation["aggregate"]["mean_mse"]
            ),
            "timing/validation_wall_seconds": float(validation["wall_seconds"]),
        }
    )
    for block, values in validation["blocks"].items():
        metrics[f"validation/{block}/objective"] = float(values["objective"])
        metrics[f"validation/{block}/EDM_loss"] = float(values["edm_loss"])
        metrics[f"validation/{block}/mean_head_MSE"] = float(values["mean_mse"])
    return metrics


def ecrd_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Compact final training facts; no checkpoint tensor is uploaded."""

    selected = result["selected_validation"]
    artifacts = result["artifacts"]
    return {
        "final/training_completed": result["status"]
        == "training_completed_checkpoint_selected",
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(result["completed_optimizer_steps"]),
        "final/candidate_count": int(result["candidate_count"]),
        "final/selected_completed_epoch": int(result["selected_completed_epoch"]),
        "final/selected_validation_objective": float(
            selected["checkpoint_score"]
        ),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
        ),
        "compute/parameter_count": int(result["parameter_count"]),
        "compute/peak_cuda_GiB": float(result["peak_cuda_memory_GiB"]),
        "compute/wall_seconds": float(result["wall_seconds"]),
        "provenance/paper0_commit": str(result["paper0_commit"]),
        "provenance/selected_checkpoint_sha256": str(
            artifacts["selected_checkpoint"]["sha256"]
        ),
        "provenance/history_sha256": str(artifacts["history"]["sha256"]),
        "scope/physics_derived_loss_used": bool(
            result["physics_derived_loss_used"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/scientific_forecast_generated": bool(
            result["scientific_forecast_generated"]
        ),
    }


class ECRDOnlineWandbTracker(OnlineWandbTracker):
    """Required online ECRD training mirror with remote completion check."""

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            ecrd_epoch_metrics(record),
            step=int(record["global_optimizer_step"]),
            commit=True,
        )
        self.epochs_logged += 1

    def log_smoke_probe(self, probe: Mapping[str, Any]) -> None:
        """Mirror bounded mechanical checks without treating them as science."""

        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            {
                "smoke/finite": int(bool(probe["finite"])),
                "smoke/member_diversity": float(probe["member_diversity"]),
                "smoke/peak_cuda_GiB": float(probe["peak_cuda_GiB"]),
                "smoke/max_generator_equivariance_error": float(
                    probe["max_generator_equivariance_error"]
                ),
                "smoke/max_mean_head_equivariance_error": float(
                    probe["max_mean_head_equivariance_error"]
                ),
                "smoke/all_mechanical_gates_passed": int(
                    bool(probe["all_mechanical_gates_passed"])
                ),
            },
            step=int(probe["optimizer_steps"]),
            commit=True,
        )

    def finish_success(self, result: Mapping[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        expected_epochs = int(result["completed_epochs"])
        if self.epochs_logged != expected_epochs:
            raise RuntimeError("ECRD W&B epoch count differs")
        if result.get("checkpoint_reload_bitwise_exact") is not True:
            raise RuntimeError("ECRD selected checkpoint did not reload exactly")
        self._run.summary.update(ecrd_result_summary(result))
        return self._finish_and_verify(extra={"epochs_logged": self.epochs_logged})

    def _finish_and_verify(self, *, extra: Mapping[str, Any]) -> dict[str, Any]:
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True
        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = verify_ecrd_remote_finished_run(
            module=self._module,
            remote_path=remote_path,
            expected_run_id=self.spec.run_id,
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
            "remote_state_after_finish": str(remote.state),
            "checkpoints_uploaded": False,
            "samples_uploaded": False,
            "local_artifacts_are_scientific_authority": True,
            **dict(extra),
        }


class ECRDParentOnlineWandbTracker(ECRDOnlineWandbTracker):
    """Online mirror for truth-free symmetrized-H1 parent generation."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.splits_logged = 0

    def log_split(self, result: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        split = str(result["split"])
        if split not in ("train", "validation"):
            raise ValueError("ECRD parent split differs")
        self._run.log(
            {
                f"parent/{split}/target_count": int(result["target_count"]),
                f"parent/{split}/wall_seconds": float(result["wall_seconds"]),
                f"parent/{split}/peak_cuda_GiB": float(
                    int(result["peak_cuda_memory_bytes"]) / 1024**3
                ),
            },
            step=self.splits_logged,
            commit=True,
        )
        self.splits_logged += 1

    def finish_success(self, result: Mapping[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        if self.splits_logged != 2:
            raise RuntimeError("ECRD parent W&B requires both splits")
        if result.get("held_out_85606_read") is not False:
            raise RuntimeError("ECRD parent result crossed the held-out boundary")
        artifact_authority = str(
            result.get("artifact_authority", "scientific_H100_parent")
        )
        is_scientific_authority = artifact_authority == "scientific_H100_parent"
        self._run.summary.update(
            {
                "final/parent_generation_completed": True,
                "final/train_target_count": int(
                    result["splits"]["train"]["target_count"]
                ),
                "final/validation_target_count": int(
                    result["splits"]["validation"]["target_count"]
                ),
                "provenance/paper0_commit": str(result["paper0_commit"]),
                "scope/target_truth_read": bool(result["target_truth_read"]),
                "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
                "scope/full_training_authorized": bool(
                    result.get("full_training_authorized", True)
                ),
                "provenance/artifact_authority": artifact_authority,
            }
        )
        return self._finish_and_verify(
            extra={
                "splits_logged": self.splits_logged,
                "artifact_authority": artifact_authority,
                "local_artifacts_are_scientific_authority": is_scientific_authority,
            }
        )
