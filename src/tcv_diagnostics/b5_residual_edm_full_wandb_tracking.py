"""Strict online W&B mirror for the full seed-1701 B5 training run."""

from __future__ import annotations

from typing import Any, Mapping

from .wandb_tracking import OnlineWandbTracker


def b5_full_epoch_metrics(
    record: Mapping[str, Any],
) -> dict[str, int | float]:
    """Map one authoritative B5 epoch, preserving sparse validation."""

    performed = bool(record["validation_candidate"])
    metrics: dict[str, int | float] = {
        "epoch/completed": int(record["completed_epoch"]),
        "optimizer/global_step": int(record["global_optimizer_step"]),
        "optimizer/ema_updates": int(record["EMA_updates"]),
        "optimizer/first_learning_rate": float(record["first_learning_rate"]),
        "optimizer/last_learning_rate": float(record["last_learning_rate"]),
        "optimizer/mean_preclip_gradient_norm": float(
            record["mean_preclip_gradient_norm"]
        ),
        "optimizer/maximum_preclip_gradient_norm": float(
            record["maximum_preclip_gradient_norm"]
        ),
        "train/target_count": int(record["train_target_count"]),
        "train/mean_EDM_loss": float(record["train_mean_EDM_loss"]),
        "train/mean_unweighted_MSE": float(record["train_mean_unweighted_MSE"]),
        "train/minimum_sigma": float(record["train_minimum_sigma"]),
        "train/maximum_sigma": float(record["train_maximum_sigma"]),
        "validation/performed": int(performed),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }
    validation = record.get("validation")
    if performed:
        if not isinstance(validation, Mapping):
            raise ValueError("B5 full validation record is missing")
        metrics.update(
            {
                "validation/mean_EDM_loss": float(validation["mean_EDM_loss"]),
                "validation/mean_unweighted_MSE": float(
                    validation["mean_unweighted_MSE"]
                ),
                "validation/probe_count": int(validation["probe_count"]),
                "validation/minimum_sigma": float(validation["minimum_sigma"]),
                "validation/maximum_sigma": float(validation["maximum_sigma"]),
                "timing/validation_wall_seconds": float(validation["wall_seconds"]),
            }
        )
    elif validation is not None:
        raise ValueError("B5 full non-candidate epoch contains validation")
    return metrics


def b5_full_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Select compact training facts; immutable local artifacts are authority."""

    selected = result["selected_validation"]
    final = result["final_candidate_validation"]
    artifacts = result["artifacts"]
    return {
        "final/training_completed": result["status"]
        == "training_completed_checkpoint_selected",
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(result["completed_optimizer_steps"]),
        "final/ema_updates": int(result["EMA_updates"]),
        "final/candidate_count": int(result["candidate_count"]),
        "final/selected_completed_epoch": int(result["selected_completed_epoch"]),
        "final/selected_optimizer_step": int(result["selected_optimizer_step"]),
        "final/selected_validation_EDM_loss": float(selected["mean_EDM_loss"]),
        "final/final_candidate_validation_EDM_loss": float(final["mean_EDM_loss"]),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
        ),
        "final/all_losses_and_gradients_finite": bool(
            result["all_losses_and_gradients_finite"]
        ),
        "compute/parameter_count": int(result["parameter_count"]),
        "compute/peak_cuda_bytes": int(result["peak_cuda_bytes"]),
        "compute/peak_cuda_GiB": float(result["peak_cuda_GiB"]),
        "compute/wall_seconds": float(result["wall_seconds"]),
        "provenance/paper0_commit": str(result["paper0_commit"]),
        "provenance/selected_checkpoint_sha256": str(
            artifacts["selected_checkpoint"]["sha256"]
        ),
        "provenance/final_training_state_sha256": str(
            artifacts["final_training_state"]["sha256"]
        ),
        "provenance/history_sha256": str(artifacts["history"]["sha256"]),
        "provenance/training_order_sha256": str(artifacts["training_order"]["sha256"]),
        "provenance/validation_seed_bank_sha256": str(
            artifacts["validation_seed_bank"]["sha256"]
        ),
        "scope/physics_derived_loss_used": bool(result["physics_derived_loss_used"]),
        "scope/physics_metric_used_for_checkpoint_selection": bool(
            result["physics_metric_used_for_checkpoint_selection"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/scientific_forecast_generated": bool(
            result["scientific_forecast_generated"]
        ),
        "scope/scientific_acceptance_evaluated": bool(
            result["scientific_acceptance_evaluated"]
        ),
    }


class B5EDMFullOnlineWandbTracker(OnlineWandbTracker):
    """Required online full B5 run with verified remote completion."""

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            b5_full_epoch_metrics(record),
            step=int(record["global_optimizer_step"]),
            commit=True,
        )
        self.epochs_logged += 1

    def finish_success(self, result: Mapping[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        if self.epochs_logged != 100 or int(result["completed_epochs"]) != 100:
            raise RuntimeError("B5 full W&B requires all 100 epochs")
        if int(result["candidate_count"]) != 20:
            raise RuntimeError("B5 full W&B requires all 20 candidates")
        if result.get("checkpoint_reload_bitwise_exact") is not True:
            raise RuntimeError("B5 full selected checkpoint did not reload exactly")
        if result.get("all_losses_and_gradients_finite") is not True:
            raise RuntimeError("B5 full training contains a non-finite value")
        self._run.summary.update(b5_full_result_summary(result))
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True

        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = self._module.Api(timeout=30).run(remote_path)
        if str(remote.id) != self.spec.run_id:
            raise RuntimeError("remote B5 full W&B identity differs after finish")
        remote_state = str(remote.state)
        if remote_state != "finished":
            raise RuntimeError(
                f"remote B5 full W&B state is {remote_state!r}, not 'finished'"
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
            "epochs_logged": self.epochs_logged,
            "checkpoints_uploaded": False,
            "samples_uploaded": False,
            "local_artifacts_are_scientific_authority": True,
        }
