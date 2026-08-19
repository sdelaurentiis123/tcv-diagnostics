"""Strict online W&B mirror for the full seed-1701 B4 training run."""

from __future__ import annotations

from typing import Any, Mapping

from .wandb_tracking import OnlineWandbTracker


def full_refiner_epoch_metrics(
    record: Mapping[str, Any],
) -> dict[str, int | float]:
    """Map one authoritative full-training epoch, with sparse validation."""

    level_losses = record["train_MSE_by_level"]
    level_counts = record["train_count_by_level"]
    if tuple(level_losses) != ("0", "1", "2", "3"):
        raise ValueError("full B4 W&B training levels differ")
    if tuple(level_counts) != ("0", "1", "2", "3"):
        raise ValueError("full B4 W&B training counts differ")
    selected = record.get("selected_so_far")
    metrics: dict[str, int | float] = {
        "epoch": int(record["epoch"]),
        "completed_epoch": int(record["completed_epoch"]),
        "optimizer/global_step": int(record["global_step"]),
        "optimizer/learning_rate": float(record["learning_rate"]),
        "optimizer/ema_decay": float(record["EMA_decay"]),
        "optimizer/ema_updates": int(record["EMA_updates"]),
        "optimizer/mean_preclip_total_gradient_norm": float(
            record["mean_preclip_total_gradient_norm"]
        ),
        "optimizer/maximum_preclip_total_gradient_norm": float(
            record["maximum_preclip_total_gradient_norm"]
        ),
        "optimizer/mean_preclip_parent_gradient_norm": float(
            record["mean_preclip_parent_gradient_norm"]
        ),
        "optimizer/mean_preclip_refinement_gradient_norm": float(
            record["mean_preclip_refinement_gradient_norm"]
        ),
        "train/standardized_latent_mse": float(
            record["train_standardized_latent_MSE"]
        ),
        "validation/performed": int(bool(record["validation_performed"])),
        "validation/selected_epoch_so_far": (
            -1 if selected is None else int(selected)
        ),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }
    for level in range(4):
        metrics[f"train/level_{level}_mse"] = float(level_losses[str(level)])
        metrics[f"train/level_{level}_count"] = int(level_counts[str(level)])

    validation = record.get("validation")
    if bool(record["validation_performed"]):
        if not isinstance(validation, Mapping):
            raise ValueError("full B4 validation record is missing")
        fields = validation["final_MAE_by_channel"]
        if tuple(fields) != ("Ne", "Pe", "Pi", "phi", "Vi"):
            raise ValueError("full B4 validation fields differ")
        level_mae = validation["equal_channel_MAE_by_level"]
        if len(level_mae) != 4:
            raise ValueError("full B4 validation level count differs")
        metrics[
            "validation/ensemble_mean_equal_channel_decoded_standardized_mae"
        ] = float(
            validation[
                "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
            ]
        )
        for level, value in enumerate(level_mae):
            metrics[f"validation/level_{level}_equal_channel_mae"] = float(value)
        for field, value in fields.items():
            metrics[f"validation/final_channel_mae/{field}"] = float(value)
    elif validation is not None:
        raise ValueError("full B4 non-validation epoch contains validation")
    return metrics


def full_refiner_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Select compact training facts; immutable Ceph artifacts are authority."""

    selected = result["selected_validation"]
    final = result["final_validation"]
    return {
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(result["completed_optimizer_steps"]),
        "final/ema_updates": int(result["EMA_updates"]),
        "final/validation_candidates_evaluated": int(
            result["validation_candidates_evaluated"]
        ),
        "final/selected_epoch": int(result["selected_epoch"]),
        "final/selected_completed_epoch": int(result["selected_completed_epoch"]),
        "final/selected_optimizer_step": int(result["selected_optimizer_step"]),
        "final/selected_validation_equal_channel_mae": float(
            selected[
                "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
            ]
        ),
        "final/final_validation_equal_channel_mae": float(
            final[
                "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
            ]
        ),
        "final/preoptimization_parent_bitwise_exact": bool(
            result["preoptimization_parent_identity"]["bitwise_exact"]
        ),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
        ),
        "final/codec_bitwise_unchanged": bool(result["codec_bitwise_unchanged"]),
        "optimizer/parent_parameter_gradient_seen": bool(
            result["parent_parameter_gradient_seen"]
        ),
        "optimizer/refinement_parameter_gradient_seen": bool(
            result["refinement_parameter_gradient_seen"]
        ),
        "optimizer/all_four_training_levels_exercised": bool(
            result["all_four_training_levels_exercised"]
        ),
        "compute/parameter_count": int(result["parameter_count"]),
        "compute/network_calls_per_unamortized_member": int(
            result["network_calls_per_unamortized_member"]
        ),
        "compute/peak_cuda_bytes": int(result["peak_cuda_bytes"]),
        "compute/wall_seconds": float(result["wall_seconds"]),
        "precision/training_float32": result["training_dtype"] == "float32",
        "precision/tf32_disabled": not bool(result["cuda_matmul_allow_tf32"])
        and not bool(result["cudnn_allow_tf32"]),
        "provenance/paper0_commit": str(result["paper0_commit"]),
        "provenance/selected_checkpoint_sha256": str(
            result["selected_checkpoint"]["sha256"]
        ),
        "provenance/final_training_state_sha256": str(
            result["final_training_state"]["sha256"]
        ),
        "provenance/history_sha256": str(result["history"]["sha256"]),
        "provenance/validation_seed_bank_sha256": str(
            result["validation_seed_bank"]["sha256"]
        ),
        "provenance/training_levels_sha256": str(
            result["training_levels"]["sha256"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/physics_derived_loss_used": bool(
            result["physics_derived_loss_used"]
        ),
        "scope/scientific_result": bool(result["scientific_result"]),
        "scope/training_complete_is_scientific_acceptance": bool(
            result["training_complete_is_scientific_acceptance"]
        ),
        "scope/H_det_evaluated": bool(result["H_det_evaluated"]),
        "scope/H_prob_evaluated": bool(result["H_prob_evaluated"]),
    }


class PDERefinerFullOnlineWandbTracker(OnlineWandbTracker):
    """Required online full B4 run with remote completion verification."""

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            full_refiner_epoch_metrics(record),
            step=int(record["global_step"]),
            commit=True,
        )
        self.epochs_logged += 1

    def finish_success(self, result: Mapping[str, Any]) -> dict[str, Any]:
        if self._finished:
            raise RuntimeError("W&B run is already finished")
        expected_epochs = int(result["completed_epochs"])
        if self.epochs_logged != expected_epochs:
            raise RuntimeError(
                f"W&B logged {self.epochs_logged} epochs, expected {expected_epochs}"
            )
        self._run.summary.update(full_refiner_result_summary(result))
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True

        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = self._module.Api(timeout=30).run(remote_path)
        if str(remote.id) != self.spec.run_id:
            raise RuntimeError("remote full B4 W&B identity differs after finish")
        remote_state = str(remote.state)
        if remote_state != "finished":
            raise RuntimeError(
                f"remote full B4 W&B state is {remote_state!r}, not 'finished'"
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
            "local_artifacts_are_scientific_authority": True,
        }
