"""Strict online W&B mirror for the bounded B4 PDE-Refiner smoke."""

from __future__ import annotations

from typing import Any, Mapping

from .wandb_tracking import OnlineWandbTracker


def pde_refiner_wandb_epoch_metrics(
    record: Mapping[str, Any],
) -> dict[str, int | float]:
    """Map one authoritative B4 epoch record to stable W&B names."""

    fields = tuple(record["validation_final_MAE_by_channel"])
    if fields != ("Ne", "Pe", "Pi", "phi", "Vi"):
        raise ValueError("B4 W&B epoch fields differ from frozen C5P order")
    level_losses = record["train_MSE_by_level"]
    level_counts = record["train_count_by_level"]
    level_mae = record["validation_equal_channel_MAE_by_level"]
    if tuple(level_losses) != ("0", "1", "2", "3"):
        raise ValueError("B4 W&B training levels differ")
    if tuple(level_counts) != ("0", "1", "2", "3") or len(level_mae) != 4:
        raise ValueError("B4 W&B level metrics differ")
    metrics: dict[str, int | float] = {
        "epoch": int(record["epoch"]),
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
        "validation/ensemble_mean_equal_channel_decoded_standardized_mae": float(
            record[
                "validation_ensemble_mean_equal_channel_decoded_standardized_field_MAE"
            ]
        ),
        "validation/selected_epoch_so_far": int(record["selected_so_far"]),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }
    for level in range(4):
        metrics[f"train/level_{level}_mse"] = float(level_losses[str(level)])
        metrics[f"train/level_{level}_count"] = int(level_counts[str(level)])
        metrics[f"validation/level_{level}_equal_channel_mae"] = float(
            level_mae[level]
        )
    for field, value in record["validation_final_MAE_by_channel"].items():
        metrics[f"validation/final_channel_mae/{field}"] = float(value)
    return metrics


def pde_refiner_wandb_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Select compact B4 smoke facts; local artifacts remain authoritative."""

    selected = result["selected_validation"]
    final = result["final_validation"]
    probe = result["member_and_stage_probe"]
    summary: dict[str, int | float | str | bool] = {
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(result["completed_optimizer_steps"]),
        "final/ema_updates": int(result["EMA_updates"]),
        "final/selected_epoch": int(result["selected_epoch"]),
        "final/selected_optimizer_step": int(result["selected_optimizer_step"]),
        "final/selected_validation_equal_channel_mae": float(
            selected[
                "ensemble_mean_equal_channel_decoded_standardized_field_MAE"
            ]
        ),
        "final/final_validation_equal_channel_mae": float(
            final["ensemble_mean_equal_channel_decoded_standardized_field_MAE"]
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
        "ensemble/level0_shared_bitwise_across_members": bool(
            probe["level0_shared_bitwise_across_members"]
        ),
        "ensemble/nonzero_final_diversity_in_every_field": bool(
            probe["nonzero_final_diversity_in_every_field"]
        ),
        "compute/parameter_count": int(result["parameter_count"]),
        "compute/network_calls_per_member": int(result["network_calls_per_member"]),
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
        "provenance/decoded_stages_sha256": str(
            result["validation_decoded_stages"]["sha256"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/physics_derived_loss_used": bool(
            result["physics_derived_loss_used"]
        ),
        "scope/scientific_result": bool(result["scientific_result"]),
        "scope/full_B4_training_authorized": bool(
            result["full_B4_training_authorized"]
        ),
    }
    for field, value in probe["final_member_RMS_difference_by_field"].items():
        summary[f"ensemble/final_member_rms_difference/{field}"] = float(value)
    return summary


class PDERefinerOnlineWandbTracker(OnlineWandbTracker):
    """Required online B4 run with remote completion verification."""

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            pde_refiner_wandb_epoch_metrics(record),
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
        self._run.summary.update(pde_refiner_wandb_result_summary(result))
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True

        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = self._module.Api(timeout=30).run(remote_path)
        if str(remote.id) != self.spec.run_id:
            raise RuntimeError("remote B4 W&B run identity differs after finish")
        remote_state = str(remote.state)
        if remote_state != "finished":
            raise RuntimeError(
                f"remote B4 W&B state is {remote_state!r}, not 'finished'"
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
