"""Strict online W&B mirror for the bounded B3 FGN smoke."""

from __future__ import annotations

from typing import Any, Mapping

from .wandb_tracking import OnlineWandbTracker


def fgn_wandb_epoch_metrics(
    record: Mapping[str, Any],
) -> dict[str, int | float]:
    """Map one authoritative B3 epoch record to stable W&B names."""

    fields = tuple(record["validation_fair_crps_by_channel"])
    if fields != ("Ne", "Pe", "Pi", "phi", "Vi"):
        raise ValueError("B3 W&B epoch fields differ from frozen C5P order")
    metrics: dict[str, int | float] = {
        "epoch": int(record["epoch"]),
        "optimizer/global_step": int(record["global_step"]),
        "optimizer/common_learning_rate": float(record["common_learning_rate"]),
        "optimizer/new_learning_rate": float(record["new_learning_rate"]),
        "optimizer/mean_preclip_total_gradient_norm": float(
            record["mean_preclip_total_gradient_norm"]
        ),
        "optimizer/maximum_preclip_total_gradient_norm": float(
            record["maximum_preclip_total_gradient_norm"]
        ),
        "optimizer/mean_preclip_common_gradient_norm": float(
            record["mean_preclip_common_gradient_norm"]
        ),
        "optimizer/mean_preclip_new_gradient_norm": float(
            record["mean_preclip_new_gradient_norm"]
        ),
        "train/equal_channel_fair_crps": float(
            record["train_equal_channel_fair_crps"]
        ),
        "validation/equal_channel_fair_crps": float(
            record["validation_equal_channel_fair_crps"]
        ),
        "validation/selected_epoch_so_far": int(record["selected_so_far"]),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }
    families = {
        "train/fair_crps": record["train_fair_crps_by_channel"],
        "train/accuracy": record["train_accuracy_by_channel"],
        "train/spread": record["train_spread_by_channel"],
        "validation/fair_crps": record["validation_fair_crps_by_channel"],
        "validation/accuracy": record["validation_accuracy_by_channel"],
        "validation/spread": record["validation_spread_by_channel"],
    }
    for prefix, values in families.items():
        if tuple(values) != fields:
            raise ValueError(f"B3 W&B {prefix} fields differ")
        for field, value in values.items():
            metrics[f"{prefix}/{field}"] = float(value)
    return metrics


def fgn_wandb_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Select compact B3 facts; checkpoints remain local artifacts."""

    selected = result["selected_validation"]
    final = result["final_validation"]
    probe = result["member_probe"]
    return {
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(result["completed_optimizer_steps"]),
        "final/selected_epoch": int(result["selected_epoch"]),
        "final/selected_validation_equal_channel_fair_crps": float(
            selected["equal_channel_fair_crps"]
        ),
        "final/final_validation_equal_channel_fair_crps": float(
            final["equal_channel_fair_crps"]
        ),
        "final/preoptimization_parent_bitwise_exact": bool(
            result["preoptimization_parent_identity"]["bitwise_exact"]
        ),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
        ),
        "final/codec_bitwise_unchanged": bool(result["codec_bitwise_unchanged"]),
        "optimizer/common_parameter_gradient_seen": bool(
            result["common_parameter_gradient_seen"]
        ),
        "optimizer/new_parameter_gradient_seen": bool(
            result["new_parameter_gradient_seen"]
        ),
        "ensemble/latent_member_rms_difference": float(
            probe["latent_member_rms_difference"]
        ),
        "ensemble/field_member_rms_difference": float(
            probe["field_member_rms_difference"]
        ),
        "ensemble/nonzero_latent_diversity": bool(
            probe["nonzero_latent_diversity"]
        ),
        "ensemble/nonzero_field_diversity": bool(
            probe["nonzero_field_diversity"]
        ),
        "compute/parameter_count": int(result["parameter_count"]),
        "compute/peak_cuda_bytes": int(result["peak_cuda_bytes"]),
        "compute/wall_seconds": float(result["wall_seconds"]),
        "provenance/paper0_commit": str(result["paper0_commit"]),
        "provenance/selected_checkpoint_sha256": str(
            result["selected_checkpoint"]["sha256"]
        ),
        "provenance/final_training_state_sha256": str(
            result["final_training_state"]["sha256"]
        ),
        "provenance/history_sha256": str(result["history"]["sha256"]),
        "provenance/validation_noise_sha256": str(
            result["validation_noise_bank"]["sha256"]
        ),
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/physics_derived_loss_used": bool(
            result["physics_derived_loss_used"]
        ),
        "scope/scientific_result": bool(result["scientific_result"]),
        "scope/full_B3_training_authorized": bool(
            result["full_B3_training_authorized"]
        ),
    }


class FGNOnlineWandbTracker(OnlineWandbTracker):
    """Required online B3 run with remote completion verification."""

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            fgn_wandb_epoch_metrics(record),
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
        self._run.summary.update(fgn_wandb_result_summary(result))
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True

        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = self._module.Api(timeout=30).run(remote_path)
        if str(remote.id) != self.spec.run_id:
            raise RuntimeError("remote W&B run identity differs after finish")
        remote_state = str(remote.state)
        if remote_state != "finished":
            raise RuntimeError(
                f"remote W&B run state is {remote_state!r}, not 'finished'"
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
