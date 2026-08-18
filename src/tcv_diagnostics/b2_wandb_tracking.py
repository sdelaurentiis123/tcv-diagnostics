"""Fail-closed online W&B mirror for Paper 0 B2 training runs."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any, Mapping

from .wandb_tracking import WandbRunSpec


def b2_wandb_epoch_metrics(record: Mapping[str, Any]) -> dict[str, int | float]:
    """Map the authoritative local B2 epoch record to stable metric names."""

    return {
        "epoch": int(record["epoch"]),
        "optimizer/global_step": int(record["global_step"]),
        "optimizer/learning_rate": float(record["learning_rate"]),
        "optimizer/mean_preclip_gradient_norm": float(
            record["mean_preclip_gradient_norm"]
        ),
        "optimizer/maximum_preclip_gradient_norm": float(
            record["maximum_preclip_gradient_norm"]
        ),
        "train/complete_denoising_loss": float(
            record["train_complete_denoising_loss"]
        ),
        "train/context_denoising_loss": float(
            record["train_context_denoising_loss"]
        ),
        "train/target_denoising_loss": float(
            record["train_target_denoising_loss"]
        ),
        "validation/complete_denoising_loss": float(
            record["validation_complete_denoising_loss"]
        ),
        "validation/context_denoising_loss": float(
            record["validation_context_denoising_loss"]
        ),
        "validation/target_denoising_loss": float(
            record["validation_target_denoising_loss"]
        ),
        "validation/selected_epoch_so_far": int(record["selected_so_far"]),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }


def b2_wandb_result_summary(
    result: Mapping[str, Any],
) -> dict[str, int | float | str | bool]:
    """Select compact run facts; checkpoints remain local scientific artifacts."""

    selected = result["selected_validation"]
    final = result["final_validation"]
    sampler = result["sampler_probe"]
    return {
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(
            result["completed_optimizer_steps"]
        ),
        "final/selected_epoch": int(result["selected_epoch"]),
        "final/selected_validation_complete_denoising_loss": float(
            selected["complete"]
        ),
        "final/selected_validation_context_denoising_loss": float(
            selected["context"]
        ),
        "final/selected_validation_target_denoising_loss": float(
            selected["target"]
        ),
        "final/final_validation_complete_denoising_loss": float(final["complete"]),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
        ),
        "sampler/latent_member_rms_difference": float(
            sampler["latent_member_rms_difference"]
        ),
        "sampler/field_member_rms_difference": float(
            sampler["field_member_rms_difference"]
        ),
        "sampler/nonzero_latent_diversity": bool(
            sampler["nonzero_latent_diversity"]
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
        "scope/held_out_85606_read": bool(result["held_out_85606_read"]),
        "scope/physics_derived_loss_used": bool(
            result["physics_derived_loss_used"]
        ),
        "scope/scientific_result": bool(result["scientific_result"]),
        "scope/full_B2_training_authorized": bool(
            result["full_B2_training_authorized"]
        ),
    }


class B2OnlineWandbTracker:
    """A required online run with remote completion verification."""

    def __init__(
        self,
        *,
        module: Any,
        run: Any,
        spec: WandbRunSpec,
        authenticated_username: str,
        tracking_directory: Path,
    ) -> None:
        self._module = module
        self._run = run
        self.spec = spec
        self.authenticated_username = authenticated_username
        self.tracking_directory = tracking_directory
        self.epochs_logged = 0
        self._finished = False

    @classmethod
    def start(
        cls,
        *,
        spec: WandbRunSpec,
        config: Mapping[str, Any],
        tracking_directory: Path,
        wandb_module: Any | None = None,
    ) -> "B2OnlineWandbTracker":
        module = wandb_module
        if module is None:
            try:
                module = importlib.import_module("wandb")
            except ImportError as error:
                raise RuntimeError(
                    "online W&B tracking is required but wandb is absent"
                ) from error

        api = module.Api(timeout=30)
        if not bool(getattr(api, "api_key", None)):
            raise RuntimeError("online W&B tracking is required but no API key is configured")
        viewer = api.viewer
        viewer_entity = str(getattr(viewer, "entity", ""))
        username = str(getattr(viewer, "username", ""))
        if viewer_entity != spec.entity:
            raise RuntimeError(
                f"authenticated W&B entity {viewer_entity!r} != required {spec.entity!r}"
            )

        directory = Path(tracking_directory)
        if directory.exists():
            raise FileExistsError(f"refusing to reuse W&B directory {directory}")
        directory.mkdir(parents=True)
        run = module.init(
            entity=spec.entity,
            project=spec.project,
            group=spec.group,
            name=spec.run_name,
            id=spec.run_id,
            resume="never",
            job_type=spec.job_type,
            tags=list(spec.tags),
            config=dict(config),
            mode="online",
            dir=str(directory),
            save_code=False,
            settings=module.Settings(init_timeout=120),
        )
        if run is None:
            raise RuntimeError("wandb.init returned no run")
        if bool(run.offline):
            run.finish(exit_code=1)
            raise RuntimeError("W&B unexpectedly initialized offline")
        if str(run.id) != spec.run_id:
            run.finish(exit_code=1)
            raise RuntimeError(f"W&B run ID {run.id!r} != required {spec.run_id!r}")
        if not str(run.url).startswith("https://"):
            run.finish(exit_code=1)
            raise RuntimeError("W&B run has no online URL")
        return cls(
            module=module,
            run=run,
            spec=spec,
            authenticated_username=username,
            tracking_directory=directory,
        )

    def log_epoch(self, record: Mapping[str, Any]) -> None:
        if self._finished:
            raise RuntimeError("cannot log to a finished W&B run")
        self._run.log(
            b2_wandb_epoch_metrics(record),
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
        self._run.summary.update(b2_wandb_result_summary(result))
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

    def finish_failure(self) -> None:
        if self._finished:
            return
        try:
            self._run.finish(exit_code=1)
        finally:
            self._finished = True
