"""Strict online Weights & Biases tracking for Paper 0 training jobs.

W&B is a monitoring mirror.  The immutable local configuration, JSON history,
result record, and checkpoint hashes remain the scientific source of truth.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib
from pathlib import Path
import re
from typing import Any, Mapping


_SLUG = re.compile(r"^[A-Za-z0-9_.-]+$")


def _validated_slug(value: str, label: str) -> str:
    text = str(value)
    if not text or not _SLUG.fullmatch(text):
        raise ValueError(f"{label} must contain only letters, digits, '.', '_', or '-'")
    return text


@dataclass(frozen=True)
class WandbRunSpec:
    entity: str
    project: str
    group: str
    run_id: str
    run_name: str
    job_type: str
    tags: tuple[str, ...]

    def __post_init__(self) -> None:
        for label in ("entity", "project", "group", "run_id", "job_type"):
            _validated_slug(getattr(self, label), label)
        if not self.run_name.strip() or "/" in self.run_name:
            raise ValueError("run_name must be nonempty and cannot contain '/'")
        for tag in self.tags:
            _validated_slug(tag, "tag")

    def to_record(self) -> dict[str, Any]:
        record = asdict(self)
        record["tags"] = list(self.tags)
        return record


def wandb_epoch_metrics(record: Mapping[str, Any]) -> dict[str, int | float]:
    """Map one authoritative local epoch record to stable W&B metric names."""

    channels = record["validation_mae_by_channel"]
    if not isinstance(channels, Mapping) or not channels:
        raise ValueError("epoch record has no channel validation metrics")
    metrics: dict[str, int | float] = {
        "epoch": int(record["epoch"]),
        "optimizer/global_step": int(record["global_step"]),
        "optimizer/learning_rate": float(record["learning_rate"]),
        "optimizer/mean_preclip_gradient_norm": float(
            record["mean_preclip_gradient_norm"]
        ),
        "optimizer/maximum_preclip_gradient_norm": float(
            record["maximum_preclip_gradient_norm"]
        ),
        "train/equal_channel_standardized_mae": float(
            record["train_equal_channel_mae"]
        ),
        "validation/equal_channel_standardized_mae": float(
            record["validation_equal_channel_mae"]
        ),
        "validation/selected_epoch_so_far": int(record["selected_so_far"]),
        "timing/epoch_wall_seconds": float(record["epoch_wall_seconds"]),
    }
    for field, value in channels.items():
        metrics[f"validation/channel_mae/{field}"] = float(value)
    return metrics


def wandb_result_summary(result: Mapping[str, Any]) -> dict[str, int | float | str | bool]:
    """Select compact final facts; checkpoints themselves are never uploaded."""

    summary: dict[str, int | float | str | bool] = {
        "final/completed_epochs": int(result["completed_epochs"]),
        "final/completed_optimizer_steps": int(result["completed_optimizer_steps"]),
        "final/selected_epoch": int(result["selected_epoch"]),
        "final/selected_validation_equal_channel_standardized_mae": float(
            result["selected_validation_equal_channel_mae"]
        ),
        "final/final_validation_equal_channel_standardized_mae": float(
            result["final_validation_equal_channel_mae"]
        ),
        "final/checkpoint_reload_bitwise_exact": bool(
            result["checkpoint_reload_bitwise_exact"]
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
        "scope/physics_derived_loss_used": bool(result["physics_derived_loss_used"]),
    }
    for field, value in result["final_validation_mae_by_channel"].items():
        summary[f"final/channel_mae/{field}"] = float(value)
    return summary


class OnlineWandbTracker:
    """A fail-closed online W&B run with an auditable completion record."""

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
    ) -> "OnlineWandbTracker":
        module = wandb_module
        if module is None:
            try:
                module = importlib.import_module("wandb")
            except ImportError as error:
                raise RuntimeError("online W&B tracking is required but wandb is absent") from error

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
        metrics = wandb_epoch_metrics(record)
        self._run.log(
            metrics,
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
        self._run.summary.update(wandb_result_summary(result))
        run_url = str(self._run.url)
        self._run.finish(exit_code=0)
        self._finished = True

        remote_path = f"{self.spec.entity}/{self.spec.project}/{self.spec.run_id}"
        remote = self._module.Api(timeout=30).run(remote_path)
        if str(remote.id) != self.spec.run_id:
            raise RuntimeError("remote W&B run identity differs after finish")
        remote_state = str(remote.state)
        if remote_state != "finished":
            raise RuntimeError(f"remote W&B run state is {remote_state!r}, not 'finished'")
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
