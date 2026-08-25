"""Streaming field and marginal-calibration scoring for B2 forecast files."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np

from .b2_field_metrics import (
    B2_ALL_REGIONS,
    B2_FIELDS,
    B2_INTERVALS,
    B2_PHI_INDEX,
    B2_PRIMARY_REGIONS,
    FieldRegionAccumulator,
    PointwiseEnsembleDiagnostics,
    gauge_fix_phi_channel,
    pointwise_ensemble_diagnostics,
)
from .b2_forecast import sampler_seed
from .b2_probabilistic_metrics import (
    monte_carlo_stability,
    moving_block_bootstrap_indices,
)
from .metrics import fair_crps, ordinary_crps


B2_MEMBER_PREFIXES = (4, 8, 16, 32)
B2_VALIDATION_TARGETS = tuple(range(498, 624))
B2_VALIDATION_BLOCKS = tuple(
    B2_VALIDATION_TARGETS[start : start + 21]
    for start in range(0, len(B2_VALIDATION_TARGETS), 21)
)


def _finite_real(name: str, values: np.ndarray) -> np.ndarray:
    array = np.asarray(values)
    if np.iscomplexobj(array) or not np.issubdtype(array.dtype, np.number):
        raise TypeError(f"{name} must be real numeric")
    result = np.asarray(array, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


class PrefixFieldAccumulator:
    """Stream the frozen prefix CRPS and spread-skill sensitivity."""

    def __init__(self, members: int) -> None:
        if int(members) not in B2_MEMBER_PREFIXES:
            raise ValueError("B2 member prefix is outside the frozen set")
        self.members = int(members)
        self.count = 0
        self.squared_error_sum = 0.0
        self.absolute_error_sum = 0.0
        self.fair_crps_sum = 0.0
        self.ordinary_crps_sum = 0.0
        self.member_variance_sum = 0.0

    def update_raw(
        self,
        forecast: np.ndarray,
        truth: np.ndarray,
        mask: np.ndarray,
    ) -> None:
        forecast_array = _finite_real("prefix forecast", forecast)
        truth_array = _finite_real("prefix truth", truth)
        selected = np.asarray(mask, dtype=bool)
        if forecast_array.shape != (self.members, truth_array.size):
            raise ValueError("B2 prefix forecast shape differs")
        if truth_array.ndim != 1 or selected.shape != truth_array.shape:
            raise ValueError("B2 prefix truth/mask shape differs")
        if not np.any(selected):
            raise ValueError("B2 prefix mask is empty")
        forecast_values = forecast_array[:, selected]
        truth_values = truth_array[selected]
        ensemble_mean = np.mean(forecast_values, axis=0)
        error = ensemble_mean - truth_values
        self.count += int(truth_values.size)
        self.squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        self.absolute_error_sum += float(
            np.sum(np.abs(error), dtype=np.float64)
        )
        self.fair_crps_sum += float(
            np.sum(
                fair_crps(forecast_values, truth_values, member_axis=0),
                dtype=np.float64,
            )
        )
        self.ordinary_crps_sum += float(
            np.sum(
                ordinary_crps(forecast_values, truth_values, member_axis=0),
                dtype=np.float64,
            )
        )
        self.member_variance_sum += float(
            np.sum(np.var(forecast_values, axis=0, ddof=1), dtype=np.float64)
        )

    def update_primary(
        self,
        diagnostics: PointwiseEnsembleDiagnostics,
        mask: np.ndarray,
    ) -> None:
        if self.members != 32 or diagnostics.members != 32:
            raise ValueError("primary diagnostic reuse applies only to M32")
        selected = np.asarray(mask, dtype=bool)
        if selected.shape != diagnostics.error.shape or not np.any(selected):
            raise ValueError("B2 primary prefix mask differs")
        error = diagnostics.error[selected]
        self.count += int(np.sum(selected))
        self.squared_error_sum += float(np.sum(error * error, dtype=np.float64))
        self.absolute_error_sum += float(
            np.sum(np.abs(error), dtype=np.float64)
        )
        self.fair_crps_sum += float(
            np.sum(diagnostics.fair_crps[selected], dtype=np.float64)
        )
        self.ordinary_crps_sum += float(
            np.sum(diagnostics.ordinary_crps[selected], dtype=np.float64)
        )
        self.member_variance_sum += float(
            np.sum(
                diagnostics.unbiased_member_variance[selected], dtype=np.float64
            )
        )

    def finalize(self) -> dict[str, Any]:
        if self.count <= 0:
            raise ValueError("cannot finalize an empty B2 prefix score")
        count = float(self.count)
        rmse = math.sqrt(self.squared_error_sum / count)
        spread = math.sqrt(
            ((self.members + 1) / self.members)
            * self.member_variance_sum
            / count
        )
        return {
            "ensemble_size": self.members,
            "scalar_count": self.count,
            "ensemble_mean_rmse": rmse,
            "ensemble_mean_mae": self.absolute_error_sum / count,
            "fair_crps": self.fair_crps_sum / count,
            "ordinary_empirical_crps": self.ordinary_crps_sum / count,
            "mean_unbiased_member_variance": self.member_variance_sum / count,
            "finite_member_variance_factor": (self.members + 1) / self.members,
            "corrected_rms_spread": spread,
            "corrected_spread_skill_ratio": spread / rmse if rmse > 0.0 else None,
        }


def _target_sufficient_record(
    diagnostics: PointwiseEnsembleDiagnostics,
    mask: np.ndarray,
) -> dict[str, Any]:
    selected = np.asarray(mask, dtype=bool)
    count = int(np.sum(selected))
    if count <= 0:
        raise ValueError("B2 per-target sufficient mask is empty")
    error = diagnostics.error[selected]
    return {
        "count": count,
        "squared_error_sum": float(np.sum(error * error, dtype=np.float64)),
        "absolute_error_sum": float(np.sum(np.abs(error), dtype=np.float64)),
        "fair_crps_sum": float(
            np.sum(diagnostics.fair_crps[selected], dtype=np.float64)
        ),
        "ordinary_crps_sum": float(
            np.sum(diagnostics.ordinary_crps[selected], dtype=np.float64)
        ),
        "member_variance_sum": float(
            np.sum(
                diagnostics.unbiased_member_variance[selected], dtype=np.float64
            )
        ),
        "coverage_counts": {
            name: int(np.count_nonzero(diagnostics.interval_covered[name][selected]))
            for name in B2_INTERVALS
        },
        "rank_counts": np.bincount(
            diagnostics.ranks[selected], minlength=33
        ).astype(np.int64).tolist(),
    }


def _aggregate_region_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(records) != B2_FIELDS:
        raise ValueError("B2 aggregate field order differs")
    counts = {int(record["scalar_count"]) for record in records.values()}
    if len(counts) != 1:
        raise ValueError("B2 fields use unequal spatial counts in one region")
    aggregate_rmse = math.sqrt(
        np.mean(
            [record["ensemble_mean"]["rmse"] ** 2 for record in records.values()]
        )
    )
    aggregate_member_variance = float(
        np.mean(
            [
                record["corrected_spread_skill"][
                    "mean_unbiased_member_variance"
                ]
                for record in records.values()
            ]
        )
    )
    aggregate_spread = math.sqrt((33 / 32) * aggregate_member_variance)
    return {
        "equal_channel_ensemble_mean_rmse": aggregate_rmse,
        "equal_channel_ensemble_mean_mae": float(
            np.mean([record["ensemble_mean"]["mae"] for record in records.values()])
        ),
        "equal_channel_ensemble_mean_bias": float(
            np.mean([record["ensemble_mean"]["bias"] for record in records.values()])
        ),
        "equal_channel_fair_crps": float(
            np.mean([record["fair_crps"] for record in records.values()])
        ),
        "equal_channel_ordinary_empirical_crps": float(
            np.mean(
                [record["ordinary_empirical_crps"] for record in records.values()]
            )
        ),
        "equal_channel_corrected_rms_spread": aggregate_spread,
        "equal_channel_corrected_spread_skill_ratio": (
            aggregate_spread / aggregate_rmse if aggregate_rmse > 0.0 else None
        ),
        "all_fields_nonzero_spread": all(
            record["spread_integrity"]["nonzero_spread"]
            for record in records.values()
        ),
    }


def _aggregate_prefix_records(
    records: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if tuple(records) != B2_FIELDS:
        raise ValueError("B2 prefix aggregate field order differs")
    counts = {int(record["scalar_count"]) for record in records.values()}
    members = {int(record["ensemble_size"]) for record in records.values()}
    if len(counts) != 1 or len(members) != 1:
        raise ValueError("B2 prefix field records are not matched")
    member_count = next(iter(members))
    aggregate_rmse = math.sqrt(
        np.mean([record["ensemble_mean_rmse"] ** 2 for record in records.values()])
    )
    aggregate_member_variance = float(
        np.mean(
            [record["mean_unbiased_member_variance"] for record in records.values()]
        )
    )
    aggregate_spread = math.sqrt(
        ((member_count + 1) / member_count) * aggregate_member_variance
    )
    return {
        "ensemble_size": member_count,
        "equal_channel_ensemble_mean_rmse": aggregate_rmse,
        "equal_channel_ensemble_mean_mae": float(
            np.mean([record["ensemble_mean_mae"] for record in records.values()])
        ),
        "equal_channel_fair_crps": float(
            np.mean([record["fair_crps"] for record in records.values()])
        ),
        "equal_channel_ordinary_empirical_crps": float(
            np.mean(
                [record["ordinary_empirical_crps"] for record in records.values()]
            )
        ),
        "equal_channel_corrected_rms_spread": aggregate_spread,
        "equal_channel_corrected_spread_skill_ratio": (
            aggregate_spread / aggregate_rmse if aggregate_rmse > 0.0 else None
        ),
    }


def _quantile_interval(values: np.ndarray) -> dict[str, float]:
    array = _finite_real("bootstrap values", values)
    return {
        "median": float(np.quantile(array, 0.5, method="linear")),
        "lower_2p5": float(np.quantile(array, 0.025, method="linear")),
        "upper_97p5": float(np.quantile(array, 0.975, method="linear")),
    }


class B2FieldScoreAccumulator:
    """Stream all field/region scores once, including six frozen blocks."""

    def __init__(
        self,
        *,
        model_seed: int,
        target_frames: Sequence[int],
        region_masks: Mapping[str, np.ndarray],
        volume_shape: tuple[int, int, int] = (64, 32, 88),
        validation_blocks: Sequence[Sequence[int]] | None = None,
        allow_sparse_targets: bool = False,
    ) -> None:
        if int(model_seed) not in (1701, 1702, 1703):
            raise ValueError("B2 field scorer model seed differs")
        targets = tuple(int(item) for item in target_frames)
        if not targets or targets != tuple(sorted(set(targets))):
            raise ValueError("B2 field scorer targets must be strictly increasing")
        contiguous = targets == tuple(range(targets[0], targets[-1] + 1))
        if not contiguous and not bool(allow_sparse_targets):
            raise ValueError("B2 field scorer targets must be contiguous")
        self.model_seed = int(model_seed)
        self.target_frames = targets
        self.sparse_targets = not contiguous
        self.volume_shape = tuple(int(item) for item in volume_shape)
        if len(self.volume_shape) != 3 or any(item <= 0 for item in self.volume_shape):
            raise ValueError("B2 field scorer volume shape differs")
        cells = int(np.prod(self.volume_shape))
        if tuple(region_masks) != B2_ALL_REGIONS:
            raise ValueError("B2 field scorer region order differs")
        self.region_masks = {
            name: np.asarray(mask, dtype=bool) for name, mask in region_masks.items()
        }
        if any(mask.shape != (cells,) for mask in self.region_masks.values()):
            raise ValueError("B2 field scorer region shape differs")
        union = self.region_masks["eligible_union"]
        multiplicity = sum(
            self.region_masks[name].astype(np.int8) for name in B2_PRIMARY_REGIONS
        )
        if not np.array_equal(multiplicity == 1, union) or np.any(multiplicity > 1):
            raise ValueError("B2 scorer primary regions do not partition union")
        if any(np.any(mask & ~union) for mask in self.region_masks.values()):
            raise ValueError("B2 scorer region leaves eligible union")
        blocks = (
            (targets,)
            if validation_blocks is None
            else tuple(
                tuple(int(item) for item in block)
                for block in validation_blocks
            )
        )
        covered = tuple(item for block in blocks for item in block)
        if covered != targets or any(not block for block in blocks):
            raise ValueError("B2 field-scoring blocks do not partition targets")
        self.blocks = blocks
        self.block_index = {
            target: index for index, block in enumerate(blocks) for target in block
        }
        self.overall = {
            field: {
                region: FieldRegionAccumulator() for region in B2_ALL_REGIONS
            }
            for field in B2_FIELDS
        }
        self.block_union = [
            {field: FieldRegionAccumulator() for field in B2_FIELDS}
            for _ in blocks
        ]
        self.raw_phi_union = FieldRegionAccumulator()
        self.prefixes = {
            field: {
                members: PrefixFieldAccumulator(members)
                for members in B2_MEMBER_PREFIXES
            }
            for field in B2_FIELDS
        }
        self.per_target: list[dict[str, Any]] = []
        self.cursor = 0
        self.cell_indices = np.arange(cells, dtype=np.int64)

    def update(
        self,
        *,
        target_frame: int,
        standardized_forecast: np.ndarray,
        standardized_truth: np.ndarray,
    ) -> None:
        if self.cursor >= len(self.target_frames):
            raise ValueError("B2 field scorer received too many targets")
        expected = self.target_frames[self.cursor]
        if int(target_frame) != expected:
            raise ValueError(
                f"B2 field-scoring target {target_frame} differs from {expected}"
            )
        forecast = _finite_real("B2 standardized ensemble", standardized_forecast)
        truth = _finite_real("B2 standardized truth", standardized_truth)
        expected_forecast_shape = (32, 1, len(B2_FIELDS), *self.volume_shape)
        expected_truth_shape = (len(B2_FIELDS), *self.volume_shape)
        if (
            forecast.shape != expected_forecast_shape
            or truth.shape != expected_truth_shape
        ):
            raise ValueError("B2 field-scoring tensor shape differs")
        tie_seed = sampler_seed(self.model_seed, expected)
        block = self.block_index[expected]
        target_record: dict[str, Any] = {
            "target_frame": expected,
            "fields": {},
        }
        union = self.region_masks["eligible_union"]
        for channel, field in enumerate(B2_FIELDS):
            member_values = forecast[:, 0, channel]
            truth_values = truth[channel]
            if channel == B2_PHI_INDEX:
                raw_diagnostics = pointwise_ensemble_diagnostics(
                    member_values.reshape(32, -1),
                    truth_values.reshape(-1),
                    target_frame=expected,
                    channel_index=channel,
                    spatial_cell_index=self.cell_indices,
                    tie_seed=tie_seed,
                )
                self.raw_phi_union.update(
                    raw_diagnostics,
                    truth_values.reshape(-1),
                    union,
                )
                member_values, truth_values = gauge_fix_phi_channel(
                    member_values,
                    truth_values,
                )
            member_flat = member_values.reshape(32, -1)
            truth_flat = truth_values.reshape(-1)
            diagnostics = pointwise_ensemble_diagnostics(
                member_flat,
                truth_flat,
                target_frame=expected,
                channel_index=channel,
                spatial_cell_index=self.cell_indices,
                tie_seed=tie_seed,
            )
            for region in B2_ALL_REGIONS:
                self.overall[field][region].update(
                    diagnostics,
                    truth_flat,
                    self.region_masks[region],
                )
            self.block_union[block][field].update(diagnostics, truth_flat, union)
            for members in B2_MEMBER_PREFIXES:
                if members == 32:
                    self.prefixes[field][members].update_primary(
                        diagnostics,
                        union,
                    )
                else:
                    self.prefixes[field][members].update_raw(
                        member_flat[:members],
                        truth_flat,
                        union,
                    )
            target_record["fields"][field] = _target_sufficient_record(
                diagnostics,
                union,
            )
        self.per_target.append(target_record)
        self.cursor += 1

    def _bootstrap(self) -> dict[str, Any] | None:
        if self.target_frames != B2_VALIDATION_TARGETS:
            return None
        indices = moving_block_bootstrap_indices(
            126,
            block_length=21,
            replicates=2000,
            seed=85604032,
            blocks_per_replicate=6,
        )
        field_records: dict[str, Any] = {}
        aggregate_fair = np.zeros(indices.shape[0], dtype=np.float64)
        aggregate_squared = np.zeros(indices.shape[0], dtype=np.float64)
        for field in B2_FIELDS:
            records = [record["fields"][field] for record in self.per_target]
            count = np.asarray(
                [record["count"] for record in records], dtype=np.float64
            )
            squared = np.asarray(
                [record["squared_error_sum"] for record in records], dtype=np.float64
            )
            fair = np.asarray(
                [record["fair_crps_sum"] for record in records], dtype=np.float64
            )
            variance = np.asarray(
                [record["member_variance_sum"] for record in records],
                dtype=np.float64,
            )
            sampled_count = np.sum(count[indices], axis=1)
            sampled_squared = np.sum(squared[indices], axis=1)
            sampled_fair = np.sum(fair[indices], axis=1)
            sampled_variance = np.sum(variance[indices], axis=1)
            rmse = np.sqrt(sampled_squared / sampled_count)
            fair_score = sampled_fair / sampled_count
            corrected_spread = np.sqrt(
                (33 / 32) * sampled_variance / sampled_count
            )
            ratio = corrected_spread / rmse
            coverage = {}
            for name in B2_INTERVALS:
                counts = np.asarray(
                    [record["coverage_counts"][name] for record in records],
                    dtype=np.float64,
                )
                coverage[name] = _quantile_interval(
                    np.sum(counts[indices], axis=1) / sampled_count
                )
            field_records[field] = {
                "ensemble_mean_rmse": _quantile_interval(rmse),
                "fair_crps": _quantile_interval(fair_score),
                "corrected_spread_skill_ratio": _quantile_interval(ratio),
                "coverage": coverage,
            }
            aggregate_fair += fair_score / len(B2_FIELDS)
            aggregate_squared += rmse * rmse / len(B2_FIELDS)
        return {
            "method": "moving_block_bootstrap",
            "block_length_frames": 21,
            "replicates": 2000,
            "seed": 85604032,
            "conditional_on_single_85604_run": True,
            "voxel_count_used_as_independent_sample_size": False,
            "fields": field_records,
            "aggregate": {
                "equal_channel_ensemble_mean_rmse": _quantile_interval(
                    np.sqrt(aggregate_squared)
                ),
                "equal_channel_fair_crps": _quantile_interval(aggregate_fair),
            },
        }

    def finalize(self) -> dict[str, Any]:
        if self.cursor != len(self.target_frames):
            raise RuntimeError("B2 field scorer did not receive every target")
        regions: dict[str, Any] = {}
        for region in B2_ALL_REGIONS:
            field_records = {
                field: self.overall[field][region].finalize()
                for field in B2_FIELDS
            }
            regions[region] = {
                "fields": field_records,
                "aggregate": _aggregate_region_records(field_records),
            }
        blocks = []
        for block_targets, accumulators in zip(self.blocks, self.block_union):
            field_records = {
                field: accumulators[field].finalize() for field in B2_FIELDS
            }
            blocks.append(
                {
                    "target_frames": (
                        list(block_targets)
                        if self.sparse_targets
                        else [block_targets[0], block_targets[-1] + 1]
                    ),
                    **(
                        {"target_frames_are_explicit_indices": True}
                        if self.sparse_targets
                        else {}
                    ),
                    "fields": field_records,
                    "aggregate": _aggregate_region_records(field_records),
                }
            )
        prefix_records = {}
        for members in B2_MEMBER_PREFIXES:
            fields = {
                field: self.prefixes[field][members].finalize()
                for field in B2_FIELDS
            }
            prefix_records[f"M{members}"] = {
                "fields": fields,
                "aggregate": _aggregate_prefix_records(fields),
            }
        m16 = prefix_records["M16"]["aggregate"]
        m32 = prefix_records["M32"]["aggregate"]
        stability = {
            metric: monte_carlo_stability(m16[metric], m32[metric])
            for metric in (
                "equal_channel_ensemble_mean_rmse",
                "equal_channel_fair_crps",
                "equal_channel_ordinary_empirical_crps",
                "equal_channel_corrected_spread_skill_ratio",
            )
        }
        result = {
            "schema_version": 1,
            "scope": "B2_gauge_aware_field_and_marginal_calibration_85604",
            "model_seed": self.model_seed,
            "target_frames": (
                list(self.target_frames)
                if self.sparse_targets
                else [self.target_frames[0], self.target_frames[-1] + 1]
            ),
            "target_count": len(self.target_frames),
            "fields": list(B2_FIELDS),
            "potential_policy": (
                "subtract_full_spatial_mean_separately_per_member_and_truth_target"
            ),
            "primary_coordinates": "frozen_training_standardized_model_coordinates",
            "regions": regions,
            "chronological_blocks_eligible_union": blocks,
            "member_prefix_sensitivity_eligible_union": prefix_records,
            "M16_vs_M32_stability": stability,
            "raw_stored_gauge_phi_eligible_union_descriptive_only": (
                self.raw_phi_union.finalize()
            ),
            "per_target_eligible_union_sufficient_statistics": self.per_target,
            "conditional_uncertainty": self._bootstrap(),
            "held_out_85606_read": False,
            "physics_derived_training_loss_used": False,
        }
        if self.sparse_targets:
            result["target_frames_are_explicit_indices"] = True
        return result
