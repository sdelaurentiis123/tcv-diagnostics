"""CPU contract tests for matched ECRD training utilities."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tcv_diagnostics.ecrd_training import (
    ECRD_ARMS,
    ECRD_REFERENCE_PARAMETER_COUNT,
    ECRDTrainingConfig,
    _loss,
    _training_order,
    build_model,
    exact_model_config,
    frozen_parameter_counts,
    validate_parameter_matching,
)
from tcv_diagnostics.models.ecrd import ECRDTransition
from tcv_diagnostics.models.field_residual_edm import JointFieldResidualEDM


def test_full_budget_and_seed_contract() -> None:
    config = ECRDTrainingConfig(arm="ECRD", seed=1702)
    assert config.epochs == 100
    assert config.train_targets == tuple(range(2, 432))
    assert config.validation_targets == tuple(range(498, 624))
    assert config.total_optimizer_steps == 10_800
    assert config.target_presentations == 43_000
    assert config.validation_epochs == tuple(range(5, 101, 5))
    assert config.to_record()["physics_derived_loss_allowed"] is False
    with pytest.raises(ValueError, match="seed"):
        ECRDTrainingConfig(arm="ECRD", seed=99)


def test_smoke_budget_is_bounded() -> None:
    config = ECRDTrainingConfig(arm="B5-Context", seed=1701, mode="smoke")
    assert config.epochs == 1
    assert config.train_targets == (2, 3, 4, 5)
    assert config.validation_targets == (498, 499, 500, 501)
    assert config.total_optimizer_steps == 2
    np.testing.assert_array_equal(_training_order(config), [[2, 3, 4, 5]])


def test_exact_arm_configs_and_parameter_matching() -> None:
    counts = frozen_parameter_counts()
    assert tuple(counts) == ECRD_ARMS
    assert counts["B5"] == ECRD_REFERENCE_PARAMETER_COUNT
    assert counts == {
        "B5": 11_604_709,
        "B5-Context": 11_350_909,
        "ECRD": 11_455_746,
        "ECRD-History": 11_462_766,
    }
    validate_parameter_matching(counts)
    assert exact_model_config("ECRD").downsample_stride == (2, 2, 1)
    assert exact_model_config("ECRD-History").condition_channels == 15


@pytest.mark.parametrize("arm", ECRD_ARMS)
def test_model_type_and_field_only_loss(arm: str) -> None:
    model = build_model(arm)
    if arm == "B5":
        assert isinstance(model, JointFieldResidualEDM)
        condition_channels = 10
    else:
        assert isinstance(model, ECRDTransition)
        condition_channels = model.config.condition_channels
    # Use the exact model on a small divisible volume; one forward pass checks
    # the shared loss adapter without allocating a production data tensor.
    target = torch.randn(1, 5, 8, 8, 16)
    condition = torch.randn(1, condition_channels, 8, 8, 16)
    values = _loss(
        model,
        arm=arm,
        target=target,
        condition=condition,
        sigma=torch.tensor([0.5]),
        noise=torch.randn_like(target),
    )
    assert set(values) == {
        "objective",
        "edm_loss",
        "unweighted_edm_mse",
        "mean_mse",
    }
    assert all(torch.isfinite(value) for value in values.values())
    if arm in ("B5", "B5-Context"):
        assert values["mean_mse"].item() == 0.0
