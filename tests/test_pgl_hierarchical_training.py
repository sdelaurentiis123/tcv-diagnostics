from __future__ import annotations

import torch
from torch import nn

from tcv_diagnostics.models.persistent_global_local import PersistentGlobalLocalEDM
from tcv_diagnostics.pgl_hierarchical_training import (
    HierarchicalControlMagnitudes,
    HierarchicalTerms,
    HierarchicalTrainingConfig,
    hierarchical_objective,
    loss_gradient_audit,
    parameter_branches,
)
from tcv_diagnostics.pgl_hierarchical_transport import HierarchicalTransportScores


def _controls() -> HierarchicalControlMagnitudes:
    values = (1.0, 1.0, 1.0, 1.0)
    return HierarchicalControlMagnitudes(
        local_spatial=values,
        local_temporal=values,
        regional=values,
        fourier_low=values,
        fourier_transport_band=values,
        global_crps=values,
    )


def _scores(value: float = 1.0) -> HierarchicalTransportScores:
    group = tuple(torch.tensor(value) for _ in range(4))
    return HierarchicalTransportScores(
        local_spatial=group,
        local_temporal=group,
        regional=group,
        fourier_low=group,
        fourier_transport_band=group,
        global_crps=group,
        ordinary={},
    )


def test_config_freezes_two_epoch_and_smoke_budgets() -> None:
    screen = HierarchicalTrainingConfig(mode="screen", arm="TRANSPORT")
    assert screen.optimizer_updates == 428
    assert screen.training_windows == 856
    assert screen.checkpoints == (107, 214, 428)
    assert screen.mean_learning_rate == screen.stochastic_learning_rate / 10.0
    assert screen.physics_derived_training_loss_used

    smoke = HierarchicalTrainingConfig(mode="smoke", arm="CONTROL")
    assert smoke.optimizer_updates == 1
    assert smoke.training_windows == 2
    assert smoke.checkpoints == (1,)
    assert not smoke.physics_derived_training_loss_used


def test_hierarchical_objective_keeps_original_and_adds_all_scales() -> None:
    terms = HierarchicalTerms(mean=torch.tensor(2.0), edm=torch.tensor(3.0), scores=_scores())
    control, local, regional, global_score = hierarchical_objective(
        arm="CONTROL", terms=terms, controls=_controls(), auxiliary_lambda=0.25
    )
    treatment, _, _, _ = hierarchical_objective(
        arm="TRANSPORT", terms=terms, controls=_controls(), auxiliary_lambda=0.25
    )
    assert control.item() == 5.0
    assert local.item() == 1.0
    assert regional.item() == 1.0
    assert global_score.item() == 1.0
    assert treatment.item() == 5.75


def test_parameter_branches_are_complete_and_disjoint() -> None:
    mean = nn.Sequential(nn.Conv3d(5, 5, 1), nn.SiLU(), nn.Conv3d(5, 5, 1))
    edm = PersistentGlobalLocalEDM(residual_scales=torch.ones((4, 5)))
    branches = parameter_branches(mean, edm)
    assert tuple(branches) == (
        "mean",
        "stochastic_global",
        "stochastic_local_encoder",
        "stochastic_local_decoder",
    )
    identifiers = [id(value) for rows in branches.values() for _, value in rows]
    assert len(identifiers) == len(set(identifiers))
    assert set(identifiers) == {
        id(value) for value in list(mean.parameters()) + list(edm.parameters())
    }


def test_gradient_audit_reports_branch_norms_and_cosines() -> None:
    first = nn.Parameter(torch.tensor([1.0, 2.0]))
    second = nn.Parameter(torch.tensor([3.0]))
    branches = {
        "first": [("first", first)],
        "second": [("second", second)],
    }
    loss_a = first.square().sum() + second.square().sum()
    loss_b = 2.0 * loss_a
    audit = loss_gradient_audit(
        {"a": loss_a, "b": loss_b}, branches, retain_graph=False
    )
    assert audit["losses"]["a"]["total_gradient_norm"] > 0.0
    assert audit["losses"]["a"]["branches"]["first"]["gradient_norm"] > 0.0
    assert audit["losses"]["a"]["branches"]["second"]["gradient_norm"] > 0.0
    assert abs(audit["cosine_similarity"]["a__b"] - 1.0) < 1.0e-7
