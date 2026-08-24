from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0" / "tools" / "inventory_nersc_85604_squash.py"


def load_namespace():
    namespace = {"__name__": "inventory_test"}
    exec(compile(TOOL.read_text(encoding="utf-8"), str(TOOL), "exec"), namespace)
    return namespace


def test_development_path_guard() -> None:
    namespace = load_namespace()
    accepted = namespace["verify_development_path"](Path("/tmp/85604/source.nc"))
    assert "85604" in str(accepted)
    with pytest.raises(ValueError, match="85606"):
        namespace["verify_development_path"](Path("/tmp/85606/source.nc"))
    with pytest.raises(ValueError, match="85604"):
        namespace["verify_development_path"](Path("/tmp/unlabeled/source.nc"))


def test_settings_parser_preserves_sections_and_values() -> None:
    namespace = load_namespace()
    parsed = namespace["parse_settings"](
        """timestep = 250  # comment
[Ne]
flux = 4.8e21  # provenance
source = flux * exp(-x)
[vorticity]
phi_boundary_timescale = 1e-5
"""
    )
    assert parsed[""]["timestep"] == "250"
    assert parsed["Ne"]["flux"] == "4.8e21"
    assert parsed["Ne"]["source"] == "flux * exp(-x)"
    assert parsed["vorticity"]["phi_boundary_timescale"] == "1e-5"


def test_streaming_summary_and_relative_step() -> None:
    namespace = load_namespace()
    summary = namespace["StreamingSummary"]()
    summary.update(np.asarray([1.0, 2.0, np.nan]))
    summary.update(np.asarray([3.0]))
    result = summary.finalize()
    assert result["total_count"] == 4
    assert result["finite_count"] == 3
    assert result["nonfinite_count"] == 1
    assert result["minimum"] == 1.0
    assert result["maximum"] == 3.0
    assert result["mean"] == 2.0
    assert result["rms"] == pytest.approx(np.sqrt(14.0 / 3.0))
    assert namespace["relative_step"](np.ones(4), np.full(4, 2.0)) == 1.0


def test_array_hash_includes_name_shape_and_dtype() -> None:
    namespace = load_namespace()
    values = np.arange(6, dtype=np.float64).reshape(2, 3)
    digest = namespace["array_sha256"]("Rxy", values)
    assert len(digest) == 64
    assert digest != namespace["array_sha256"]("Zxy", values)
    assert digest != namespace["array_sha256"]("Rxy", values.astype(np.float32))
