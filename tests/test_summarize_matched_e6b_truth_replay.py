"""Known-answer tests for the all-frame E6B truth-replay summary."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import tempfile


ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "paper0/tools/summarize_matched_e6b_truth_replay.py"
SPEC = importlib.util.spec_from_file_location("summarize_e6b_truth_replay", TOOL)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_summary_requires_exact_ordered_all_frame_coverage() -> None:
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        sources = []
        records = []
        paths = []
        for index in range(8):
            start = index * 78
            source_path = root / f"source_{index}.h5"
            source_path.write_bytes(b"source")
            sources.append(
                {
                    "canonical_file": str(source_path),
                    "canonical_file_sha256": "source-hash",
                }
            )
            record = {
                "scope": "phase2_matched_e6b_elliptic_output",
                "status": "completed",
                "development_run": "85604",
                "held_out_85606_read": False,
                "paper0_commit": "commit",
                "truth_layout": True,
                "frame_interval": [start, start + 78],
                "frame_count": 78,
                "source_input": {
                    "path": str(source_path),
                    "sha256": "source-hash",
                },
                "derived_phi": {"path": "phi", "sha256": "phi-hash"},
                "truth_replay_gate": {
                    "all_frames_passed": True,
                    "maximum_absolute_difference": 1.0e-12,
                    "maximum_relative_l2": 2.0e-13,
                    "per_frame": [
                        {"frame_index": frame, "passes": True}
                        for frame in range(start, start + 78)
                    ],
                },
            }
            path = root / f"result_{index}.json"
            path.write_text(json.dumps(record), encoding="utf-8")
            records.append(record)
            paths.append(path)
        native = {
            "development_run": "85604",
            "held_out_85606_read": False,
            "decision": {"all_frame_bidirectional_closure_validated": True},
            "ordered_gates": {"G0": True},
            "extraction": {"canonical_shards": sources},
        }
        result = MODULE.summarize_truth_replay(
            records=records,
            record_paths=paths,
            native_truth=native,
            paper0_commit="commit",
            slurm_job_id="job",
        )
        assert result["status"] == "pass"
        assert result["frame_count"] == 624
        assert result["coverage"] == [0, 624]

        records[3]["frame_interval"] = [1, 79]
        try:
            MODULE.summarize_truth_replay(
                records=records,
                record_paths=paths,
                native_truth=native,
                paper0_commit="commit",
                slurm_job_id="job",
            )
        except ValueError as error:
            assert "shard 3 identity" in str(error)
        else:
            raise AssertionError("misordered replay shard was accepted")
