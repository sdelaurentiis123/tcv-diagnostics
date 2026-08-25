"""Integrity checks for the tracked old-85604 bounded-rollout report."""

from __future__ import annotations

import csv
import json
from html.parser import HTMLParser
from pathlib import Path

from paper0.tools.build_old_85604_bounded_rollout_report import method_order


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "paper0/reports/paper0-old-85604-bounded-rollout-2026-08-25.html"
FIGURE_DIR = ROOT / "paper0/figures/post_ecrd_old_85604_bounded_rollout"
RESULT_DIR = ROOT / "paper0/results"


class _ReportParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.images: list[str] = []
        self.text: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "img":
            values = dict(attrs)
            assert values.get("alt")
            assert values.get("src")
            self.images.append(str(values["src"]))

    def handle_data(self, data: str) -> None:
        self.text.append(data)


def _csv_rows(name: str) -> list[dict[str, str]]:
    with (RESULT_DIR / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_frozen_method_inventory() -> None:
    assert method_order(4) == [
        "persistence",
        "autoregressive_lead1",
        "autoregressive_lead2",
        "direct",
    ]
    assert method_order(8) == [
        "persistence",
        "autoregressive_lead1",
        "autoregressive_lead2",
        "autoregressive_lead4",
        "direct",
    ]


def test_compact_evidence_row_counts_and_scope() -> None:
    assert len(_csv_rows("post_ecrd_old_85604_bounded_rollout_state_6937051.csv")) == 21
    assert len(_csv_rows("post_ecrd_old_85604_bounded_rollout_spectra_6937203.csv")) == 345
    assert len(_csv_rows("post_ecrd_old_85604_bounded_rollout_cross_field_6937203.csv")) == 207
    assert len(_csv_rows("post_ecrd_old_85604_bounded_rollout_transport_6937203.csv")) == 184
    summary = json.loads(
        (RESULT_DIR / "post_ecrd_old_85604_bounded_rollout_summary_6937203.json").read_text(
            encoding="utf-8"
        )
    )
    assert summary["development_run"] == "85604"
    assert summary["held_out_85606_read"] is False
    assert summary["new_nersc_data_read"] is False
    assert summary["guard_frames_read"] is False
    assert summary["physics_derived_loss_used"] is False
    assert summary["zperiod"] == 5
    assert summary["mode_mapping"] == "n=5k"


def test_html_references_all_labeled_figures_and_states_field_level_exception() -> None:
    source = REPORT.read_text(encoding="utf-8")
    parser = _ReportParser()
    parser.feed(source)
    assert len(parser.images) == 10
    assert len(set(parser.images)) == 10
    for src in parser.images:
        assert src.startswith("../figures/post_ecrd_old_85604_bounded_rollout/")
        assert (REPORT.parent / src).resolve().is_file()
    text = " ".join(" ".join(parser.text).split())
    assert "seed 1701’s eight-frame repeated one-frame Pe and Pi skills are −0.021 and −0.025" in text
    assert "Every seed and every field has positive persistence-relative skill" not in text
    figure_text = "\n".join(
        path.read_text(encoding="utf-8") for path in sorted(FIGURE_DIR.glob("*.svg"))
    )
    for label in ("persistence", "1-frame steps", "2-frame steps", "4-frame steps", "direct terminal"):
        assert label in figure_text.lower()


def test_every_svg_has_title_and_description() -> None:
    svgs = sorted(FIGURE_DIR.glob("*.svg"))
    assert len(svgs) == 10
    for path in svgs:
        source = path.read_text(encoding="utf-8")
        assert "<dc:title>" in source
        assert "<dc:description>" in source
