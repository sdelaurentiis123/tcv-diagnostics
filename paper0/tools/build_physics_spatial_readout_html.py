#!/usr/bin/env python3
"""Build the self-contained, figure-first Paper 0 physics readout."""

from __future__ import annotations

import argparse
import base64
import html
import json
from pathlib import Path


FORECAST_FIGURES = (
    ("matched_one_step_fields.png", "Matched one-step fields"),
    ("memberwise_particle_transport.png", "Memberwise particle transport"),
    ("particle_transport_variogram.png", "Transport coherence by separation"),
    ("ne_phi_coupling.png", "Density–potential coupling by mode"),
)
S0_FIGURES = (
    ("s0_spatial_reconstruction_hero.png", "Synthetic spatial reconstruction"),
    ("s0_spatial_skill_by_scale.png", "Spatial skill by scale and region"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--forecast-figure-dir", type=Path, required=True)
    parser.add_argument("--s0-figure-dir", type=Path, required=True)
    parser.add_argument("--s0-summary", type=Path, required=True)
    parser.add_argument("--forecast-summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    return parser.parse_args()


def data_uri(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(path)
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def strict_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"), parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)))


def fmt(value: float, digits: int = 3) -> str:
    return f"{float(value):.{digits}f}"


def build_document(*, figures: dict[str, str], s0: dict, forecast: dict, commit: str) -> str:
    heldout = s0["heldout_diagnostic_c"]
    band = s0["retained_power_ratio_n20_to_n35"]
    high = s0["retained_power_ratio_n_ge_40"]
    target = int(forecast["representative_target_frame"])
    cadence = float(forecast["shared_horizon_microseconds"])
    starts = int(forecast["shared_target_count"])
    report_status = html.escape(str(forecast.get("status", "complete")))
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Physics-first readout of old-85604 TCV/Hermes emulator forecasts and simultaneous synthetic diagnostic reconstruction.">
<title>Paper 0 · What the models learn and miss</title>
<style>
:root{--bg:#eef2f3;--paper:#fff;--ink:#17232d;--muted:#5d6a75;--rule:#dce3e7;--teal:#167d80;--blue:#3478a8;--gold:#a97422;--soft:#eaf4f4;--warn:#fbf4e5;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;--serif:"Iowan Old Style","Charter",Georgia,serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);font-size:18px;line-height:1.58}.page{max-width:76rem;margin:auto;background:var(--paper);min-height:100vh;padding:0 3rem 5rem;box-shadow:0 0 42px rgba(18,32,42,.08)}header{padding:4rem 0 2.3rem;border-bottom:1px solid var(--rule)}.eyebrow,.label{font-family:var(--sans);font-size:.73rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}h1{font-size:3.25rem;line-height:1.04;letter-spacing:-.035em;max-width:66rem;margin:.7rem 0 1rem}h2{font-size:2rem;line-height:1.16;letter-spacing:-.015em;margin:.3rem 0 .8rem}.dek,.lede{color:var(--muted);max-width:58rem}.dek{font-size:1.18rem}.meta{display:flex;gap:.55rem 1.25rem;flex-wrap:wrap;margin-top:1.2rem;font: .78rem var(--sans);color:var(--muted)}nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);font:.78rem var(--sans);overflow:auto}nav div{display:flex;gap:1.3rem;padding:.9rem 0;white-space:nowrap}nav a{color:var(--muted);text-decoration:none}section{padding:3.2rem 0;border-bottom:1px solid var(--rule);scroll-margin-top:54px}.answer{background:var(--soft);border-left:4px solid var(--teal);padding:1.2rem 1.4rem;margin:1.4rem 0}.answer p{margin:.25rem 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-top:1.5rem}.card{border-top:3px solid var(--rule);padding:.85rem .9rem;background:#fafbfb}.card strong{display:block;font:700 1.35rem var(--sans);color:var(--teal);margin-bottom:.25rem}.card span{font:.82rem/1.4 var(--sans);color:var(--muted)}figure{margin:2rem 0 0;border:1px solid var(--rule);border-radius:8px;overflow:hidden;background:#fff;box-shadow:0 9px 28px -24px #111}.figure-image{padding:10px;background:#fff}.figure-image img{display:block;width:100%;height:auto}figcaption{border-top:1px solid var(--rule);padding:1rem 1.1rem 1.1rem;font:.83rem/1.52 var(--sans);color:var(--muted)}.fignum{font-weight:750;color:var(--teal)}.definition{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:1.2rem 0}.definition div{border-left:2px solid var(--rule);padding-left:.8rem}.definition dt{font:700 .82rem var(--sans);color:var(--ink)}.definition dd{margin:.2rem 0 0;font:.78rem/1.45 var(--sans);color:var(--muted)}table{border-collapse:collapse;width:100%;font:.83rem/1.45 var(--sans);margin-top:1.2rem}th,td{text-align:left;vertical-align:top;padding:.68rem .55rem;border-bottom:1px solid var(--rule)}th{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}.callout{border-left:4px solid var(--gold);background:var(--warn);padding:1rem 1.2rem;margin-top:1.4rem}.small{font:.78rem/1.5 var(--sans);color:var(--muted)}footer{padding:2rem 0;color:var(--muted);font:.75rem/1.5 var(--sans)}@media(max-width:900px){.page{padding:0 1.2rem 3rem}.cards,.definition{grid-template-columns:1fr 1fr}h1{font-size:2.55rem}}@media(max-width:570px){body{font-size:16px}.cards,.definition{grid-template-columns:1fr}h1{font-size:2.15rem}}@media print{body{background:#fff;font-size:10pt}.page{max-width:none;padding:0;box-shadow:none}nav{display:none}section{padding:1.3rem 0}figure{break-inside:avoid;box-shadow:none}}
</style>
</head>
<body><div class="page">
<header><p class="eyebrow">Paper 0 · physics readout · 28 August 2026</p><h1>What the models learn—and what they still miss</h1><p class="dek">Four models are compared on the same 85604 plasma state, time, geometry, and color scale. A separate test asks whether two local density measurements can predict an unseen region.</p><div class="meta"><span>85604 development evidence only</span><span>85606 remains unopened</span><span>target frame __TARGET__</span><span>one-step horizon __CADENCE__ µs</span><span>snapshot __COMMIT__</span></div></header>
<nav><div><a href="#answer">Answer</a><a href="#fields">Fields</a><a href="#transport">Transport</a><a href="#scale">Coherence scale</a><a href="#coupling">Density–potential coupling</a><a href="#spatial">Spatial diagnostics</a><a href="#meaning">Meaning</a></div></nav>
<main>
<section id="answer"><p class="label">01 · Short answer</p><h2>The models learn the average future better than the uncertainty around it</h2><div class="answer"><p><strong>Forecast:</strong> one step ahead, the models retain much of the large density and potential pattern. Their transport fluctuations still do not add together correctly across the plasma.</p><p><strong>Spatial reconstruction:</strong> two local density measurements predict a separate unseen patch with correlation <strong>__C_CORR__</strong> and <strong>__C_SKILL__%</strong> lower error than predicting no fluctuation.</p></div><div class="cards"><div class="card"><strong>4 models</strong><span>one fixed forecast and three probabilistic forecasts</span></div><div class="card"><strong>__STARTS__ starts</strong><span>the same one-step cases for every curve</span></div><div class="card"><strong>__BAND_RANGE__%</strong><span>recovered power at physical modes n=20–35</span></div><div class="card"><strong>~__HIGH__%</strong><span>recovered power at n≥40</span></div></div></section>

<section id="fields"><p class="label">02 · Fields</p><h2>One step ahead: the large pattern is learnable</h2><p class="lede">Frame __TARGET__ is the median one-step transport-error case, chosen automatically. It was not selected for appearance.</p><figure><div class="figure-image"><img alt="Matched truth forecast and error for density and potential across four models" src="__FIG_FIELDS__"></div><figcaption><span class="fignum">Figure 1.</span> Density and potential fluctuations at toroidal plane z=44. Every row uses the same truth scale. Grey: simulation boundary. Black: separatrix. Red: X-point. Blue: outboard midplane. Probabilistic forecasts show the mean of 32 members.</figcaption></figure></section>

<section id="transport"><p class="label">03 · Particle transport</p><h2>Locally plausible transport can still add up incorrectly</h2><div class="definition"><div><dt>Top row</dt><dd>Outward particle transport at each point along the separatrix.</dd></div><div><dt>Bottom row</dt><dd>The running sum. It shows whether positive and negative patches cancel correctly.</dd></div><div><dt>Members first</dt><dd>Transport is calculated for each possible future before averaging.</dd></div></div><figure><div class="figure-image"><img alt="Truth and forecast memberwise particle transport around and cumulatively along the separatrix" src="__FIG_TRANSPORT__"></div><figcaption><span class="fignum">Figure 2.</span> Black: truth. Thin colored lines: 16 forecast members. Thick colored line: ensemble mean. Shading: central 90%. The key failure appears in the running sum: the models do not yet coordinate distant transport patches correctly.</figcaption></figure></section>

<section id="scale"><p class="label">04 · Spatial scale</p><h2>Where does transport stop moving together?</h2><p class="lede">The variogram compares transport at two points as their physical separation grows. Too high means the forecast loses spatial agreement too quickly; too low means it is too smooth or too coherent.</p><figure><div class="figure-image"><img alt="Truth and forecast particle transport variogram versus physical toroidal separation" src="__FIG_VARIOGRAM__"></div><figcaption><span class="fignum">Figure 3.</span> Left: all four models after one frame. Right: the persistent global–local model after four frames, the only stored four-frame ensemble in this comparison. Bands show uncertainty across chronological 85604 blocks, not across independent shots.</figcaption></figure></section>

<section id="coupling"><p class="label">05 · Density and potential</p><h2>Transport depends on these fields moving together</h2><p class="lede">The domain covers one fifth of the torus, so stored index k means physical mode n=5k. The blue band marks n=20–35. Separate field accuracy is not enough if density and potential have the wrong phase or coherence.</p><figure><div class="figure-image"><img alt="Density potential cross spectrum phase and coherence versus physical toroidal mode" src="__FIG_COUPLING__"></div><figcaption><span class="fignum">Figure 4.</span> Black: truth. Cross-spectrum is shared power, phase is relative position, and coherence is how consistently the fields move together. Each probabilistic curve is calculated member by member before averaging.</figcaption></figure></section>

<section id="spatial"><p class="label">06 · Sparse measurements</p><h2>Two local density views predict some unseen structure</h2><p class="lede">Measurements A and B are taken at the same instant. The model reconstructs the full fluctuation field and predicts a separate region C. This test has no time forecast.</p><figure><div class="figure-image"><img alt="Truth observed diagnostics reconstruction error and heldout channels for simultaneous spatial reconstruction" src="__FIG_S0_HERO__"></div><figcaption><span class="fignum">Figure 5.</span> A simple linear reconstruction is fit on training frames only. The displayed frame has median error in the unseen region C. A, B, and C are BES/GPI-like box averages, not complete experimental instrument models.</figcaption></figure><figure><div class="figure-image"><img alt="Spatial reconstruction skill versus distance toroidal mode and plasma region" src="__FIG_S0_SCALE__"></div><figcaption><span class="fignum">Figure 6.</span> The unseen region is predicted better than zero fluctuation, including at large separation. The reconstruction is band-limited: retained power at n=20–35 is __BAND_DETAIL__; at n≥40 it is only __HIGH_DETAIL__.</figcaption></figure></section>

<section id="meaning"><p class="label">07 · Bottom line</p><h2>What these results do—and do not—show</h2><table><thead><tr><th>We can say</th><th>We cannot say yet</th></tr></thead><tbody><tr><td>The main one-step density and potential pattern is learnable.</td><td>No forecast ensemble yet has reliable total transport uncertainty.</td></tr><tr><td>We can locate transport errors by position, distance, and physical mode.</td><td>The earlier ETKF diagnostic ranking is not yet definitive.</td></tr><tr><td>Two sparse density views contain useful nonlocal low/mid-mode information.</td><td>The spatial test is not a complete BES/GPI instrument model or cross-shot test.</td></tr><tr><td>The simple spatial reconstruction misses most n≥40 structure.</td><td>85606, assimilation, sensor ranking, and steering are not evaluated here.</td></tr></tbody></table><div class="callout"><strong>Next decision:</strong> confirm the transport surface and synthetic measurement locations with Ben. The variogram-loss training campaign is paused; no training arm is active.</div><p class="small">Forecast extraction: __REPORT_STATUS__. Forecast figures use seed 1702, target __TARGET__, and fixed 32-member seeds. Spatial split: train [0,432), guard [432,496), validation [496,624).</p></section>
</main><footer>Self-contained HTML: all six raster figures are embedded. Source data are old TCV/Hermes simulation 85604 development artifacts only. The report builder neither opens nor enumerates simulation 85606.</footer>
</div></body></html>'''
    replacements = {
        "__TARGET__": str(target),
        "__CADENCE__": fmt(cadence, 4),
        "__COMMIT__": html.escape(commit[:12]),
        "__STARTS__": str(starts),
        "__C_CORR__": fmt(heldout["pearson_correlation"]),
        "__C_SKILL__": fmt(100.0 * float(heldout["relative_mse_skill_vs_zero"]), 1),
        "__BAND_RANGE__": f"{100 * min(band.values()):.0f}–{100 * max(band.values()):.0f}",
        "__HIGH__": fmt(100.0 * sum(high.values()) / len(high), 0),
        "__BAND_DETAIL__": ", ".join(f"{name} {100 * float(value):.0f}%" for name, value in band.items()),
        "__HIGH_DETAIL__": ", ".join(f"{name} {100 * float(value):.0f}%" for name, value in high.items()),
        "__REPORT_STATUS__": report_status,
        "__FIG_FIELDS__": figures["matched_one_step_fields.png"],
        "__FIG_TRANSPORT__": figures["memberwise_particle_transport.png"],
        "__FIG_VARIOGRAM__": figures["particle_transport_variogram.png"],
        "__FIG_COUPLING__": figures["ne_phi_coupling.png"],
        "__FIG_S0_HERO__": figures["s0_spatial_reconstruction_hero.png"],
        "__FIG_S0_SCALE__": figures["s0_spatial_skill_by_scale.png"],
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    if "__" in template:
        unresolved = sorted({piece.split("__", 1)[0] for piece in template.split("__")[1::2]})
        raise ValueError(f"unresolved HTML template values: {unresolved}")
    return template


def main() -> None:
    args = parse_args()
    s0 = strict_json(args.s0_summary)
    forecast = strict_json(args.forecast_summary)
    if s0.get("development_run") != "85604" or s0.get("held_out_85606_read") is not False:
        raise ValueError("S0 summary scope differs")
    if forecast.get("development_run") != "85604" or forecast.get("held_out_85606_read") is not False:
        raise ValueError("forecast summary scope differs")
    figures: dict[str, str] = {}
    for name, _ in FORECAST_FIGURES:
        figures[name] = data_uri(args.forecast_figure_dir / name)
    for name, _ in S0_FIGURES:
        figures[name] = data_uri(args.s0_figure_dir / name)
    document = build_document(figures=figures, s0=s0, forecast=forecast, commit=args.commit)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(document, encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "bytes": args.output.stat().st_size}, sort_keys=True))


if __name__ == "__main__":
    main()
