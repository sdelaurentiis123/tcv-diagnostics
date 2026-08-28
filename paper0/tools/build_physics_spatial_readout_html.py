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
    cadence = float(forecast["cadence_microseconds"])
    starts = int(forecast["shared_one_step_target_count"])
    report_status = html.escape(str(forecast.get("status", "complete")))
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Physics-first readout of old-85604 TCV/Hermes emulator forecasts and simultaneous synthetic diagnostic reconstruction.">
<title>Paper 0 · Physics and spatial diagnostic readout</title>
<style>
:root{--bg:#eef2f3;--paper:#fff;--ink:#17232d;--muted:#5d6a75;--rule:#dce3e7;--teal:#167d80;--blue:#3478a8;--gold:#a97422;--soft:#eaf4f4;--warn:#fbf4e5;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;--serif:"Iowan Old Style","Charter",Georgia,serif}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--bg);color:var(--ink);font-family:var(--serif);font-size:18px;line-height:1.58}.page{max-width:76rem;margin:auto;background:var(--paper);min-height:100vh;padding:0 3rem 5rem;box-shadow:0 0 42px rgba(18,32,42,.08)}header{padding:4rem 0 2.3rem;border-bottom:1px solid var(--rule)}.eyebrow,.label{font-family:var(--sans);font-size:.73rem;font-weight:750;letter-spacing:.12em;text-transform:uppercase;color:var(--teal)}h1{font-size:3.25rem;line-height:1.04;letter-spacing:-.035em;max-width:66rem;margin:.7rem 0 1rem}h2{font-size:2rem;line-height:1.16;letter-spacing:-.015em;margin:.3rem 0 .8rem}.dek,.lede{color:var(--muted);max-width:58rem}.dek{font-size:1.18rem}.meta{display:flex;gap:.55rem 1.25rem;flex-wrap:wrap;margin-top:1.2rem;font: .78rem var(--sans);color:var(--muted)}nav{position:sticky;top:0;z-index:10;background:rgba(255,255,255,.96);backdrop-filter:blur(8px);border-bottom:1px solid var(--rule);font:.78rem var(--sans);overflow:auto}nav div{display:flex;gap:1.3rem;padding:.9rem 0;white-space:nowrap}nav a{color:var(--muted);text-decoration:none}section{padding:3.2rem 0;border-bottom:1px solid var(--rule);scroll-margin-top:54px}.answer{background:var(--soft);border-left:4px solid var(--teal);padding:1.2rem 1.4rem;margin:1.4rem 0}.answer p{margin:.25rem 0}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:.8rem;margin-top:1.5rem}.card{border-top:3px solid var(--rule);padding:.85rem .9rem;background:#fafbfb}.card strong{display:block;font:700 1.35rem var(--sans);color:var(--teal);margin-bottom:.25rem}.card span{font:.82rem/1.4 var(--sans);color:var(--muted)}figure{margin:2rem 0 0;border:1px solid var(--rule);border-radius:8px;overflow:hidden;background:#fff;box-shadow:0 9px 28px -24px #111}.figure-image{padding:10px;background:#fff}.figure-image img{display:block;width:100%;height:auto}figcaption{border-top:1px solid var(--rule);padding:1rem 1.1rem 1.1rem;font:.83rem/1.52 var(--sans);color:var(--muted)}.fignum{font-weight:750;color:var(--teal)}.definition{display:grid;grid-template-columns:repeat(3,1fr);gap:.8rem;margin:1.2rem 0}.definition div{border-left:2px solid var(--rule);padding-left:.8rem}.definition dt{font:700 .82rem var(--sans);color:var(--ink)}.definition dd{margin:.2rem 0 0;font:.78rem/1.45 var(--sans);color:var(--muted)}table{border-collapse:collapse;width:100%;font:.83rem/1.45 var(--sans);margin-top:1.2rem}th,td{text-align:left;vertical-align:top;padding:.68rem .55rem;border-bottom:1px solid var(--rule)}th{color:var(--muted);font-size:.72rem;text-transform:uppercase;letter-spacing:.06em}.callout{border-left:4px solid var(--gold);background:var(--warn);padding:1rem 1.2rem;margin-top:1.4rem}.small{font:.78rem/1.5 var(--sans);color:var(--muted)}footer{padding:2rem 0;color:var(--muted);font:.75rem/1.5 var(--sans)}@media(max-width:900px){.page{padding:0 1.2rem 3rem}.cards,.definition{grid-template-columns:1fr 1fr}h1{font-size:2.55rem}}@media(max-width:570px){body{font-size:16px}.cards,.definition{grid-template-columns:1fr}h1{font-size:2.15rem}}@media print{body{background:#fff;font-size:10pt}.page{max-width:none;padding:0;box-shadow:none}nav{display:none}section{padding:1.3rem 0}figure{break-inside:avoid;box-shadow:none}}
</style>
</head>
<body><div class="page">
<header><p class="eyebrow">Paper 0 · physics-first readout · 28 August 2026</p><h1>What the emulator can forecast—and where transport coherence is lost</h1><p class="dek">A concise comparison of four old-85604 forecast models, followed by the first simultaneous leave-one-diagnostic-out reconstruction. Every model panel uses the same seed, target, horizon, geometry, and truth scale.</p><div class="meta"><span>85604 development evidence only</span><span>85606 remains unopened</span><span>representative target __TARGET__</span><span>one-step horizon __CADENCE__ µs</span><span>snapshot __COMMIT__</span></div></header>
<nav><div><a href="#answer">Answer</a><a href="#fields">Fields</a><a href="#transport">Transport</a><a href="#scale">Coherence scale</a><a href="#coupling">Density–potential coupling</a><a href="#spatial">Spatial diagnostics</a><a href="#meaning">Meaning</a></div></nav>
<main>
<section id="answer"><p class="label">01 · One-minute answer</p><h2>The mean future is becoming credible; the ensemble covariance is not yet transport-faithful</h2><div class="answer"><p><strong>Forecasting:</strong> the models preserve part of the visible large-scale density and potential structure over one saved step. The transport figures expose errors that field images and RMSE can hide.</p><p><strong>Spatial inference:</strong> two localized simultaneous density measurements predict a separated held-out patch with correlation <strong>__C_CORR__</strong> and reduce its MSE by <strong>__C_SKILL__%</strong> relative to predicting no fluctuation.</p></div><div class="cards"><div class="card"><strong>4 models</strong><span>one deterministic operator and three joint stochastic generators</span></div><div class="card"><strong>__STARTS__ starts</strong><span>same chronological one-step population for model curves</span></div><div class="card"><strong>__BAND_RANGE__%</strong><span>power retained by linear spatial reconstruction in physical n=20–35</span></div><div class="card"><strong>~__HIGH__%</strong><span>power retained at n≥40: high-mode structure is mostly unrecoverable here</span></div></div></section>

<section id="fields"><p class="label">02 · Forecasted plasma fields</p><h2>The same truth, plane, and color scale across all models</h2><p class="lede">This is target frame __TARGET__, selected mechanically as the lower median one-step particle-transport error of the conditioned stochastic forecast—not because it looked attractive. Each row shows the truth again so the comparison is readable without scanning across the page.</p><figure><div class="figure-image"><img alt="Matched truth forecast and error for density and potential across four models" src="__FIG_FIELDS__"></div><figcaption><span class="fignum">Figure 1.</span> Electron-density and potential fluctuations at stored toroidal plane z=44. The grey line is the strict simulation boundary, black is the confined separatrix, red crosses mark the X-point stencil, and the blue circle marks the outboard-midplane separatrix location. Probabilistic panels show the 32-member field mean; the deterministic model has one forecast.</figcaption></figure></section>

<section id="transport"><p class="label">03 · The nonlinear quantity that matters</p><h2>Local particle transport and its cumulative cancellation</h2><div class="definition"><div><dt>Local contribution</dt><dd>Outward particle transport at each poloidal location on the confined separatrix, already integrated around the simulated toroidal wedge.</dd></div><div><dt>Cumulative contribution</dt><dd>The running signed sum along the separatrix. It reveals whether positive and negative patches cancel in the right way.</dd></div><div><dt>Memberwise evaluation</dt><dd>Transport is calculated separately for every generated future before any ensemble average. This preserves nonlinear field coupling.</dd></div></div><figure><div class="figure-image"><img alt="Truth and forecast memberwise particle transport around and cumulatively along the separatrix" src="__FIG_TRANSPORT__"></div><figcaption><span class="fignum">Figure 2.</span> Black is truth. Colored thin curves are the first 16 frozen ensemble members, the colored thick curve is their mean, and the translucent region is the central 90% member envelope. The lower panels make spatial cancellation visible: plausible local amplitudes can still sum to the wrong global transport.</figcaption></figure></section>

<section id="scale"><p class="label">04 · At what scale coherence is lost</p><h2>The transport variogram measures disagreement versus physical separation</h2><p class="lede">For two locations separated around the torus, the curve measures their mean absolute transport difference. A model that rises too quickly decorrelates patches too fast; one that stays too low makes the transport field unrealistically smooth or coherent.</p><figure><div class="figure-image"><img alt="Truth and forecast particle transport variogram versus physical toroidal separation" src="__FIG_VARIOGRAM__"></div><figcaption><span class="fignum">Figure 3.</span> One-frame comparison uses all four stochastic/deterministic families available at that horizon. The four-frame panel is shown only where a genuine stored forecast exists. Bands are chronological moving-block bootstrap intervals over the same ordered 85604 starts; they are not cross-shot confidence intervals.</figcaption></figure></section>

<section id="coupling"><p class="label">05 · Why density and potential must be judged together</p><h2>Cross-spectrum, phase, and coherence by physical toroidal mode</h2><p class="lede">The stored domain is one fifth of the torus, so stored Fourier index k maps to physical mode n=5k. The shaded n=20–35 band is the preregistered transport-relevant band. Matching density and potential separately is insufficient if their relative phase or coherence is wrong.</p><figure><div class="figure-image"><img alt="Density potential cross spectrum phase and coherence versus physical toroidal mode" src="__FIG_COUPLING__"></div><figcaption><span class="fignum">Figure 4.</span> Truth is black. Each stochastic curve is the ensemble expectation of memberwise coupling—not coupling computed from the ensemble-mean fields. The three panels separate amplitude, relative phase, and the stability of the relationship across samples.</figcaption></figure></section>

<section id="spatial"><p class="label">06 · Ben’s simultaneous spatial proposal</p><h2>Two localized density views contain real information about an unobserved region</h2><p class="lede">This first S0 test contains no temporal model. It asks only whether simultaneous sparse measurements A and B can reconstruct the full fluctuation state and predict a spatially separated diagnostic C. The observations are BES/GPI-like box averages—not faithful experimental forward models.</p><figure><div class="figure-image"><img alt="Truth observed diagnostics reconstruction error and heldout channels for simultaneous spatial reconstruction" src="__FIG_S0_HERO__"></div><figcaption><span class="fignum">Figure 5.</span> A closed-form ridge/Wiener map is fit chronologically on training frames only. The hero frame is the median held-out-C error case. The result establishes nonlocal simultaneous information; it does not establish causal forecasting or experimental realism.</figcaption></figure><figure><div class="figure-image"><img alt="Spatial reconstruction skill versus distance toroidal mode and plasma region" src="__FIG_S0_SCALE__"></div><figcaption><span class="fignum">Figure 6.</span> Held-out-C error falls below the no-fluctuation baseline, correlation remains nonzero at large separation, and recovery is sharply band-limited. In n=20–35 the retained-power ratios are __BAND_DETAIL__; at n≥40 they are only __HIGH_DETAIL__.</figcaption></figure></section>

<section id="meaning"><p class="label">07 · What is supported now</p><h2>A clean scientific readout, with the boundary kept explicit</h2><table><thead><tr><th>Supported by these figures</th><th>Not supported yet</th></tr></thead><tbody><tr><td>Large-scale one-step field motion is forecastable to a useful degree.</td><td>No current ensemble has passed the full transport-covariance gate.</td></tr><tr><td>Transport errors can be localized by position, separation, and physical mode.</td><td>The apparent diagnostic ranking from earlier ETKF tests is not yet definitive.</td></tr><tr><td>Sparse simultaneous density measurements contain nonlocal low/mid-mode information.</td><td>S0 is not a faithful BES/GPI instrument model and is not cross-shot validation.</td></tr><tr><td>Higher-mode spatial fluctuations are poorly reconstructed by the linear S0 baseline.</td><td>85606, assimilation, sensor ranking, and steering remain outside this readout.</td></tr></tbody></table><div class="callout"><strong>Immediate next decision.</strong> Review the plotted transport surface and synthetic footprints with Ben. The bounded variogram-loss campaign is implemented but paused at the user’s request; no training arm is active in this report.</div><p class="small">Forecast extraction status: __REPORT_STATUS__. All forecast visuals use model-training seed 1702, target __TARGET__, and the fixed 32-member seed banks. S0 uses frames [0,432) for training, [432,496) as the untouched guard, and [496,624) for validation.</p></section>
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
