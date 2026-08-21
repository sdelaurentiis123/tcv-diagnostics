#!/usr/bin/env python3
"""Build the self-contained Paper 0 scientific status explainer.

This report reads only tracked 85604-era figures and frozen summary values.
It does not load simulation arrays or touch the sequestered 85606 run.
"""

from __future__ import annotations

import argparse
import base64
import io
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch


INK = "#17212b"
MUTED = "#61707c"
GRID = "#dce2e6"
TEAL = "#177c83"
BLUE = "#3f6798"
GOLD = "#b2741f"
CORAL = "#b34d43"
PURPLE = "#79639a"
GREEN = "#4f7d58"
PAPER = "#ffffff"


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "legend.frameon": False,
            "svg.fonttype": "none",
        }
    )


def svg_uri(fig: plt.Figure) -> str:
    stream = io.BytesIO()
    fig.savefig(stream, format="svg", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return "data:image/svg+xml;base64," + base64.b64encode(stream.getvalue()).decode()


def png_uri(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def plot_data_structure() -> str:
    fields = ["Ne", "Te", "Ti", "phi", "Vi"]
    fixed = np.array([0.336, -0.193, 0.133, -0.118, 0.306])
    aligned = np.array([0.516, 0.662, 0.488, 0.638, 0.482])
    full_tau = np.array([19.042, 0.819, 2.244, 10.222, 1.680])
    tor_tau = np.array([0.952, 0.530, 0.729, 0.566, 0.910])

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.4), constrained_layout=True)
    x = np.arange(len(fields))
    width = 0.36
    axes[0].bar(x - width / 2, fixed, width, color=BLUE, label="fixed grid")
    axes[0].bar(x + width / 2, aligned, width, color=TEAL, label="oracle shift aligned")
    axes[0].axhline(0, color=INK, linewidth=0.8)
    axes[0].set_xticks(x, fields)
    axes[0].set_ylim(-0.28, 0.78)
    axes[0].set_ylabel("one-step correlation")
    axes[0].set_title("A. A shared toroidal shift recovers structure", loc="left")
    axes[0].legend(ncol=2, loc="upper left")
    for xpos, value in zip(x - width / 2, fixed):
        axes[0].text(xpos, value + (0.025 if value >= 0 else -0.055), f"{value:.3f}", ha="center", va="bottom" if value >= 0 else "top", fontsize=8)
    for xpos, value in zip(x + width / 2, aligned):
        axes[0].text(xpos, value + 0.025, f"{value:.3f}", ha="center", va="bottom", fontsize=8)

    y = np.arange(len(fields))
    axes[1].scatter(full_tau, y, s=70, color=GOLD, marker="o", label="full mean-removed pattern", zorder=3)
    axes[1].scatter(tor_tau, y, s=70, color=CORAL, marker="s", label="toroidal residual", zorder=3)
    for i, (full, tor) in enumerate(zip(full_tau, tor_tau)):
        axes[1].plot([tor, full], [i, i], color=GRID, linewidth=2, zorder=1)
        axes[1].text(full * 1.06, i, f"{full:.2f}", va="center", fontsize=8)
    axes[1].axvline(1, color=MUTED, linestyle="--", linewidth=1)
    axes[1].set_xscale("log")
    axes[1].set_xlim(0.4, 27)
    axes[1].set_yticks(y, fields)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("first 1/e crossing (saved frames, log scale)")
    axes[1].set_title("B. Fast toroidal change coexists with slow background memory", loc="left")
    axes[1].legend(loc="lower right")
    fig.suptitle("85604 temporal structure · cadence 3.131905 µs · zperiod=5 · n=5k", fontsize=15, fontweight="bold")
    return svg_uri(fig)


def plot_phase35() -> str:
    labels = [
        "codec / predictor non-equivariance",
        "nonstationary interval",
        "forecast-state-dependent covariance",
        "history-dependent hidden state",
        "incomplete retained state",
        "coherent transport coordinates",
        "insufficient effective sample size",
        "unexplained failure",
    ]
    tier = np.array([3, 3, 3, 3, 3, 2, 0, 0], dtype=float)
    colors = [TEAL if value == 3 else GOLD if value == 2 else GRID for value in tier]
    budgets = np.array([32, 64, 128, 256, 416])
    pca = np.array([0.0515, 0.0822, 0.1243, 0.1762, 0.2192])
    fourier = np.array([0.0849, 0.1471, 0.2240, 0.3172, 0.3940])

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), constrained_layout=True)
    y = np.arange(len(labels))
    axes[0].barh(y, tier, color=colors, height=0.6)
    axes[0].set_yticks(y, labels)
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 3.35)
    axes[0].set_xticks([0, 2, 3], ["none", "moderate", "strong"])
    axes[0].set_title("A. Evidence-ranked causes", loc="left")
    axes[0].grid(axis="y", visible=False)
    for yi, value in zip(y, tier):
        axes[0].text(value + 0.07, yi, "strong" if value == 3 else "moderate" if value == 2 else "none", va="center", fontsize=8)

    axes[1].plot(budgets, pca * 100, "o-", color=BLUE, linewidth=2.3, label="global PCA / KL")
    axes[1].plot(budgets, fourier * 100, "s-", color=TEAL, linewidth=2.3, label="toroidal Fourier KL")
    axes[1].set_xlabel("matched coefficient budget")
    axes[1].set_ylabel("later-block residual variance captured (%)")
    axes[1].set_ylim(0, 45)
    axes[1].set_title("B. Toroidal coordinates transfer better, but remain incomplete", loc="left")
    axes[1].legend(loc="upper left")
    axes[1].annotate("39.4%", (416, 39.4), xytext=(-45, 8), textcoords="offset points", color=TEAL, fontweight="bold")
    axes[1].annotate("21.9%", (416, 21.92), xytext=(-45, -18), textcoords="offset points", color=BLUE, fontweight="bold")
    axes[1].text(
        0.02,
        0.04,
        "truth displacement: 11/88 cells\nshifted persistence MSE gain: 67.1%\nH1 alignment gain: 0.0%",
        transform=axes[1].transAxes,
        va="bottom",
        fontsize=9,
        bbox={"boxstyle": "round,pad=.5", "facecolor": "#f5f7f7", "edgecolor": GRID},
    )
    fig.suptitle("Phase 3.5: the failure is conditional and coordinate-sensitive", fontsize=15, fontweight="bold")
    return svg_uri(fig)


def plot_ecrd_status(completed: int, running: int, pending: int) -> str:
    arms = ["B5", "B5-Context", "ECRD", "ECRD-History"]
    features = ["deep context", "no z stride", "sym H1 + mean head", "multiscale noise", "2-frame history"]
    matrix = np.array(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 1, 1, 0],
            [1, 1, 1, 1, 1],
        ]
    )
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={"width_ratios": [1.7, 1]}, constrained_layout=True)
    axes[0].imshow(matrix, cmap=matplotlib.colors.ListedColormap(["#edf0f1", TEAL]), vmin=0, vmax=1, aspect="auto")
    axes[0].set_xticks(np.arange(len(features)), features, rotation=25, ha="right")
    axes[0].set_yticks(np.arange(len(arms)), arms)
    axes[0].set_title("A. Controlled intervention ladder", loc="left")
    axes[0].grid(False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axes[0].text(col, row, "✓" if matrix[row, col] else "—", ha="center", va="center", color=PAPER if matrix[row, col] else MUTED, fontweight="bold")

    status = [completed, running, pending]
    names = ["complete", "running", "queued"]
    status_colors = [GREEN, BLUE, GRID]
    bars = axes[1].bar(names, status, color=status_colors, width=0.62)
    axes[1].set_ylim(0, 12)
    axes[1].set_ylabel("M32 evaluation tasks (of 12)")
    axes[1].set_title("B. Rusty array 6913512", loc="left")
    for bar, value in zip(bars, status):
        axes[1].text(bar.get_x() + bar.get_width() / 2, value + 0.35, str(value), ha="center", fontweight="bold")
    axes[1].text(0.02, 0.91, "4 arms × 3 seeds\n126 targets × 32 members", transform=axes[1].transAxes, va="top", fontsize=9)
    fig.suptitle("ECRD asks whether conditioning + symmetry repair transport covariance", fontsize=15, fontweight="bold")
    return svg_uri(fig)


def html_document(figures: dict[str, str], commit: str, completed: int, running: int, pending: int) -> str:
    template = r'''<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta name="description" content="Rigorous visual summary of Paper 0 TCV/Hermes emulator evidence through 21 August 2026.">
<title>Paper 0 — scientific state of the union</title>
<style>
:root{--paper:#fff;--ground:#f3f5f6;--ink:#17212b;--muted:#5f6b78;--faint:#87919c;--rule:#dfe4e8;--panel:#fff;--teal:#177c83;--teal-soft:#e7f3f3;--red:#aa3d32;--red-soft:#faece9;--gold:#866416;--gold-soft:#fbf4df;--serif:"Iowan Old Style","Charter",Georgia,serif;--sans:-apple-system,BlinkMacSystemFont,"Segoe UI",Helvetica,Arial,sans-serif;--mono:"SFMono-Regular",Consolas,monospace}
@media(prefers-color-scheme:dark){:root{--paper:#12171d;--ground:#0b0f13;--ink:#dde4ea;--muted:#a1abb5;--faint:#77828e;--rule:#29323b;--panel:#151b22;--teal:#67bdc2;--teal-soft:#143236;--red:#e38e83;--red-soft:#37201e;--gold:#e1bd62;--gold-soft:#332d1b}}
:root[data-theme="light"]{--paper:#fff;--ground:#f3f5f6;--ink:#17212b;--muted:#5f6b78;--faint:#87919c;--rule:#dfe4e8;--panel:#fff;--teal:#177c83;--teal-soft:#e7f3f3;--red:#aa3d32;--red-soft:#faece9;--gold:#866416;--gold-soft:#fbf4df}
:root[data-theme="dark"]{--paper:#12171d;--ground:#0b0f13;--ink:#dde4ea;--muted:#a1abb5;--faint:#77828e;--rule:#29323b;--panel:#151b22;--teal:#67bdc2;--teal-soft:#143236;--red:#e38e83;--red-soft:#37201e;--gold:#e1bd62;--gold-soft:#332d1b}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;padding-top:48px;background:var(--ground);color:var(--ink);font-family:var(--serif);font-size:18px;line-height:1.62}.topbar{position:fixed;z-index:20;top:0;left:0;right:0;height:48px;background:var(--paper);border-bottom:1px solid var(--rule);display:flex;align-items:center}.topbar-inner{width:100%;max-width:70rem;margin:auto;padding:0 1.3rem;display:flex;justify-content:space-between;align-items:center;font-family:var(--sans);font-size:.78rem}.theme{border:1px solid var(--rule);background:var(--panel);color:var(--muted);padding:.3rem .7rem;border-radius:1rem;cursor:pointer}.wrap{max-width:70rem;margin:auto;background:var(--paper);min-height:100vh;padding:0 2.5rem 5rem;box-shadow:0 0 45px rgba(17,26,34,.06)}header{padding:3.5rem 0 2.2rem;border-bottom:1px solid var(--rule)}.eyebrow,.kicker{font-family:var(--sans);font-size:.72rem;text-transform:uppercase;letter-spacing:.14em;color:var(--teal);font-weight:700}.eyebrow{margin:0 0 1rem}h1{font-size:3.15rem;line-height:1.05;letter-spacing:-.03em;max-width:58rem;margin:0 0 1rem}h2{font-size:1.75rem;line-height:1.2;margin:.2rem 0 .7rem}.dek,.lede{color:var(--muted);max-width:52rem}.dek{font-size:1.2rem;line-height:1.45}.meta{display:flex;gap:.7rem 1.2rem;flex-wrap:wrap;font-family:var(--sans);font-size:.78rem;color:var(--faint);margin-top:1rem}nav{border-bottom:1px solid var(--rule);font-family:var(--sans);font-size:.78rem;overflow-x:auto}nav div{display:flex;gap:1.25rem;padding:1rem 0;white-space:nowrap}nav a{color:var(--muted);text-decoration:none}main section{padding:3.3rem 0;border-bottom:1px solid var(--rule);scroll-margin-top:64px}.verdict{border-left:4px solid var(--teal);background:var(--teal-soft);padding:1.2rem 1.4rem;margin:1.5rem 0 0;max-width:54rem}.verdict ul{margin:.4rem 0}.verdict li{margin:.35rem 0}.warning{border-left-color:var(--gold);background:var(--gold-soft)}figure{margin:2rem 0;border:1px solid var(--rule);background:#fff;border-radius:9px;overflow:hidden;box-shadow:0 8px 28px -22px #111}.figimg{padding:12px;background:#fff}.figimg img{display:block;width:100%;height:auto}figcaption{border-top:1px solid var(--rule);padding:.9rem 1rem 1rem;font-family:var(--sans);font-size:.82rem;line-height:1.5;color:var(--muted)}.fignum{font-weight:750;color:var(--teal)}.source{display:block;margin-top:.3rem;color:var(--faint);font-family:var(--mono);font-size:.72rem}.columns{display:grid;grid-template-columns:1fr 1fr;gap:1.4rem}.columns>div{border-top:3px solid var(--rule);padding-top:.8rem}.columns h3{margin:.1rem 0 .4rem;font-size:1.12rem}.small{font-family:var(--sans);font-size:.82rem;color:var(--muted)}code{font-family:var(--mono);font-size:.82em}footer{padding:2rem 0;color:var(--faint);font-family:var(--sans);font-size:.78rem}@media(max-width:760px){body{font-size:16px}.wrap{padding:0 1rem 3rem;box-shadow:none}h1{font-size:2.25rem}.columns{grid-template-columns:1fr}}@media print{body{padding:0;background:#fff;font-size:10.5pt}.topbar,nav,.theme{display:none}.wrap{max-width:none;margin:0;padding:0;box-shadow:none}figure{break-inside:avoid;box-shadow:none}main section{padding:1.5rem 0}h1{font-size:28pt}}
</style>
</head>
<body>
<div class="topbar"><div class="topbar-inner"><strong>Paper 0 · TCV/Hermes emulator</strong><button class="theme" type="button" aria-label="Toggle light and dark theme">theme</button></div></div>
<div class="wrap">
<header>
<p class="eyebrow">Scientific state of the union · 21 August 2026</p>
<h1>What we have actually learned about the emulator</h1>
<p class="dek">A figure-first account of the 85604 data, representation ladder, deterministic and stochastic one-step failures, the localized covariance defect, and the ECRD repair now running on Rusty.</p>
<div class="meta"><span>85604 development evidence only</span><span>85606 remains sealed</span><span>transport is evaluation-only</span><span>snapshot commit __COMMIT__</span></div>
</header>
<nav aria-label="Contents"><div><a href="#answer">One-minute answer</a><a href="#data">Data</a><a href="#codec">Representation</a><a href="#models">Models</a><a href="#covariance">Covariance</a><a href="#causes">Causes</a><a href="#current">Current run</a><a href="#claims">Claims</a></div></nav>
<main>
<section id="answer"><p class="kicker">01 · Start here</p><h2>The one-minute answer</h2><div class="verdict"><ul>
<li><strong>The data are not incoherent.</strong> Fast toroidal translation is superposed on slower profile and amplitude drift.</li>
<li><strong>Compression is no longer the main blocker.</strong> A matched C5P DCAE-L10 passes the complete O1 reconstruction gate at all three seeds.</li>
<li><strong>The failure occurs in one saved step.</strong> H1/H2 and B2–B5 improve ordinary field scores but fail realization-level spectra and nonlinear transport.</li>
<li><strong>B5 is the best stochastic baseline so far.</strong> It improves fair CRPS, mean error, and expected power, but its covariance is organized incorrectly.</li>
<li><strong>ECRD is the controlled repair.</strong> All four arms and three seeds trained; the full M32 gate is still running.</li>
</ul></div></section>

<section id="data"><p class="kicker">02 · Data and time</p><h2>Slow drift plus fast coherent motion</h2><p class="lede">The saved cadence is 3.131905 µs. The simulated wedge is one fifth of the torus, so stored Fourier index k corresponds to physical toroidal mode n=5k.</p><figure><div class="figimg"><img alt="Fixed and shift-aligned correlations plus decorrelation times" src="__FIG_DATA__"></div><figcaption><span class="fignum">Figure 1.</span> A shared truth-assisted toroidal shift raises one-step correlation from the fixed-grid values to roughly 0.48–0.66. Toroidal residuals cross 1/e within one saved frame, while the full pattern retains slower background memory.<span class="source">PHASE1_READOUT.md · job 6890606 · oracle alignment is nondeployable</span></figcaption></figure></section>

<section id="codec"><p class="kicker">03 · Representation</p><h2>More latent capacity was not automatically better</h2><p class="lede">The historical f8/z44 comparison was not a controlled capacity ablation. It nevertheless showed exactly why pixel error cannot stand in for transport fidelity.</p><div class="columns"><div><h3>Field and spectral reconstruction</h3><p>f8 has lower five-field RMSE and preserves the dominant n=20–25 band. z44 retains some higher modes but is not uniformly better.</p></div><div><h3>Transport reconstruction</h3><p>Net separatrix transport can look excellent while local signed face contributions remain wrong. Both scales must be checked.</p></div></div><figure><div class="figimg"><img alt="Historical codec field reconstruction" src="__FIG_CODEC_FIELD__"></div><figcaption><span class="fignum">Figure 2.</span> Both historical codecs reconstruct fields well; f8 aggregate standardized RMSE is 0.02492 versus 0.03279 for z44. The later matched C5P DCAE-L10—not either historical checkpoint—passes complete O1 at 3/3 seeds.<span class="source">PHASE2_O1_READOUT.md · job 6890650 · amendment A014</span></figcaption></figure><figure><div class="figimg"><img alt="Codec local and integrated transport errors" src="__FIG_CODEC_TRANSPORT__"></div><figcaption><span class="fignum">Figure 3.</span> f8 local-face transport error is about 29–30%, but integrated separatrix error is only 2.7–5.3%. Spatial cancellation makes the integral easier than the local map.<span class="source">PHASE2_O1_TRANSPORT_READOUT.md · job 6891766</span></figcaption></figure></section>

<section id="models"><p class="kicker">04 · One-step models</p><h2>Good marginal forecasts still miss the realized physics</h2><p class="lede">The fault appears before autoregressive feedback: these are teacher-forced one-step tests on all 126 chronological validation targets.</p><figure><div class="figimg"><img alt="B3 to B5 one-step model comparison" src="__FIG_MODEL__"></div><figcaption><span class="fignum">Figure 4.</span> B4 gives the strongest deterministic mean but nearly collapsed spread. B5 gives the best marginal fair CRPS and 13/15 expected-power checks, yet B3, B4, and B5 all pass only 4/15 realization-coherence checks.<span class="source">PHASE3_B3_READOUT.md · PHASE3_B4_READOUT.md · PHASE3_B5_READOUT.md</span></figcaption></figure></section>

<section id="covariance"><p class="kicker">05 · The decisive B5 finding</p><h2>Local variance is present; coherent transport variance is not</h2><p class="lede">B5 is not simply too narrow everywhere. It is close to calibrated for pooled local face-flux contributions and badly underdispersed after those same contributions are integrated.</p><figure><div class="figimg"><img alt="B5 local versus integrated transport calibration" src="__FIG_TRANSPORT__"></div><figcaption><span class="fignum">Figure 5.</span> Local transport spread–skill is near one, while integrated separatrix spread–skill is only 0.413–0.485. The defect is spatial, modal, and cross-field covariance—not a missing global multiplier.<span class="source">PHASE3_B5_READOUT.md · job 6901587</span></figcaption></figure><figure><div class="figimg"><img alt="B5 separatrix covariance localization" src="__FIG_COVARIANCE__"></div><figcaption><span class="fignum">Figure 6.</span> Scalar inflation sufficient to repair integrated spread would overdisperse the local field by about 2.1–2.4×. The ensemble must reorganize covariance, not merely increase amplitude.<span class="source">PHASE3_B5_COVARIANCE_LOCALIZATION_READOUT.md · job 6901914</span></figcaption></figure></section>

<section id="causes"><p class="kicker">06 · Cause localization</p><h2>The transition is conditional and not exactly equivariant</h2><p class="lede">Phase 3.5 rejects the broad claim that stochastic emulation is impossible. It rejects only one fixed, global, condition-independent linear residual law.</p><figure><div class="figimg"><img alt="Phase 3.5 evidence ranking and representation transfer" src="__FIG_PHASE35__"></div><figcaption><span class="fignum">Figure 7.</span> Strong evidence supports symmetry defects, drift, state dependence, retained-state limitations, and memory. Effective sample size is not the primary explanation. Fourier coordinates improve later-block variance capture from 21.9% to 39.4% at the largest matched budget, but do not solve every transport dependency.<span class="source">phase3_5/PHASE3_5_DECISION_MEMO.md · job 6907468</span></figcaption></figure></section>

<section id="current"><p class="kicker">07 · What is running</p><h2>ECRD tests the mechanisms implicated by the evidence</h2><p class="lede">The ladder adds deep raw-field conditioning, exact toroidal handling in the generator, a symmetrized H1 parent plus mean correction, joint multiscale noise, and then the smallest history extension.</p><figure><div class="figimg"><img alt="ECRD intervention matrix and live Rusty status" src="__FIG_ECRD__"></div><figcaption><span class="fignum">Figure 8.</span> All 12 model runs completed 100 epochs and 10,800 optimizer updates. At this snapshot, __COMPLETED__/12 M32 evaluations are complete, __RUNNING__ are running, and __PENDING__ are queued. Training objectives differ across mean-head arms and are not a scientific ranking.<span class="source">ECRD_MODEL_DEVELOPMENT_PROTOCOL.md · array 6913512 · reducer 6913520</span></figcaption></figure><div class="verdict warning"><strong>Next decision:</strong> the frozen seven-family reducer checks fair CRPS, spectral retention, Ne–phi dependence, spatial transport covariance, integrated spread without local overdispersion, and multi-seed robustness. Only a passing arm may open one evaluation on 85606.</div></section>

<section id="claims"><p class="kicker">08 · Scientific boundary</p><h2>What we can and cannot say</h2><div class="columns"><div><h3>Supported</h3><ul><li>There is coherent toroidal motion and slow chronological drift.</li><li>The matched codec can preserve the evaluated representation.</li><li>Low field error and mean cross-phase do not guarantee transport-faithful joint forecasts.</li><li>B5's dominant defect is covariance organization, not zero spread.</li></ul></div><div><h3>Not supported</h3><ul><li>No model has yet passed the complete Paper 0 forecast gate.</li><li>No autonomous rollout has been accepted.</li><li>85606 generalization remains unknown.</li><li>ETKF/EnKF, diagnostic ranking, and steering remain unauthorized.</li></ul></div></div></section>
</main>
<footer>Self-contained report. Figures are regenerated from tracked Paper 0 summaries or embedded from tracked publication plots. No raw simulation data or held-out 85606 artifact is accessed by the builder.</footer>
</div>
<script>const b=document.querySelector('.theme');b.addEventListener('click',()=>{const r=document.documentElement;const d=r.dataset.theme==='dark'||(!r.dataset.theme&&matchMedia('(prefers-color-scheme: dark)').matches);r.dataset.theme=d?'light':'dark';});</script>
</body></html>'''
    replacements = {
        "__COMMIT__": commit,
        "__COMPLETED__": str(completed),
        "__RUNNING__": str(running),
        "__PENDING__": str(pending),
        "__FIG_DATA__": figures["data"],
        "__FIG_CODEC_FIELD__": figures["codec_field"],
        "__FIG_CODEC_TRANSPORT__": figures["codec_transport"],
        "__FIG_MODEL__": figures["model"],
        "__FIG_TRANSPORT__": figures["transport"],
        "__FIG_COVARIANCE__": figures["covariance"],
        "__FIG_PHASE35__": figures["phase35"],
        "__FIG_ECRD__": figures["ecrd"],
    }
    for key, value in replacements.items():
        template = template.replace(key, value)
    return template


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--commit", required=True)
    parser.add_argument("--completed", type=int, default=8)
    parser.add_argument("--running", type=int, default=2)
    parser.add_argument("--pending", type=int, default=2)
    args = parser.parse_args()
    if args.completed + args.running + args.pending != 12:
        raise ValueError("ECRD task counts must sum to 12")
    configure_matplotlib()
    figures = {
        "data": plot_data_structure(),
        "codec_field": png_uri(args.repo / "paper0/figures/phase2_o1/codec-field-reconstruction.png"),
        "codec_transport": png_uri(args.repo / "paper0/figures/phase2_o1_transport/codec-transport-attribution.png"),
        "model": png_uri(args.repo / "paper0/figures/phase3_b5/b5-model-comparison.png"),
        "transport": png_uri(args.repo / "paper0/figures/phase3_b5/b5-transport-localization.png"),
        "covariance": png_uri(args.repo / "paper0/figures/phase3_b5_covariance_localization/b5-covariance-separatrix-transport.png"),
        "phase35": plot_phase35(),
        "ecrd": plot_ecrd_status(args.completed, args.running, args.pending),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        html_document(figures, args.commit, args.completed, args.running, args.pending),
        encoding="utf-8",
    )
    print(args.output)


if __name__ == "__main__":
    main()
