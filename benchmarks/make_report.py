"""Figures and tables for the README benchmark section.

Reads : benchmarks/results/fem_reference.json, benchmarks/results/pwem_convergence.json
Writes: benchmarks/figures/error_vs_N.png
        benchmarks/figures/bands_overlay.png
        benchmarks/results/benchmark_section.md   (tables + captions, spliced into
                                                   README.md between the markers
                                                   <!-- benchmark:begin --> / <!-- benchmark:end -->)
Run with --readme to update README.md in place.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.config import FIG_DIR, RESULTS_DIR  # noqa: E402

COLORS = {"direct": "#2a78d6", "inverse": "#eb6834", "fff": "#1baf7a", "fem": "#0b0b0b"}
NAMES = {"direct": "direct  [1/ε]  (legacy)", "inverse": "inverse rule  [ε]⁻¹",
         "fff": "inverse rule + FFF (TE)", "fem": "FEM reference (P2, conformal)"}
TEXT, TEXT2, GRID = "#0b0b0b", "#52514e", "#e6e5e1"

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": TEXT,
    "xtick.color": TEXT2, "ytick.color": TEXT2, "axes.titlecolor": TEXT,
    "axes.spines.top": False, "axes.spines.right": False,
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.dpi": 160,
})


def load():
    with open(os.path.join(RESULTS_DIR, "fem_reference.json")) as f:
        ref = json.load(f)
    with open(os.path.join(RESULTS_DIR, "pwem_convergence.json")) as f:
        conv = json.load(f)
    return ref, conv


def fig_error_vs_N(conv):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.6), sharey=True)
    ref_floor = 100 * max(conv["fem_convergence"]["max_rel_change_TE_lowest4"],
                          conv["fem_convergence"]["max_rel_change_TM_lowest4"])
    for ax, pol in zip(axes, ("TE", "TM")):
        for m in ("direct", "inverse", "fff"):
            if pol == "TM" and m == "fff":
                continue                       # TM: fff == inverse by construction
            rows = conv["sweep"][m]
            N = [r["N"] for r in rows]
            e = [max(r[f"max_abs_err_{pol}_lowest4"], 1e-5) for r in rows]
            ax.plot(N, e, "-o", color=COLORS[m], lw=2, ms=5, label=NAMES[m])
            ax.annotate(NAMES[m].split("  ")[0].split(" (")[0], (N[-1], e[-1]),
                        xytext=(6, 0), textcoords="offset points", va="center",
                        fontsize=8.5, color=TEXT)
        ax.axhline(ref_floor, color=GRID, lw=1.5, ls="--")
        ax.text(N[0] - 0.3, ref_floor * 0.55, "FEM reference uncertainty", fontsize=8,
                color=TEXT2, ha="left", va="top")
        ax.set_yscale("log")
        ax.set_ylim(3e-4, 80)
        ax.set_xlabel("plane-wave order N   (G = l b₁ + m b₂, |l|,|m| ≤ N)")
        ax.set_title(f"{pol} polarization", loc="left", fontsize=11)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.set_xticks(N)
        ax.set_xlim(N[0] - 0.5, N[-1] + 5.5)
    axes[0].set_ylabel("max |error| vs FEM  (%)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.suptitle("PWEM convergence: lowest 4 bands at Γ, X, M — square lattice, ε = 12 rods in air, r/a = 0.2",
                 x=0.01, ha="left", fontsize=11, color=TEXT)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "error_vs_N.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def fig_bands_overlay(ref, conv):
    x = np.array(ref["x_axis"])
    ticks, labels = ref["x_ticks"], ["Γ", "X", "M", "Γ"]
    Nf = conv["N_final"]
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.2), sharey=True)
    for ax, pol in zip(axes, ("TE", "TM")):
        methods = ("direct", "inverse", "fff") if pol == "TE" else ("direct", "inverse")
        styles = {"direct": (0, (1.2, 1.2)), "inverse": (0, (4, 2)), "fff": (0, (2, 1, 0.5, 1))}
        for m in methods:
            B = np.array(conv["final"][m][pol])[:, :4]
            for b in range(4):
                ax.plot(x, B[:, b], color=COLORS[m], lw=1.8, ls=styles[m],
                        label=f"PWEM {NAMES[m].split('  ')[0]} (N = {Nf})" if b == 0 else None)
        R = np.array(ref[pol])[:, :4]
        for b in range(4):
            ax.plot(x, R[:, b], color=COLORS["fem"], lw=1.1, alpha=0.9,
                    label=NAMES["fem"] if b == 0 else None)
        for g in conv["fem_gaps"][pol]:
            if g["gap_midgap_pct"] > 0.5:
                ax.axhspan(g["lower"], g["upper"], color="#f0efec", zorder=0)
                ax.text(ticks[-1] * 0.99, 0.5 * (g["lower"] + g["upper"]),
                        f"FEM gap {g['gap_midgap_pct']:.1f} %", ha="right", va="center",
                        fontsize=8, color=TEXT2)
        for t in ticks:
            ax.axvline(t, color=GRID, lw=0.8)
        ax.set_xticks(ticks)
        ax.set_xticklabels(labels)
        ax.set_xlim(0, ticks[-1])
        ax.set_ylim(0, 0.8)
        ax.set_title(f"{pol} polarization, lowest 4 bands", loc="left", fontsize=11)
        ax.grid(True, axis="y", color=GRID, lw=0.8)
        ax.legend(frameon=False, fontsize=8, loc="lower center")
    axes[0].set_ylabel("frequency  ωa / 2πc")
    fig.suptitle(f"Band structure overlay at N = {Nf}: PWEM vs FEM",
                 x=0.01, ha="left", fontsize=11.5, color=TEXT)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "bands_overlay.png")
    fig.savefig(path)
    plt.close(fig)
    return path


def tables(ref, conv):
    sym = conv["sym_indices"]
    Nf = conv["N_final"]
    fin = {m: conv["sweep"][m][-1] for m in conv["sweep"]}
    assert all(fin[m]["N"] == Nf for m in fin)
    lines = []

    # --- table 1: band frequencies at Gamma, X, M, lowest 4 bands, N = N_final
    lines.append(f"**Table 1 - lowest four bands at Γ, X, M (ωa/2πc), PWEM at N = {Nf} "
                 f"({fin['inverse']['NG']} plane waves) vs FEM. Signed error in % in parentheses.**\n")
    lines.append("| pol. | k | band | FEM | direct [1/ε] | inverse rule [ε]⁻¹ | inverse + FFF |")
    lines.append("|---|---|---|---|---|---|---|")
    for pol in ("TM", "TE"):
        for kname, i in sym.items():
            for b in range(4):
                fem = ref[pol][i][b]
                if fem < 1e-6:
                    continue
                cells = []
                for m in ("direct", "inverse", "fff"):
                    v = fin[m][pol][list(sym).index(kname)][b]
                    e = fin[m][f"err_{pol}_pct"][list(sym).index(kname)][b]
                    cells.append(f"{v:.4f} ({e:+.3f})")
                lines.append(f"| {pol} | {kname} | {b + 1} | {fem:.4f} | " + " | ".join(cells) + " |")
    lines.append("")

    # --- table 2: max error vs N
    lines.append("**Table 2 - max |error| over the same 4 bands x 3 k-points, in %, vs plane-wave order N.**\n")
    lines.append("| N | plane waves | direct TE | direct TM | inverse TE | inverse TM | FFF TE |")
    lines.append("|---|---|---|---|---|---|---|")
    for j, N in enumerate(conv["N_list"]):
        d, iv, ff = (conv["sweep"][m][j] for m in ("direct", "inverse", "fff"))
        lines.append(f"| {N} | {d['NG']} | {d['max_abs_err_TE_lowest4']:.3f} | {d['max_abs_err_TM_lowest4']:.3f} "
                     f"| {iv['max_abs_err_TE_lowest4']:.3f} | {iv['max_abs_err_TM_lowest4']:.4f} "
                     f"| {ff['max_abs_err_TE_lowest4']:.4f} |")
    lines.append("")

    # --- table 3: gaps
    lines.append(f"**Table 3 - gaps between consecutive bands over the full Γ-X-M-Γ path "
                 f"(lower edge - upper edge, gap/midgap %). PWEM at N = {Nf}.**\n")
    lines.append("| pol. | bands | FEM | direct [1/ε] | inverse rule [ε]⁻¹ | inverse + FFF |")
    lines.append("|---|---|---|---|---|---|")
    for pol in ("TM", "TE"):
        keys = set()
        src = {"FEM": conv["fem_gaps"][pol]}
        for m in ("direct", "inverse", "fff"):
            src[m] = conv["final"][m][f"gaps_{pol}"]
        for s in src.values():
            keys |= {g["below_band"] for g in s}
        for bb in sorted(keys):
            row = []
            for name in ("FEM", "direct", "inverse", "fff"):
                g = next((g for g in src[name] if g["below_band"] == bb), None)
                row.append("—" if g is None else f"{g['lower']:.4f} - {g['upper']:.4f} ({g['gap_midgap_pct']:.2f} %)")
            lines.append(f"| {pol} | {bb}-{bb + 1} | " + " | ".join(row) + " |")
    lines.append("")

    mesh, cv = conv["fem_mesh"], conv["fem_convergence"]
    lines.append(f"FEM reference: scikit-fem, {mesh['element']}, conformal triangle mesh, "
                 f"{mesh['n_elements']} elements / {mesh['n_dofs']} DOFs (h = {mesh['h']}); halving h "
                 f"from {cv['h_coarse']} to {cv['h_fine']} changes the tabulated bands by at most "
                 f"{100 * max(cv['max_rel_change_TE_lowest4'], cv['max_rel_change_TM_lowest4']):.4f} %, "
                 f"which is the resolution floor of every error quoted above.\n")
    return "\n".join(lines)


def splice_readme(section_md, readme_path):
    b, e = "<!-- benchmark:begin -->", "<!-- benchmark:end -->"
    with open(readme_path) as f:
        s = f.read()
    if b not in s or e not in s:
        raise SystemExit(f"markers {b} / {e} not found in {readme_path}")
    pre, rest = s.split(b, 1)
    _, post = rest.split(e, 1)
    with open(readme_path, "w") as f:
        f.write(pre + b + "\n" + section_md + "\n" + e + post)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readme", action="store_true", help="splice tables into README.md")
    args = ap.parse_args()
    os.makedirs(FIG_DIR, exist_ok=True)
    ref, conv = load()
    print("wrote", fig_error_vs_N(conv))
    print("wrote", fig_bands_overlay(ref, conv))
    md = tables(ref, conv)
    path = os.path.join(RESULTS_DIR, "benchmark_section.md")
    with open(path, "w") as f:
        f.write(md)
    print("wrote", path)
    if args.readme:
        readme = os.path.join(os.path.dirname(__file__), "..", "README.md")
        splice_readme(md, readme)
        print("updated", readme)


if __name__ == "__main__":
    main()
