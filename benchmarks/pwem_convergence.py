"""PWEM convergence sweep: direct vs inverse factorization against the FEM
reference, for increasing plane-wave order N.

Reads : benchmarks/results/fem_reference.json   (run fem_reference.py first)
Writes: benchmarks/results/pwem_convergence.json

For each N in config.N_LIST and each method:
  * lowest NBANDS TE/TM bands at Gamma, X, M   (error study)
  * for N = N_FINAL also the full Gamma-X-M-Gamma path (overlay plot, gaps)
Gaps (between consecutive bands, over the whole k-path) are listed for the
FEM reference and for both methods at N_FINAL, so spurious gaps are visible.
"""

from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.config import (CASE, K_PER_SEG, N_FINAL, N_LIST, NBANDS,  # noqa: E402
                               RESULTS_DIR, sym_indices)
from pbs.pwem2d import pwem_bands  # noqa: E402


def gaps(bands: np.ndarray, min_width: float = 1e-4):
    """List of (lower band index (1-based), lower edge, upper edge, width,
    gap/midgap %) for every positive gap between consecutive bands, taken over
    all k in ``bands`` (shape Nk x nb).  Gaps narrower than ``min_width``
    (band-touching points, e.g. degeneracies at Γ or M) are not listed."""
    out = []
    for b in range(bands.shape[1] - 1):
        lo, hi = bands[:, b].max(), bands[:, b + 1].min()
        if hi - lo > min_width:
            mid = 0.5 * (lo + hi)
            out.append(dict(below_band=b + 1, lower=float(lo), upper=float(hi),
                            width=float(hi - lo), gap_midgap_pct=float(100 * (hi - lo) / mid)))
    return out


def main():
    ref_path = os.path.join(RESULTS_DIR, "fem_reference.json")
    with open(ref_path) as f:
        ref = json.load(f)
    assert ref["case"] == CASE and ref["k_per_segment"] == K_PER_SEG
    kpts = np.array(ref["k"])
    sym = sym_indices(K_PER_SEG)
    idx = list(sym.values())
    ref_TE = np.array(ref["TE"])
    ref_TM = np.array(ref["TM"])

    sweep = {m: [] for m in ("direct", "inverse", "fff")}
    final = {}
    for N in N_LIST:
        for method in ("direct", "inverse", "fff"):
            t0 = time.time()
            if N == N_FINAL:
                r = pwem_bands(**{k: v for k, v in CASE.items() if k != "lattice"},
                               lattice_type=CASE["lattice"], N=N, method=method,
                               kpoints=kpts, nbands=NBANDS)
                TE, TM = r["TE"], r["TM"]
                final[method] = dict(TE=TE.tolist(), TM=TM.tolist(), NG=r["NG"],
                                     gaps_TE=gaps(TE), gaps_TM=gaps(TM))
                TE_s, TM_s = TE[idx], TM[idx]
            else:
                r = pwem_bands(**{k: v for k, v in CASE.items() if k != "lattice"},
                               lattice_type=CASE["lattice"], N=N, method=method,
                               kpoints=kpts[idx], nbands=NBANDS)
                TE_s, TM_s = r["TE"], r["TM"]
            with np.errstate(divide="ignore", invalid="ignore"):
                e_te = 100 * (TE_s - ref_TE[idx]) / ref_TE[idx]
                e_tm = 100 * (TM_s - ref_TM[idx]) / ref_TM[idx]
            e_te[~np.isfinite(e_te) | (ref_TE[idx] < 1e-6)] = 0.0   # omega = 0 at Gamma, band 1
            e_tm[~np.isfinite(e_tm) | (ref_TM[idx] < 1e-6)] = 0.0
            sweep[method].append(dict(
                N=N, NG=r["NG"], seconds=time.time() - t0,
                TE=TE_s.tolist(), TM=TM_s.tolist(),
                err_TE_pct=e_te.tolist(), err_TM_pct=e_tm.tolist(),
                max_abs_err_TE_lowest4=float(np.abs(e_te[:, :4]).max()),
                max_abs_err_TM_lowest4=float(np.abs(e_tm[:, :4]).max()),
                signed_mean_err_TE_lowest4=float(e_te[:, :4][e_te[:, :4] != 0].mean()),
                signed_mean_err_TM_lowest4=float(e_tm[:, :4][e_tm[:, :4] != 0].mean()),
            ))
            print(f"N={N:2d} NG={r['NG']:4d} {method:7s} "
                  f"max|err| lowest-4: TE {sweep[method][-1]['max_abs_err_TE_lowest4']:.3f}%  "
                  f"TM {sweep[method][-1]['max_abs_err_TM_lowest4']:.3f}%  ({time.time() - t0:.1f}s)",
                  flush=True)

    out = dict(case=CASE, N_list=N_LIST, N_final=N_FINAL, nbands=NBANDS,
               sym_indices=sym, sweep=sweep, final=final,
               fem_gaps=dict(TE=gaps(ref_TE), TM=gaps(ref_TM)),
               fem_mesh=ref["mesh"], fem_convergence=ref["convergence"])
    path = os.path.join(RESULTS_DIR, "pwem_convergence.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)

    print("\nGaps over the full path (lower band, edges, width, gap/midgap %):")
    for name, g in [("FEM TE", out["fem_gaps"]["TE"]), ("FEM TM", out["fem_gaps"]["TM"])] + [
            (f"{m} N={N_FINAL} {p}", final[m][f"gaps_{p}"]) for m in final for p in ("TE", "TM")]:
        print(f" {name}:")
        for gg in g:
            print(f"   band {gg['below_band']}-{gg['below_band'] + 1}: "
                  f"{gg['lower']:.4f} - {gg['upper']:.4f}  width {gg['width']:.4f}  ({gg['gap_midgap_pct']:.2f} %)")


if __name__ == "__main__":
    main()
