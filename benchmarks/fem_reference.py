"""Independent FEM reference for the 2D PWEM benchmark (scikit-fem).

Solves the in-plane Bloch problem on one unit cell of a square lattice of
circular dielectric rods with a *conformal* triangular mesh (the rod boundary
is resolved by mesh edges) and quadratic (P2) Lagrange elements.

Bloch ansatz  E_z, H_z = u(r) exp(i k.r) with u periodic on the cell:

    TM (E_z):  -(grad + ik).(grad + ik) u = (w/c)^2 eps u
    TE (H_z):  -(grad + ik).(1/eps)(grad + ik) u = (w/c)^2 u

Periodicity is imposed by identifying the DOFs on opposite cell edges
(skfem.MeshTri1DG.periodic).  Both problems are Hermitian generalized
eigenproblems A x = lambda B x; the lowest eigenvalues are obtained by
shift-invert Lanczos.

Usage:  python benchmarks/fem_reference.py [--quick]
Writes: benchmarks/results/fem_reference.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla
import triangle as tr
from skfem import (BilinearForm, Basis, ElementTriP2, MeshTri, MeshTri1DG)
from skfem.helpers import grad

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from benchmarks.config import (A, CASE, EPS_BG, EPS_ROD, K_PER_SEG, NBANDS,  # noqa: E402
                               R_A, RESULTS_DIR, sym_indices)
from pbs.pwem2d import kpath, lattice_geometry  # noqa: E402


# --------------------------------------------------------------------------
# mesh
# --------------------------------------------------------------------------
def unit_cell_mesh(h: float, r: float = R_A * A):
    """Conformal triangulation of [-a/2, a/2]^2 with a circle of radius r.

    Opposite cell edges carry identical node sets (same parametrisation) and
    the 'Y' switch forbids Steiner points on boundary segments, so the mesh
    is exactly periodic.  Returns (MeshTri, element attribute 1=rod 0=bg).
    """
    nb = max(int(round(A / h)), 4)                # nodes per cell edge
    s = np.linspace(-A / 2, A / 2, nb + 1)[:-1]
    bottom = np.c_[s, np.full(nb, -A / 2)]
    right = np.c_[np.full(nb, A / 2), s]
    top = np.c_[-s, np.full(nb, A / 2)]
    left = np.c_[np.full(nb, -A / 2), -s]
    outer = np.vstack([bottom, right, top, left])
    n_out = len(outer)
    seg_out = np.c_[np.arange(n_out), (np.arange(n_out) + 1) % n_out]

    # Rod boundary: at least 200 nodes, and the polygon radius is scaled so
    # the polygon has exactly the area pi r^2 (the fill factor is what the
    # bands are most sensitive to; the residual shape error is second order).
    nc = max(int(round(2 * np.pi * r / h)), 200)
    th = np.linspace(0, 2 * np.pi, nc, endpoint=False)
    r_poly = r * np.sqrt(2 * np.pi / (nc * np.sin(2 * np.pi / nc)))
    circ = np.c_[r_poly * np.cos(th), r_poly * np.sin(th)]
    seg_c = n_out + np.c_[np.arange(nc), (np.arange(nc) + 1) % nc]

    pslg = dict(vertices=np.vstack([outer, circ]),
                segments=np.vstack([seg_out, seg_c]),
                regions=np.array([[0.0, 0.0, 1.0, 0.0],        # inside rod
                                  [A / 2 * 0.9, A / 2 * 0.9, 0.0, 0.0]]))
    area = 0.5 * h * h
    t = tr.triangulate(pslg, f"pq30Aa{area:.8f}Y")
    m = MeshTri(t["vertices"].T.copy(), t["triangles"].T.astype(np.int64).copy())
    attr = t["triangle_attributes"].ravel().astype(int)
    return m, attr


def periodify(m: MeshTri) -> MeshTri1DG:
    p = m.p
    tol = 1e-9
    left = np.where(np.abs(p[0] + A / 2) < tol)[0]
    right = np.where(np.abs(p[0] - A / 2) < tol)[0]
    bottom = np.where(np.abs(p[1] + A / 2) < tol)[0]
    top = np.where(np.abs(p[1] - A / 2) < tol)[0]
    # match right->left by y, top->bottom by x; corners handled by both maps
    left = left[np.argsort(p[1, left])]
    right = right[np.argsort(p[1, right])]
    bottom = bottom[np.argsort(p[0, bottom])]
    top = top[np.argsort(p[0, top])]
    assert len(left) == len(right) and len(bottom) == len(top)
    assert np.allclose(p[1, left], p[1, right]) and np.allclose(p[0, bottom], p[0, top])
    # eliminate right and top nodes.  The top-right corner maps to top-left
    # (via right->left) which itself maps to bottom-left (via top->bottom);
    # build the map so each eliminated node points to a surviving node.
    ix, ix0 = [], []
    surv = {}
    for r_, l_ in zip(right, left):
        surv[r_] = l_
    for t_, b_ in zip(top, bottom):
        surv[t_] = surv.get(b_, b_)
    # resolve chains
    for k in list(surv):
        v = surv[k]
        while v in surv:
            v = surv[v]
        surv[k] = v
    for k, v in surv.items():
        ix.append(k)
        ix0.append(v)
    return MeshTri1DG.periodic(m, np.array(ix), np.array(ix0))


# --------------------------------------------------------------------------
# Bloch eigenproblem
# --------------------------------------------------------------------------
def bloch_matrices(basis: Basis, eps_elem: np.ndarray, k: np.ndarray):
    """(A_TE, B_TE, A_TM, B_TM) at Bloch vector k (complex Hermitian)."""
    kx, ky = float(k[0]), float(k[1])

    @BilinearForm(dtype=np.complex128)
    def a_tm(u, v, w):
        gu, gv = grad(u), grad(v)
        return ((gu[0] + 1j * kx * u) * (gv[0] - 1j * kx * v)
                + (gu[1] + 1j * ky * u) * (gv[1] - 1j * ky * v))

    @BilinearForm(dtype=np.complex128)
    def b_tm(u, v, w):
        return w["eps"] * u * v

    @BilinearForm(dtype=np.complex128)
    def a_te(u, v, w):
        gu, gv = grad(u), grad(v)
        return (1.0 / w["eps"]) * ((gu[0] + 1j * kx * u) * (gv[0] - 1j * kx * v)
                                   + (gu[1] + 1j * ky * u) * (gv[1] - 1j * ky * v))

    @BilinearForm(dtype=np.complex128)
    def b_te(u, v, w):
        return u * v

    eps_q = eps_elem[:, None] * np.ones((1, basis.X.shape[1]))
    A_TM = a_tm.assemble(basis, eps=eps_q)
    B_TM = b_tm.assemble(basis, eps=eps_q)
    A_TE = a_te.assemble(basis, eps=eps_q)
    B_TE = b_te.assemble(basis, eps=eps_q)
    return A_TE, B_TE, A_TM, B_TM


def lowest(Amat, Bmat, n=NBANDS, sigma=-0.05):
    vals = spla.eigsh(Amat.tocsc(), k=n, M=Bmat.tocsc(), sigma=sigma, which="LM",
                      return_eigenvectors=False)
    vals = np.sort(np.real(vals))
    return np.sqrt(np.maximum(vals, 0)) / (2 * np.pi / A)   # omega a / 2 pi c


def solve_path(h: float, kpts: np.ndarray, verbose=True):
    m, attr = unit_cell_mesh(h)
    mp = periodify(m)
    basis = Basis(mp, ElementTriP2())
    eps_elem = np.where(attr == 1, EPS_ROD, EPS_BG).astype(float)
    TE = np.empty((len(kpts), NBANDS))
    TM = np.empty((len(kpts), NBANDS))
    t0 = time.time()
    for i, k in enumerate(kpts):
        A_TE, B_TE, A_TM, B_TM = bloch_matrices(basis, eps_elem, k)
        TE[i] = lowest(A_TE, B_TE)
        TM[i] = lowest(A_TM, B_TM)
        if verbose and (i % 10 == 0 or i == len(kpts) - 1):
            print(f"  h={h:.4f}  k {i + 1}/{len(kpts)}  {time.time() - t0:.0f}s", flush=True)
    info = dict(h=h, n_elements=int(m.t.shape[1]), n_dofs=int(basis.N),
                element="P2 (ElementTriP2)")
    return TE, TM, info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="coarser meshes (smoke test)")
    args = ap.parse_args()

    geo = lattice_geometry(CASE["lattice"], A)
    kpts, x_axis, x_ticks = kpath(geo["path_pts"], K_PER_SEG)
    sym = sym_indices(K_PER_SEG)

    h_fine, h_coarse = (0.015, 0.03) if not args.quick else (0.05, 0.10)

    print(f"FEM reference: {CASE}, {len(kpts)} k-points")
    TE_c, TM_c, info_c = solve_path(h_coarse, kpts[list(sym.values())])
    TE, TM, info = solve_path(h_fine, kpts)

    # mesh-convergence estimate at Gamma, X, M (lowest NBANDS bands)
    idx = list(sym.values())
    with np.errstate(divide="ignore", invalid="ignore"):
        d_te = np.abs(TE[idx] - TE_c) / TE[idx]
        d_tm = np.abs(TM[idx] - TM_c) / TM[idx]
    d_te[~np.isfinite(d_te) | (TE[idx] < 1e-6)] = 0   # omega = 0 at Gamma
    d_tm[~np.isfinite(d_tm) | (TM[idx] < 1e-6)] = 0
    conv = dict(h_coarse=h_coarse, h_fine=h_fine,
                max_rel_change_TE_lowest4=float(d_te[:, :4].max()),
                max_rel_change_TM_lowest4=float(d_tm[:, :4].max()),
                note="relative change of the lowest 4 bands at Gamma/X/M when h is halved to h_fine (P2 elements)")
    print("mesh convergence (h/2 -> h):", conv)

    out = dict(case=CASE, k_per_segment=K_PER_SEG, nbands=NBANDS,
               k=kpts.tolist(), x_axis=x_axis.tolist(), x_ticks=x_ticks,
               labels=["Γ", "X", "M", "Γ"], sym_indices=sym,
               TE=TE.tolist(), TM=TM.tolist(), mesh=info, mesh_coarse=info_c,
               TE_coarse_sym=TE_c.tolist(), TM_coarse_sym=TM_c.tolist(),
               convergence=conv)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, "fem_reference.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", path)


if __name__ == "__main__":
    main()
