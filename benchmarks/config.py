"""Shared benchmark case: square lattice of dielectric rods in air.

Every number in the README benchmark section derives from this case.
"""
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
FIG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")

A = 1.0            # lattice constant
EPS_ROD = 12.0     # rod permittivity
EPS_BG = 1.0       # background permittivity  (12:1 contrast)
R_A = 0.2          # rod radius / a
K_PER_SEG = 20     # k-points per leg of Gamma-X-M-Gamma (same as app.py)
NBANDS = 8         # bands stored; lowest 4 are tabulated
N_LIST = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21]   # plane-wave orders swept
N_FINAL = 21

CASE = dict(lattice="square", eps_rod=EPS_ROD, eps_bg=EPS_BG, r_a=R_A, a=A)


def sym_indices(k_per_seg=K_PER_SEG):
    """Row indices of Gamma, X, M in the k-path produced by pbs.pwem2d.kpath."""
    return {"Γ": 0, "X": k_per_seg - 1, "M": 2 * (k_per_seg - 1)}
