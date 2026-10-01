"""2D plane-wave expansion (PWEM) for in-plane (k_z = 0) photonic-crystal bands.

Square or triangular lattice of circular rods (dielectric ``eps_rod``) in a
background (``eps_bg``).  TE (H_z, in-plane E) and TM (E_z) are solved
separately.

Two Fourier factorizations of the dielectric are available:

Default (``method=None``): ``"fff"`` on the square lattice, ``"inverse"`` on
the triangular lattice (where ``"fff"`` is not implemented).  TM is the
inverse rule in both cases; the default only changes the TE operator.

``method="inverse"`` (correct)
    Build the Toeplitz matrix ``E[i, j] = eps_hat(G_i - G_j)`` from the Fourier
    coefficients of eps(r) and use its matrix inverse ``E^{-1}`` in the
    operators.  This is the "inverse rule" of Ho, Chan & Soukoulis, PRL 65,
    3152 (1990), whose convergence was explained by Li, JOSA A 13, 1870 (1996):
    for a product of two functions with concurrent, complementary jumps (here
    1/eps and the field derivatives at a dielectric boundary) the truncated
    Fourier product must be formed with the inverse of the truncated matrix
    of eps, not with the truncated coefficients of 1/eps.

``method="direct"`` (legacy)
    Use the truncated Fourier coefficients of 1/eps directly (Laurent's rule).
    Kept for benchmarking; converges slowly at high contrast.

``method="fff"`` (inverse rule + fast Fourier factorization for TE)
    TM is identical to ``"inverse"`` (E_z is a scalar; the inverse rule is the
    complete answer).  For TE the field derivative has a normal and a
    tangential component at the rod boundary and Li's rules assign the inverse
    rule to one and Laurent's rule to the other.  Following Popov & Nevière,
    JOSA A 18, 2886 (2001) and David, Benisty & Weisbuch, JOSA A 23, 1141
    (2006), the matrix standing for 1/eps in the TE operator becomes

        F_ab = [1/eps] d_ab - ([1/eps] - [eps]^-1) [N_a N_b],   a, b in {x, y}

    where [N_a N_b] is the Toeplitz matrix of the Fourier coefficients of the
    unit normal field of the rod boundary (radial for a circle, blended to an
    isotropic tensor away from the boundary).  The TE matrix is then not
    Hermitian; eigenvalues are real to round-off and obtained with ``eig``.

Frequencies are returned as omega a / 2 pi c (dimensionless).
"""

from __future__ import annotations

import numpy as np
from scipy.linalg import eig, eigh, inv
from scipy.special import j1

__all__ = [
    "lattice_geometry",
    "reciprocal_vectors",
    "fourier_coefficients",
    "eps_matrix",
    "inv_eps_matrix",
    "normal_field_matrices",
    "te_factorization",
    "kpath",
    "bands_at_k",
    "pwem_bands",
]

METHODS = ("inverse", "direct", "fff")


def lattice_geometry(lattice_type: str, a: float = 1.0):
    """Real/reciprocal basis, unit-cell area and high-symmetry k-path.

    Identical to the geometry used in ``app.py``.
    """
    if lattice_type in ("triangular", "2D Triangular Lattice"):
        a1 = a * np.array([1.0, 0.0])
        a2 = a * np.array([0.5, np.sqrt(3) / 2])
        Au = a * a * np.sqrt(3) / 2
        ra1 = (2 * np.pi / a) * np.array([1.0, -1 / np.sqrt(3)])
        ra2 = (2 * np.pi / a) * np.array([0.0, 2 / np.sqrt(3)])
        G_pt = np.array([0.0, 0.0])
        M_pt = (2 * np.pi / a) * np.array([0.0, 1 / np.sqrt(3)])
        K_pt = (2 * np.pi / a) * np.array([1 / 3, np.sqrt(3) / 3])
        path_pts = [G_pt, M_pt, K_pt, G_pt]
        labels = [r"$\Gamma$", "M", "K", r"$\Gamma$"]
    elif lattice_type in ("square", "2D Square Lattice"):
        a1 = a * np.array([1.0, 0.0])
        a2 = a * np.array([0.0, 1.0])
        Au = a * a
        ra1 = (2 * np.pi / a) * np.array([1.0, 0.0])
        ra2 = (2 * np.pi / a) * np.array([0.0, 1.0])
        G_pt = np.array([0.0, 0.0])
        X_pt = (np.pi / a) * np.array([1.0, 0.0])
        M_pt = (np.pi / a) * np.array([1.0, 1.0])
        path_pts = [G_pt, X_pt, M_pt, G_pt]
        labels = [r"$\Gamma$", "X", "M", r"$\Gamma$"]
    else:
        raise ValueError(f"unknown lattice_type {lattice_type!r}")
    return dict(a1=a1, a2=a2, Au=Au, ra1=ra1, ra2=ra2,
                path_pts=path_pts, labels=labels)


def reciprocal_vectors(ra1, ra2, N: int) -> np.ndarray:
    """Set of G = l ra1 + m ra2, |l|,|m| <= N, inside a circle of radius
    1.1 N |ra1| (the truncation used by ``app.py``).  Shape (NG, 2)."""
    Gmax = 1.1 * N * np.linalg.norm(ra1)
    lm = np.arange(-N, N + 1)
    L, M = np.meshgrid(lm, lm, indexing="ij")
    G = L.reshape(-1, 1) * ra1 + M.reshape(-1, 1) * ra2
    return G[np.linalg.norm(G, axis=1) < Gmax]


def fourier_coefficients(G: np.ndarray, value_rod: float, value_bg: float,
                         Rc: float, Au: float) -> np.ndarray:
    """Toeplitz matrix C[i, j] = f_hat(G_i - G_j) for the piecewise-constant
    function f(r) = value_rod inside the rod (radius Rc), value_bg outside.

    f_hat(0)   = value_rod P + value_bg (1 - P),   P = pi Rc^2 / Au
    f_hat(G)   = (value_rod - value_bg) P * 2 J1(|G| Rc) / (|G| Rc)
    """
    Pf = np.pi * Rc ** 2 / Au
    dG = G[:, None, :] - G[None, :, :]
    gr = np.linalg.norm(dG, axis=2) * Rc
    C = np.empty(gr.shape)
    zero = gr < 1e-10
    C[zero] = value_rod * Pf + value_bg * (1 - Pf)
    nz = ~zero
    C[nz] = (value_rod - value_bg) * Pf * 2 * j1(gr[nz]) / gr[nz]
    return C


def eps_matrix(G, eps_rod, eps_bg, Rc, Au):
    """Toeplitz matrix of the Fourier coefficients of eps(r)."""
    return fourier_coefficients(G, eps_rod, eps_bg, Rc, Au)


def inv_eps_matrix(G, eps_rod, eps_bg, Rc, Au, method: str = "inverse"):
    """The matrix that stands for 1/eps in the PWEM operators.

    method="inverse": inv(eps_matrix)      (inverse rule, Ho-Chan-Soukoulis)
    method="direct":  Toeplitz of 1/eps    (Laurent's rule, legacy app.py path)
    """
    if method == "inverse":
        return inv(eps_matrix(G, eps_rod, eps_bg, Rc, Au), check_finite=False)
    if method == "direct":
        return fourier_coefficients(G, 1.0 / eps_rod, 1.0 / eps_bg, Rc, Au)
    if method == "fff":          # TM operator is the inverse-rule one
        return inv(eps_matrix(G, eps_rod, eps_bg, Rc, Au), check_finite=False)
    raise ValueError(f"method must be one of {METHODS}, got {method!r}")


def normal_field_matrices(G, Rc, a=1.0, ra1=None, ra2=None, n=512):
    """Toeplitz matrices [N_x N_x], [N_x N_y], [N_y N_y] of the Fourier
    coefficients of the boundary-normal tensor N N^T for a circular rod of
    radius Rc (square cell of side a).

    N = r_hat in an annulus around the rod boundary (0.7 Rc < r < 1.5 Rc),
    blended smoothly to the isotropic tensor I/2 at the rod centre and toward
    the cell edge, so the field is periodic and finite everywhere.  The
    coefficients are taken from an n x n FFT of the cell.
    """
    x = (np.arange(n) / n - 0.5) * a
    X, Y = np.meshgrid(x, x, indexing="ij")
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X)

    def smooth(t):
        t = np.clip(t, 0.0, 1.0)
        return t * t * (3 - 2 * t)

    r_lo, r_in, r_out, r_edge = 0.3 * Rc, 0.7 * Rc, 1.5 * Rc, 0.5 * a
    f = smooth((r - r_lo) / (r_in - r_lo)) * (1 - smooth((r - r_out) / (r_edge - r_out)))
    # index of G_i - G_j in the FFT grid (G = 2 pi (l, m) / a on a square cell)
    dG = (G[:, None, :] - G[None, :, :]) * a / (2 * np.pi)
    li = np.rint(dG[..., 0]).astype(int) % n
    mi = np.rint(dG[..., 1]).astype(int) % n
    ll = np.fft.fftfreq(n, 1.0 / n)
    phase = (-1.0) ** (ll[:, None] + ll[None, :])        # grid starts at -a/2
    out = {}
    for key, val, iso in (("xx", np.cos(th) ** 2, 0.5),
                          ("xy", np.sin(th) * np.cos(th), 0.0),
                          ("yy", np.sin(th) ** 2, 0.5)):
        c = np.fft.fft2(f * val + (1 - f) * iso) / n ** 2 * phase
        out[key] = c[li, mi]
    return out


def te_factorization(G, eps_rod, eps_bg, Rc, Au, a=1.0):
    """The four blocks F_xx, F_xy, F_yx, F_yy of the fast-Fourier-factorized
    1/eps for the TE operator (see module docstring, method="fff")."""
    L = fourier_coefficients(G, 1.0 / eps_rod, 1.0 / eps_bg, Rc, Au)   # Laurent
    Ei = inv(eps_matrix(G, eps_rod, eps_bg, Rc, Au), check_finite=False)
    D = L - Ei
    nn = normal_field_matrices(G, Rc, a)
    return dict(xx=L - D @ nn["xx"], xy=-D @ nn["xy"],
                yx=-D @ nn["xy"], yy=L - D @ nn["yy"])


def kpath(path_pts, n_per_segment: int = 20):
    """k-points along the polyline ``path_pts`` (same sampling as app.py).

    Returns (k (Nk,2), x_axis (Nk,), x_ticks)."""
    all_k, x_axis, x_ticks, x0 = [], [], [0.0], 0.0
    for i in range(len(path_pts) - 1):
        s, e = np.asarray(path_pts[i]), np.asarray(path_pts[i + 1])
        d = np.linalg.norm(e - s)
        seg_k = [s + t * (e - s) for t in np.linspace(0, 1, n_per_segment)]
        seg_x = np.linspace(x0, x0 + d, n_per_segment)
        if i > 0:
            seg_k, seg_x = seg_k[1:], seg_x[1:]
        all_k.extend(seg_k)
        x_axis.extend(seg_x)
        x0 += d
        x_ticks.append(x0)
    return np.array(all_k), np.array(x_axis), x_ticks


def bands_at_k(k, G, Finv, nbands=None, Fte=None):
    """Eigenfrequencies (omega/c, ascending) of the TE and TM operators at k.

    TM (E_z):  M_ij = |k+G_i| |k+G_j|  Finv_ij
    TE (H_z):  M_ij = (k+G_i).(k+G_j) Finv_ij                       (Fte None)
               M_ij = sum_ab (k+G_i)_a Fte[ab]_ij (k+G_j)_b          (fff)
    """
    kG = k + G
    nk = np.linalg.norm(kG, axis=1)
    M_TM = np.outer(nk, nk) * Finv
    M_TM = 0.5 * (M_TM + M_TM.T)          # symmetrize against round-off in inv()
    sub = None if nbands is None else [0, nbands - 1]
    tm = eigh(M_TM, eigvals_only=True, subset_by_index=sub, check_finite=False)
    if Fte is None:
        M_TE = (kG @ kG.T) * Finv
        M_TE = 0.5 * (M_TE + M_TE.T)
        te = eigh(M_TE, eigvals_only=True, subset_by_index=sub, check_finite=False)
    else:
        kx, ky = kG[:, 0], kG[:, 1]
        M_TE = (np.outer(kx, kx) * Fte["xx"] + np.outer(kx, ky) * Fte["xy"]
                + np.outer(ky, kx) * Fte["yx"] + np.outer(ky, ky) * Fte["yy"])
        te = np.sort(np.real(eig(M_TE, right=False, check_finite=False)))
        if nbands is not None:
            te = te[:nbands]
    return np.sqrt(np.maximum(te, 0)), np.sqrt(np.maximum(tm, 0))


def pwem_bands(lattice_type, eps_rod, eps_bg, r_a, N, method=None,
               a=1.0, kpoints=None, n_per_segment=20, nbands=None):
    """Band structure of a 2D photonic crystal by PWEM.

    Parameters
    ----------
    lattice_type : "square" | "triangular" (app.py's long names also accepted)
    eps_rod, eps_bg : rod and background permittivity
    r_a : rod radius / lattice constant
    N : plane-wave order; G = l ra1 + m ra2 with |l|,|m| <= N (circular cut)
    method : None (default: "fff" on the square lattice, "inverse" on the
             triangular lattice), "inverse", "direct" or "fff"
    kpoints : optional (Nk,2) array of k in units where a = ``a``; default is
              the Gamma-...-Gamma path with ``n_per_segment`` points per leg
    nbands : number of lowest bands to return (default: all NG)

    Returns
    -------
    dict with keys k, x_axis, x_ticks, labels, TE, TM (arrays of
    omega a / 2 pi c, shape (Nk, nbands)), NG, G, method
    """
    geo = lattice_geometry(lattice_type, a)
    if method is None:
        method = "fff" if geo["Au"] == a * a else "inverse"
    if method not in METHODS:
        raise ValueError(f"method must be one of {METHODS}, got {method!r}")
    G = reciprocal_vectors(geo["ra1"], geo["ra2"], N)
    Finv = inv_eps_matrix(G, eps_rod, eps_bg, r_a * a, geo["Au"], method)
    Fte = None
    if method == "fff":
        if geo["Au"] != a * a:
            raise NotImplementedError('method="fff" is implemented for the square lattice only')
        Fte = te_factorization(G, eps_rod, eps_bg, r_a * a, geo["Au"], a)
    if kpoints is None:
        k, x_axis, x_ticks = kpath(geo["path_pts"], n_per_segment)
    else:
        k = np.asarray(kpoints, dtype=float)
        x_axis, x_ticks = np.arange(len(k), dtype=float), None
    nb = len(G) if nbands is None else min(nbands, len(G))
    TE = np.empty((len(k), nb))
    TM = np.empty((len(k), nb))
    for i, kk in enumerate(k):
        TE[i], TM[i] = bands_at_k(kk, G, Finv, nb, Fte)
    norm = 2 * np.pi / a
    return dict(k=k, x_axis=x_axis, x_ticks=x_ticks, labels=geo["labels"],
                TE=TE / norm, TM=TM / norm, NG=len(G), G=G, method=method)
