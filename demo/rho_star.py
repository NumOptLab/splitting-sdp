"""Bound rho_star on the local linear rate of PDHG (Theorem 5.6).

For a PDHG run whose final iterate approximates a strictly complementary KKT
point z_star = (X_star, y_star), the script computes rho_star, the largest
modulus of the eigenvalues other than one of the linearized PDHG map

    G_star(U, v) = (J(W), v + sigma*A(U) - 2*sigma*A(J(W))),
    W = U + tau*A^*(v),

where J is the derivative of the projection onto the PSD cone at
Z_star = X_star - tau*S_star.

Reduction.  Let Z_star = Q diag(lambda) Q^T.  The matrices B_ij = Q E_ij Q^T,
with E_ij = e_i e_i^T or (e_i e_j^T + e_j e_i^T)/sqrt(2), form an orthonormal
eigenbasis of J: J(B_ij) = Gamma_ij B_ij, with Gamma_ij = 1 if lambda_i > 0
and lambda_j > 0, Gamma_ij = lambda_i/(lambda_i - lambda_j) if lambda_i > 0 >
lambda_j, and Gamma_ij = 0 if both are negative.  The range of J is spanned by
the d = r(r+1)/2 + r(n-r) basis matrices with Gamma_ij > 0, and range(J) x R^m
is invariant under G_star and contains every eigenvector with a nonzero
eigenvalue.  Let M (m x d) have columns A(B_ij) and D = diag(Gamma_ij).  In
the coordinates u of U in this basis,

    G_V = [ D                        tau*D*M^T                ]
          [ sigma*M - 2*sigma*M*D    I - 2*sigma*tau*M*D*M^T  ],

of order d + m.  When m is large, write M^T M = V S^2 V^T; the vectors v
orthogonal to range(M) are fixed points (eigenvalue one), and the remaining
eigenvalues are those of the order-2d matrix

    [ D                        tau*D*V*S                   ]
    [ sigma*S*V^T*(I - 2D)     I - 2*sigma*tau*S*V^T*D*V*S ].

The eigenvalues are computed densely.  Eigenvalues within `near_one` of one
belong to the tangent space T_M of the local KKT set; rho_star is the largest
modulus of the others.

Usage (from the code directory):

    python demo/rho_star.py results/QS_20_*_solution.h5 [more .h5 files]
        [--out rho_star.csv] [--near-one 1e-6]

Each .h5 file is a solution file written by `python -m cli solve`.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

import h5py
import numpy as np
import scipy.sparse as sp

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from core.operators import svec, smat  # noqa: E402

DIRECT_MAX = 6000      # use G_V directly when d + m is at most this


def load(path):
    with h5py.File(path, "r") as f:
        def csr(g, shape=None):                       # CSR group or dense dataset
            if isinstance(g, h5py.Dataset):
                return sp.csr_matrix(g[:])
            data, ind, ptr = g["data"][:], g["indices"][:], g["indptr"][:]
            return sp.csr_matrix((data, ind, ptr), shape=shape)
        X = f["X_last"][:]
        n = X.shape[0]
        gA = f["A"]
        A = csr(gA, shape=None if isinstance(gA, h5py.Dataset)
                else (len(gA["indptr"]) - 1, n * (n + 1) // 2))
        C = csr(f["C"], shape=(n, n)).toarray()
        y = f["y_last"][:]
        attrs = dict(f.attrs)
    return dict(name=str(attrs["problem_name"]), A=A, C=C, X=X, y=y,
                tau=float(attrs["tau"]), sigma=float(attrs["sigma"]))


def rho_star(d, near_one=1e-6, verbose=True):
    A, C, X, y, tau, sigma = d["A"], d["C"], d["X"], d["y"], d["tau"], d["sigma"]
    n, m = X.shape[0], A.shape[0]
    At = A.T.tocsr()
    Aop = lambda U: A @ svec(U)                      # noqa: E731
    Aadj = lambda v: smat(At @ v)                    # noqa: E731

    t0 = time.time()
    S = C - Aadj(y)
    S = 0.5 * (S + S.T)
    Z = X - tau * S
    lam, Q = np.linalg.eigh(0.5 * (Z + Z.T))
    pos = np.flatnonzero(lam > 0)
    neg = np.flatnonzero(lam <= 0)
    r = len(pos)
    min_abs = float(np.min(np.abs(lam)))

    # Orthonormal eigenbasis of J on its range, and the matrix M = [A(B_ij)].
    pairs = [(i, j, 1.0) for a, i in enumerate(pos) for j in pos[a:]]
    pairs += [(i, j, lam[i] / (lam[i] - lam[j])) for i in pos for j in neg]
    dim = len(pairs)
    M = np.empty((m, dim))
    for c, (i, j, _) in enumerate(pairs):
        if i == j:
            B = np.outer(Q[:, i], Q[:, i])
        else:
            B = (np.outer(Q[:, i], Q[:, j]) + np.outer(Q[:, j], Q[:, i])) / np.sqrt(2)
        M[:, c] = Aop(B)
    Dg = np.array([g for _, _, g in pairs])

    if dim + m <= DIRECT_MAX:
        MD = M * Dg
        G = np.block([[np.diag(Dg), tau * Dg[:, None] * M.T],
                      [sigma * M - 2 * sigma * MD, np.eye(m) - 2 * sigma * tau * MD @ M.T]])
        extra_ones = 0
    else:
        s2, V = np.linalg.eigh(M.T @ M)
        s = np.sqrt(np.maximum(s2, 0.0))
        VS = V * s
        DVS = Dg[:, None] * VS
        G = np.block([[np.diag(Dg), tau * DVS],
                      [sigma * VS.T - 2 * sigma * DVS.T,
                       np.eye(dim) - 2 * sigma * tau * VS.T @ DVS]])
        extra_ones = m - dim                       # v orthogonal to range(M)
    mu = np.linalg.eigvals(G)
    dist1 = np.abs(mu - 1)
    one = dist1 <= near_one
    others = mu[~one]
    order = np.argsort(-np.abs(others))
    rho = float(np.abs(others[order[0]]))
    dim_TM = int(one.sum()) + extra_ones
    t_eig = time.time() - t0

    # Finite-difference check of G_star against the PDHG map at the final iterate.
    rng = np.random.default_rng(0)
    U0 = rng.standard_normal((n, n))
    U0 = (U0 + U0.T) / 2
    v0 = rng.standard_normal(m)
    nrm = np.sqrt(np.sum(U0 * U0) + v0 @ v0)
    U0, v0 = U0 / nrm, v0 / nrm
    Gam = np.zeros((n, n))
    for i, j, g in pairs:
        Gam[i, j] = Gam[j, i] = g

    def J(H):
        return Q @ (Gam * (Q.T @ H @ Q)) @ Q.T

    def pdhg(Xc, yc):                               # PDHG step without b (it cancels)
        Sc = C - Aadj(yc)
        Zc = Xc - tau * 0.5 * (Sc + Sc.T)
        w_, V_ = np.linalg.eigh(0.5 * (Zc + Zc.T))
        Xn = (V_ * np.maximum(w_, 0)) @ V_.T
        return Xn, yc - sigma * Aop(2 * Xn - Xc)
    eps = 1e-7 * max(1.0, np.linalg.norm(X))
    X1, y1 = pdhg(X + eps * U0, y + eps * v0)
    X0, y0 = pdhg(X, y)
    fd = np.concatenate([svec((X1 - X0) / eps), (y1 - y0) / eps])
    JW = J(U0 + tau * Aadj(v0))
    G0 = np.concatenate([svec(JW), v0 + sigma * Aop(U0) - 2 * sigma * Aop(JW)])
    fd_err = float(np.linalg.norm(fd - G0) / np.linalg.norm(G0))

    out = dict(name=d["name"], n=n, m=m, tau=tau, sigma=sigma, rank_X=r,
               min_abs_eig_Z=min_abs, dim_TM=dim_TM, rho_star=rho,
               top=others[order[:8]], fd_err=fd_err,
               max_dist_one=float(dist1[one].max()) if one.any() else 0.0,
               time=t_eig)
    if verbose:
        print(f"{d['name']}: n={n}, m={m}, rank X={r}, order {G.shape[0]}, "
              f"dim T_M={dim_TM}, rho_star={rho:.8f}, fd err={fd_err:.1e}, "
              f"|mu-1| <= {out['max_dist_one']:.1e} on T_M vs {1 - rho:.1e} for rho_star, "
              f"time {t_eig:.0f} s", flush=True)
        print("  largest eigenvalues other than one: "
              + ", ".join(f"{z.real:.6f}{z.imag:+.6f}i" for z in out["top"]), flush=True)
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("files", nargs="+")
    p.add_argument("--out", default=None, help="CSV file for the results")
    p.add_argument("--near-one", type=float, default=1e-6)
    a = p.parse_args()
    rows = ["instance,n,m,tau,sigma,rank_X,min_abs_eig_Z,dim_TM,rho_star,fd_err"]
    for path in a.files:
        o = rho_star(load(path), near_one=a.near_one)
        rows.append(f"{o['name']},{o['n']},{o['m']},{o['tau']},{o['sigma']},{o['rank_X']},"
                    f"{o['min_abs_eig_Z']:.6e},{o['dim_TM']},{o['rho_star']:.10f},{o['fd_err']:.2e}")
    if a.out:
        with open(a.out, "w") as f:
            f.write("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
