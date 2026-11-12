"""PDHG on a small SDP where strict complementarity and nondegeneracy fail
but metric subregularity holds.

The instance (``msr-only``) has n = 4 and m = 5:

    minimize    X_44
    subject to  X_11 = 1,
                2 X_12 = 0,
                2 X_13 = 0,
                2 X_23 = 0,
                X_22 + X_33 - 2 X_14 = 0,
                X >= 0.

The KKT point is unique:

    X_* = E_11,   y_* = 0,   S_* = C - A^* y_* = E_44.

* Primal uniqueness: X_44 = 0 and X >= 0 force the fourth row and column
  of X to vanish, so X_22 + X_33 = 2 X_14 = 0, and then X = E_11.
* Dual uniqueness: S_11 = -y_1 >= 0 and the dual objective is y_1, so
  y_1 = 0.  Then S_11 = 0 forces the first row of S to vanish
  (y_2 = y_3 = y_5 = 0), and the (2,3) block of S, [[0, -y_4], [-y_4, 0]],
  is PSD only if y_4 = 0.

Regularity at the KKT point:

* Strict complementarity fails: rank X_* + rank S_* = 2 < 4.  The matrix
  X_* - S_* = diag(1, 0, 0, -1) has a two-dimensional null space.
* Primal nondegeneracy fails: A_4 = e_2 e_3^T + e_3 e_2^T lies in
  Range(A^*) and in N_{X_*} (matrices supported on rows/columns 2-4).
  Equivalently, m = 5 > dim T_{X_*} = 4.
* Dual nondegeneracy fails: E_22 - E_33 lies in Null(A) and in N_{S_*}
  (matrices supported on rows/columns 1-3).  Equivalently,
  m = 5 < dim N_{S_*} = 6.
* Metric subregularity of the KKT mapping holds.  The script checks it
  through the error bound

      ||z - z_*||_P <= kappa * ||z - R z||_P   for z near z_*,

  where R is the PDHG map and ||.||_P is the norm induced by
  P = [[I/tau, A^*], [A, I/sigma]].  When tau*sigma*||A||^2 < 1, this
  bound is equivalent to metric subregularity at a unique KKT point
  (Theorem 4.1 of the paper).  The script estimates

      q(t) = min { ||z - R z||_P / t  :  ||z - z_*||_P = t }

  by random sampling followed by Nelder-Mead.  For this instance, q(t)
  stays bounded away from zero as t -> 0.  For Example 6.1 of the paper,
  where metric subregularity fails, q(t) tends to zero.

Usage:

    python demo_msr_only.py [--outdir DIR] [--plot]

The script writes two CSV files to DIR (default: current directory):

    msr_only.csv        iter,res,Zdiff,rankX   (one PDHG run)
    msr_only_ratio.csv  t,q_msr_only,q_ex61    (error-bound test)

With --plot, it also saves a PNG preview with matplotlib.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
from scipy.optimize import minimize

RANK_TOL = 1e-12


# ----------------------------------------------------------------------
# Problem data
# ----------------------------------------------------------------------
def unit(n: int, i: int, j: int) -> np.ndarray:
    """e_i e_j^T + e_j e_i^T for i != j, and e_i e_i^T for i == j."""
    M = np.zeros((n, n))
    M[i, j] += 1.0
    if i != j:
        M[j, i] += 1.0
    return M


def msr_only_instance() -> dict:
    """The 4 x 4 instance described in the module docstring."""
    n = 4
    A = np.array(
        [
            unit(n, 0, 0),                                  # X_11
            unit(n, 0, 1),                                  # 2 X_12
            unit(n, 0, 2),                                  # 2 X_13
            unit(n, 1, 2),                                  # 2 X_23
            unit(n, 1, 1) + unit(n, 2, 2) - unit(n, 0, 3),  # X_22 + X_33 - 2 X_14
        ]
    )
    b = np.array([1.0, 0.0, 0.0, 0.0, 0.0])
    C = unit(n, 3, 3)
    return dict(name="msr-only", A=A, b=b, C=C,
                X_star=unit(n, 0, 0), y_star=np.zeros(5), S_star=unit(n, 3, 3))


def example_61_instance() -> dict:
    """Example 6.1 of the paper: minimize X_33 s.t. X_11 = 1, X_13 + X_22 = 0."""
    n = 3
    A = np.array([unit(n, 0, 0), unit(n, 1, 1) + 0.5 * unit(n, 0, 2)])
    b = np.array([1.0, 0.0])
    C = unit(n, 2, 2)
    return dict(name="Example 6.1", A=A, b=b, C=C,
                X_star=unit(n, 0, 0), y_star=np.zeros(2), S_star=unit(n, 2, 2))


# ----------------------------------------------------------------------
# Linear algebra helpers
# ----------------------------------------------------------------------
def A_map(A: np.ndarray, X: np.ndarray) -> np.ndarray:
    return np.tensordot(A, X, axes=([1, 2], [0, 1]))


def A_adj(A: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.tensordot(y, A, axes=(0, 0))


def A_norm(A: np.ndarray) -> float:
    return float(np.linalg.norm(A.reshape(A.shape[0], -1), 2))


def proj_psd(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(0.5 * (M + M.T))
    return (V * np.maximum(w, 0.0)) @ V.T


def pdhg_map(d: dict, X: np.ndarray, y: np.ndarray, tau: float, sigma: float):
    """One PDHG step (1.2) of the paper."""
    X_next = proj_psd(X - tau * (d["C"] - A_adj(d["A"], y)))
    y_next = y + sigma * (d["b"] - A_map(d["A"], 2.0 * X_next - X))
    return X_next, y_next


def p_norm(d: dict, U: np.ndarray, v: np.ndarray, tau: float, sigma: float) -> float:
    """||(U, v)||_P with P = [[I/tau, A^*], [A, I/sigma]]."""
    sq = np.sum(U * U) / tau + 2.0 * v @ A_map(d["A"], U) + v @ v / sigma
    return float(np.sqrt(max(sq, 0.0)))


def kkt_residual(d: dict, X: np.ndarray, y: np.ndarray) -> float:
    """KKT residual eps_KKT as defined in Section 8 of the paper."""
    A, b, C = d["A"], d["b"], d["C"]
    S = C - A_adj(A, y)
    w, V = np.linalg.eigh(S)
    S_plus = (V * np.maximum(w, 0.0)) @ V.T
    pobj, dobj = float(np.sum(C * X)), float(b @ y)
    denom = 1.0 + abs(pobj) + abs(dobj)
    eps_p = np.linalg.norm(A_map(A, X) - b) / (1.0 + np.linalg.norm(b))
    eps_d = max(0.0, -w[0] / (1.0 + np.linalg.norm(C)))
    eps_c = abs(np.sum(X * S_plus)) / denom
    eps_gap = abs(pobj - dobj) / denom
    return float(max(eps_p, eps_d, eps_c, eps_gap))


def numerical_rank(M: np.ndarray) -> int:
    w = np.linalg.eigvalsh(M)
    thresh = max(RANK_TOL, RANK_TOL * float(np.max(np.abs(w))))
    return int(np.count_nonzero(w > thresh))


def sym_basis(n: int, rows: list[int]) -> np.ndarray:
    """Orthonormal basis of symmetric matrices supported on rows x rows."""
    B = []
    for a, i in enumerate(rows):
        for j in rows[a:]:
            M = unit(n, i, j)
            B.append(M / np.linalg.norm(M))
    return np.array(B)


def tangent_basis(n: int, r: int) -> np.ndarray:
    """Orthonormal basis of matrices [[H11, H21^T], [H21, 0]] with H11 r x r."""
    B = []
    for i in range(r):
        for j in range(i, n):
            M = unit(n, i, j)
            B.append(M / np.linalg.norm(M))
    return np.array(B)


def intersection_dim(B1: np.ndarray, B2: np.ndarray, tol: float = 1e-10) -> int:
    """dim(span B1 cap span B2) for orthonormal bases (rows are vectorized)."""
    M1 = B1.reshape(B1.shape[0], -1)
    M2 = B2.reshape(B2.shape[0], -1)
    s = np.linalg.svd(M1 @ M2.T, compute_uv=False)
    return int(np.count_nonzero(s > 1.0 - tol))


def range_basis(A: np.ndarray) -> np.ndarray:
    M = A.reshape(A.shape[0], -1)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return Vt[s > 1e-12]


def null_basis(A: np.ndarray) -> np.ndarray:
    M = A.reshape(A.shape[0], -1)
    n = A.shape[1]
    # Work in the space of symmetric matrices via an orthonormal basis.
    Bsym = sym_basis(n, list(range(n))).reshape(-1, n * n)
    K = M @ Bsym.T
    _, s, Vt = np.linalg.svd(K)
    rank = int(np.count_nonzero(s > 1e-12))
    return (Vt[rank:] @ Bsym).reshape(-1, n, n)


# ----------------------------------------------------------------------
# Diagnostics at the KKT point
# ----------------------------------------------------------------------
def report_kkt_point(d: dict) -> None:
    A, b, C = d["A"], d["b"], d["C"]
    Xs, ys, Ss = d["X_star"], d["y_star"], d["S_star"]
    n, m = Xs.shape[0], len(b)
    print(f"Instance {d['name']}: n = {n}, m = {m}, ||A|| = {A_norm(A):.4f}")
    print(f"  primal feasibility ||A X* - b||      = {np.linalg.norm(A_map(A, Xs) - b):.1e}")
    print(f"  dual feasibility ||C - A^* y* - S*|| = {np.linalg.norm(C - A_adj(A, ys) - Ss):.1e}")
    print(f"  complementarity <X*, S*>             = {np.sum(Xs * Ss):.1e}")
    r, s = numerical_rank(Xs), numerical_rank(Ss)
    print(f"  eig(X* - S*) = {np.round(np.linalg.eigvalsh(Xs - Ss), 12)}")
    print(f"  rank X* + rank S* = {r} + {s} = {r + s} < n = {n}: "
          "strict complementarity fails")

    # The KKT point is diagonal, so T_{X*}, N_{X*}, N_{S*} are coordinate subspaces.
    T_X = tangent_basis(n, r)
    N_X = sym_basis(n, list(range(r, n)))
    N_S = sym_basis(n, list(range(0, n - s)))
    dim_p = intersection_dim(N_X, range_basis(A))
    dim_d = intersection_dim(N_S, null_basis(A))
    print(f"  dim T_X* = {len(T_X)}, dim N_S* = {len(N_S)}, m = {m}")
    print(f"  dim(N_X* cap Range A^*) = {dim_p}: primal nondegeneracy "
          + ("fails" if dim_p > 0 else "holds"))
    print(f"  dim(N_S* cap Null A)    = {dim_d}: dual nondegeneracy "
          + ("fails" if dim_d > 0 else "holds"))


# ----------------------------------------------------------------------
# Numerical test of metric subregularity
# ----------------------------------------------------------------------
def error_bound_ratio(d: dict, t: float, tau: float, sigma: float, rng: np.random.Generator,
                      num_samples: int = 200, num_starts: int = 3,
                      warm_start: np.ndarray | None = None) -> tuple[float, np.ndarray]:
    """Estimate q(t) = min ||z - Rz||_P / t over ||z - z_*||_P = t.

    A direction u (a symmetric matrix and a vector) is mapped to the point
    z = z_* + t u / ||u||_P on the sphere.  The minimum is estimated by
    sampling Gaussian directions and directions along rank-one curves
    X = v v^T, v = e_1 + s g, followed by Nelder-Mead from the best samples.
    """
    Xs, ys = d["X_star"], d["y_star"]
    n, m = Xs.shape[0], len(ys)
    iu = np.triu_indices(n)
    dim = len(iu[0]) + m

    def unpack(u):
        U = np.zeros((n, n))
        U[iu] = u[: len(iu[0])]
        U = U + U.T - np.diag(np.diag(U))
        return U, u[len(iu[0]):]

    def ratio(u):
        U, v = unpack(u)
        nrm = p_norm(d, U, v, tau, sigma)
        if nrm == 0.0:
            return np.inf
        X, y = Xs + (t / nrm) * U, ys + (t / nrm) * v
        X1, y1 = pdhg_map(d, X, y, tau, sigma)
        return p_norm(d, X1 - X, y1 - y, tau, sigma) / t

    def rank_one_direction():
        g = rng.standard_normal(n)
        g[0] = 0.0
        v = np.eye(n)[0] + t * g
        U = np.outer(v, v) - Xs
        w = t * rng.random() * rng.standard_normal(m)
        return np.concatenate([U[iu], w])

    cands = [rng.standard_normal(dim) for _ in range(num_samples)]
    cands += [rank_one_direction() for _ in range(num_samples)]
    vals = np.array([ratio(u) for u in cands])
    starts = [cands[i] for i in np.argsort(vals)[:num_starts]]
    if warm_start is not None:
        starts.append(warm_start)

    best, best_u = np.inf, None
    for u0 in starts:
        res = minimize(ratio, u0, method="Nelder-Mead",
                       options=dict(maxiter=20000, maxfev=40000, xatol=1e-12,
                                    fatol=1e-15, adaptive=True))
        if res.fun < best:
            best, best_u = float(res.fun), res.x / np.linalg.norm(res.x)
    return best, best_u


def error_bound_curve(d: dict, ts: np.ndarray, tau: float, sigma: float, seed: int) -> np.ndarray:
    """Estimate q(t) for each t in ts.

    Two sweeps over ts (decreasing, then increasing t), each warm-started
    from the minimizer at the neighboring radius; the smaller value is kept.
    """
    rng = np.random.default_rng(seed)
    q = np.full(len(ts), np.inf)
    dirs: list[np.ndarray | None] = [None] * len(ts)
    for order in (range(len(ts)), reversed(range(len(ts)))):
        u = None
        for i in order:
            val, u_new = error_bound_ratio(d, ts[i], tau, sigma, rng, warm_start=u)
            if val < q[i]:
                q[i], dirs[i] = val, u_new
            u = dirs[i]
    for t, val in zip(ts, q):
        print(f"    t = {t:.1e}:  q(t) = {val:.4e}")
    return q


# ----------------------------------------------------------------------
# PDHG runs
# ----------------------------------------------------------------------
def random_start(rng: np.random.Generator, n: int, m: int):
    """Same random initialization as the main code (symmetric Gaussian)."""
    M = rng.standard_normal((n, n))
    return 0.5 * (M + M.T), rng.standard_normal(m)


def run_pdhg(d: dict, X: np.ndarray, y: np.ndarray, tau: float, sigma: float,
             tol: float = 1e-12, max_iters: int = 100000) -> dict:
    """Run PDHG and record the quantities plotted in Section 8."""
    A, C = d["A"], d["C"]
    Xs, ys = d["X_star"], d["y_star"]
    hist = dict(iter=[], res=[], Zdiff=[], rankX=[], dist=[])
    Z = X - tau * (C - A_adj(A, y))
    for k in range(max_iters + 1):
        res = kkt_residual(d, X, y)
        hist["iter"].append(k)
        hist["res"].append(res)
        hist["rankX"].append(numerical_rank(X))
        hist["dist"].append(p_norm(d, X - Xs, y - ys, tau, sigma))
        if res <= tol or k == max_iters:
            hist["Zdiff"].append(np.nan)
            break
        X, y = pdhg_map(d, X, y, tau, sigma)
        Z_next = X - tau * (C - A_adj(A, y))
        hist["Zdiff"].append(float(np.linalg.norm(Z_next - Z)))
        Z = Z_next
    out = {key: np.array(val) for key, val in hist.items()}
    out["Z_last"] = Z
    return out


def tail_rate(dist: np.ndarray, window: int = 100, floor: float = 1e-12) -> float:
    """Geometric-mean contraction factor of dist over the last window above floor."""
    k2 = int(np.nonzero(dist > floor)[0][-1])
    k1 = max(k2 - window, 0)
    return float((dist[k2] / dist[k1]) ** (1.0 / (k2 - k1)))


def identification_iter(rank: np.ndarray) -> int:
    """First k after which rank X_k stays constant."""
    changes = np.nonzero(rank[1:] != rank[:-1])[0]
    return int(changes[-1] + 1) if changes.size else 0


def save_csv(path: str, header: str, data: np.ndarray, fmt="%.10e") -> None:
    np.savetxt(path, data, delimiter=",", header=header, comments="", fmt=fmt)


# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--outdir", default=".", help="directory for the CSV files")
    parser.add_argument("--plot", action="store_true", help="save a PNG preview")
    args = parser.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    d = msr_only_instance()
    ex = example_61_instance()
    n, m = d["X_star"].shape[0], len(d["b"])
    tau = sigma = 0.45                      # tau*sigma*||A||^2 = 0.81 < 1
    tau_ex = sigma_ex = 0.5                 # stepsizes of Section 6

    print("=" * 72)
    report_kkt_point(d)
    print(f"  tau = {tau}, sigma = {sigma}, tau*sigma*||A||^2 = "
          f"{tau * sigma * A_norm(d['A']) ** 2:.2f}")

    # ---- Error-bound test of metric subregularity ----
    print("=" * 72)
    print("Error-bound test  q(t) = min ||z - Rz||_P / t  over  ||z - z_*||_P = t")
    ts = np.logspace(-1, -4, 7)
    print(f"  {d['name']}:")
    q_new = error_bound_curve(d, ts, tau, sigma, seed=1)
    print(f"  {ex['name']} (tau = sigma = {tau_ex}):")
    q_ex = error_bound_curve(ex, ts, tau_ex, sigma_ex, seed=1)
    save_csv(os.path.join(args.outdir, "msr_only_ratio.csv"),
             "t,q_msr_only,q_ex61", np.column_stack([ts, q_new, q_ex]))

    # ---- One PDHG run for the figure ----
    print("=" * 72)
    X0, y0 = random_start(np.random.default_rng(0), n, m)
    h = run_pdhg(d, X0, y0, tau, sigma)
    k_end = int(h["iter"][-1])
    lam_min = float(np.min(np.abs(np.linalg.eigvalsh(h["Z_last"]))))
    k_id = identification_iter(h["rankX"])
    print(f"PDHG from a random start: eps_KKT = {h['res'][-1]:.2e} after {k_end} iterations")
    print(f"  smallest |eigenvalue| of the final Z-iterate = {lam_min:.2e}")
    print(f"  rank X_k = {h['rankX'][-1]} for k >= {k_id}; before that, X_k keeps a second "
          f"positive eigenvalue that decays linearly until it falls below {RANK_TOL:.0e}")
    print(f"  observed rate of ||z_k - z_*||_P = {tail_rate(h['dist']):.4f}")
    save_csv(os.path.join(args.outdir, "msr_only.csv"), "iter,res,Zdiff,rankX",
             np.column_stack([h["iter"], h["res"], h["Zdiff"], h["rankX"]]),
             fmt=["%d", "%.10e", "%.10e", "%d"])

    # ---- Uniformity over random starts ----
    rng = np.random.default_rng(2026)
    rates, iters = [], []
    for _ in range(20):
        X0, y0 = random_start(rng, n, m)
        hh = run_pdhg(d, X0, y0, tau, sigma)
        rates.append(tail_rate(hh["dist"]))
        iters.append(int(hh["iter"][-1]))
    print(f"20 random starts: observed rate in [{min(rates):.4f}, {max(rates):.4f}], "
          f"iterations to eps_KKT <= 1e-12 in [{min(iters)}, {max(iters)}]")

    if args.plot:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(1, 2, figsize=(10, 3.8))
        ax[0].semilogy(h["iter"], h["res"], "r", label="KKT residual")
        ax[0].semilogy(h["iter"], h["Zdiff"], "b", label=r"$\|Z_{k+1}-Z_k\|_F$")
        ax[0].set_xlabel("iteration $k$")
        ax[0].legend()
        ax[1].loglog(ts, q_new, "o-", label=d["name"])
        ax[1].loglog(ts, q_ex, "s-", label=ex["name"])
        ax[1].set_xlabel("$t$")
        ax[1].set_ylabel("$q(t)$")
        ax[1].legend()
        fig.tight_layout()
        fig.savefig(os.path.join(args.outdir, "msr_only_preview.png"), dpi=150)


if __name__ == "__main__":
    main()
