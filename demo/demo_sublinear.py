import numpy as np
import matplotlib.pyplot as plt


def proj_psd(M: np.ndarray) -> np.ndarray:
    """Euclidean projection of a symmetric matrix onto S_+^n."""
    M = 0.5 * (M + M.T)
    eigvals, eigvecs = np.linalg.eigh(M)
    eigvals_plus = np.maximum(eigvals, 0.0)
    return (eigvecs * eigvals_plus) @ eigvecs.T


def A_map(X: np.ndarray) -> np.ndarray:
    """
    Linear map A(X) = (X_11, X_13 + X_22).

    Python uses zero-based indexing, so:
      X_11 = X[0,0],
      X_13 = X[0,2],
      X_22 = X[1,1].
    """
    return np.array([X[0, 0], X[0, 2] + X[1, 1]], dtype=float)


def A_star(y: np.ndarray) -> np.ndarray:
    """
    Adjoint map A^*(y) = y_1 E_11 + y_2(F_13 + E_22),
    where F_13 = (e_1 e_3^T + e_3 e_1^T)/2.

    With the Frobenius inner product,
        <F_13, X> = X_13.
    """
    Y = np.zeros((3, 3), dtype=float)

    Y[0, 0] = y[0]

    Y[1, 1] = y[1]
    Y[0, 2] = 0.5 * y[1]
    Y[2, 0] = 0.5 * y[1]

    return Y


def slow_branch_initial_point(alpha: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Leading-order slow-branch initialization.

    This is the truncated expansion
        X(alpha), y(alpha)
    from the local slow branch. It is not the exact invariant curve, but
    it is close enough to show the asymptotic k^{-1/2} behavior after a
    short stable transient.
    """
    X0 = np.array(
        [
            [1.0, alpha, -alpha**2],
            [alpha, alpha**2, -alpha**3],
            [-alpha**2, -alpha**3, alpha**4],
        ],
        dtype=float,
    )

    y0 = np.array(
        [
            -3.0 * alpha**4,
            -2.0 * alpha**2,
        ],
        dtype=float,
    )

    return X0, y0


def run_pdhg(alpha0: float = 2.0e-2, num_iter: int = 200000):
    """
    Run PDHG on the SDP

        minimize     <E_33, X>
        subject to   X_11 = 1,
                     X_13 + X_22 = 0,
                     X >= 0.

    The unique KKT point is
        X_* = E_11, y_* = 0, S_* = E_33.
    """
    tau = 0.5
    sigma = 0.5

    C = np.zeros((3, 3), dtype=float)
    C[2, 2] = 1.0

    b = np.array([1.0, 0.0], dtype=float)

    X_star = np.zeros((3, 3), dtype=float)
    X_star[0, 0] = 1.0

    y_star = np.zeros(2, dtype=float)

    X, y = slow_branch_initial_point(alpha0)

    dist_X = np.empty(num_iter + 1)
    dist_y = np.empty(num_iter + 1)
    dist_total = np.empty(num_iter + 1)
    alpha_hist = np.empty(num_iter + 1)
    feas_res = np.empty(num_iter + 1)

    for k in range(num_iter + 1):
        dist_X[k] = np.linalg.norm(X - X_star, ord="fro")
        dist_y[k] = np.linalg.norm(y - y_star)
        dist_total[k] = np.sqrt(dist_X[k] ** 2 + dist_y[k] ** 2)
        alpha_hist[k] = X[0, 1]
        feas_res[k] = np.linalg.norm(A_map(X) - b)

        if k == num_iter:
            break

        X_old = X.copy()

        # Primal step
        S = C - A_star(y)
        X = proj_psd(X - tau * S)

        # Fully extrapolated primal variable
        X_bar = 2.0 * X - X_old

        # Dual step
        y = y + sigma * (b - A_map(X_bar))

    return {
        "dist_X": dist_X,
        "dist_y": dist_y,
        "dist_total": dist_total,
        "alpha": alpha_hist,
        "feas_res": feas_res,
    }


def safe_ratio(a: np.ndarray) -> np.ndarray:
    """Compute a[k+1]/a[k], avoiding division by zero."""
    denom = a[:-1].copy()
    numer = a[1:].copy()
    ratio = np.full_like(numer, np.nan, dtype=float)
    mask = denom > 0
    ratio[mask] = numer[mask] / denom[mask]
    return ratio


def estimate_power_law_slope(k: np.ndarray, values: np.ndarray, start: int):
    """
    Estimate slope in log(values) = c + slope * log(k)
    using data k >= start.
    """
    mask = (k >= start) & (values > 0)
    coeff = np.polyfit(np.log(k[mask]), np.log(values[mask]), deg=1)
    return coeff[0]


def save_csv(filename: str, header: str, data: np.ndarray) -> None:
    """Save a CSV file with a header line suitable for pgfplots/TikZ."""
    np.savetxt(
        filename,
        data,
        delimiter=",",
        header=header,
        comments="",
        fmt="%.16e",
    )


def main():
    alpha0 = 2.0e-2
    num_iter = 1000000
    csv_stride = 1

    out = run_pdhg(alpha0=alpha0, num_iter=num_iter)

    dist_X = out["dist_X"]
    dist_y = out["dist_y"]
    dist_total = out["dist_total"]
    alpha_hist = out["alpha"]
    feas_res = out["feas_res"]

    k = np.arange(num_iter + 1)
    k_ref = np.maximum(k, 1)

    # Reference curve: scaled (2k)^(-1/2)
    ref = (2.0 * k_ref) ** (-0.5)
    match_index = 1_000
    ref_scaled = ref * dist_X[match_index] / ref[match_index]

    # Diagnostics for sublinear behavior
    ratio_X = safe_ratio(dist_X)
    ratio_total = safe_ratio(dist_total)
    ratio_alpha = safe_ratio(np.abs(alpha_hist))

    sqrt2k_alpha = np.sqrt(2.0 * k_ref) * np.abs(alpha_hist)
    sqrtk_distX = np.sqrt(k_ref) * dist_X
    sqrtk_dist_total = np.sqrt(k_ref) * dist_total

    # Indices actually used in Figure 1 and Figure 3.
    # We exclude k=0 since the first figure is log-log.
    plot_idx = np.arange(1, num_iter + 1, csv_stride)

    # ------------------------------------------------------------
    # Save CSV files for TikZ/pgfplots.
    # ------------------------------------------------------------
    fig1_data = np.column_stack(
        [
            k_ref[plot_idx],
            dist_X[plot_idx],
            dist_y[plot_idx],
            dist_total[plot_idx],
            ref_scaled[plot_idx],
        ]
    )
    save_csv(
        "pdhg_sdp_sublinear_1.csv",
        "k,dist_X,dist_y,dist_total,ref_scaled",
        fig1_data,
    )

    fig2_data = np.column_stack(
        [
            k_ref[plot_idx],
            sqrt2k_alpha[plot_idx],
            sqrtk_distX[plot_idx],
            sqrtk_dist_total[plot_idx],
        ]
    )
    save_csv(
        "pdhg_sdp_sublinear_2.csv",
        "k,sqrt2k_abs_alpha,sqrtk_distX,sqrtk_dist_total",
        fig2_data,
    )

    # Estimate power-law slope from the second half of the run.
    slope_X = estimate_power_law_slope(k_ref, dist_X, start=num_iter // 2)
    slope_total = estimate_power_law_slope(k_ref, dist_total, start=num_iter // 2)
    slope_alpha = estimate_power_law_slope(k_ref, np.abs(alpha_hist), start=num_iter // 2)

    print("Estimated log-log slopes using the second half of the run:")
    print(f"  slope for ||X_k - X_*||_F       : {slope_X:.6f}")
    print(f"  slope for total distance       : {slope_total:.6f}")
    print(f"  slope for |alpha_k|            : {slope_alpha:.6f}")
    print("Expected slope for k^{-1/2}: -0.5")
    print()

    print("Late-iteration diagnostics:")
    for kk in [1_000, 10_000, 50_000, 100_000, 200_000]:
        print(
            f"k={kk:6d}  "
            f"||X-X*||_F={dist_X[kk]:.6e}  "
            f"||y-y*||_2={dist_y[kk]:.6e}  "
            f"|alpha_k|={abs(alpha_hist[kk]):.6e}  "
            f"ratio_X={ratio_X[kk-1]:.10f}  "
            f"sqrt(2k)|alpha_k|={sqrt2k_alpha[kk]:.10f}  "
            f"sqrt(k)||X-X*||_F={sqrtk_distX[kk]:.10f}  "
            f"feas_res={feas_res[kk]:.3e}"
        )

    # Plot 1: log-log distance plot
    plt.figure(figsize=(8, 5))
    plt.loglog(k_ref, dist_X, label=r"$\|X_k-X_\star\|_F$")
    plt.loglog(k_ref, dist_y, label=r"$\|y_k-y_\star\|_2$")
    # plt.loglog(k_ref, dist_total, label=r"total distance")
    plt.loglog(k_ref, ref_scaled, "--", label=r"scaled $(2k)^{-1/2}$")
    plt.xlabel("iteration k")
    plt.ylabel("distance")
    plt.title("PDHG on degenerate SDP: log-log distance plot")
    plt.grid(True, which="both", linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pdhg_sdp_sublinear_distances.png", dpi=200)
    plt.show()

    # Plot 2: contraction ratios
    # plt.figure(figsize=(8, 5))
    # plt.plot(k_ref[1:], ratio_X, label=r"$\|X_{k+1}-X_\star\|_F/\|X_k-X_\star\|_F$")
    # # plt.plot(k_ref[1:], ratio_total, label=r"total-distance ratio")
    # plt.plot(k_ref[1:], ratio_alpha, label=r"$|\alpha_{k+1}|/|\alpha_k|$")
    # plt.xlabel("iteration k")
    # plt.ylabel("successive ratio")
    # plt.title("Successive ratios approach 1")
    # plt.grid(True, linewidth=0.5)
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("pdhg_sdp_sublinear_ratios.png", dpi=200)
    # plt.show()

    # Plot 3: scaled k^{1/2} quantities
    plt.figure(figsize=(8, 5))
    # plt.plot(k_ref, sqrt2k_alpha, label=r"$\sqrt{2k}\,|\alpha_k|$")
    plt.plot(k_ref, sqrtk_distX, label=r"$\sqrt{k}\,\|X_k-X_\star\|_F$")
    # plt.plot(k_ref, sqrtk_dist_total, label=r"$\sqrt{k}$ times total distance")
    plt.xlabel("iteration k")
    plt.ylabel("scaled quantity")
    plt.title(r"Evidence for $k^{-1/2}$ decay")
    plt.grid(True, linewidth=0.5)
    plt.legend()
    plt.tight_layout()
    plt.savefig("pdhg_sdp_sublinear_scaled.png", dpi=200)
    plt.show()


if __name__ == "__main__":
    main()
