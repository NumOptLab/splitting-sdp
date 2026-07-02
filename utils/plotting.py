"""Convergence plotting."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils.output import results_dir, safe_filename


def rank_stabilization_iteration(ranks: list[int]) -> int | None:
    """First index after which the rank sequence is constant."""
    if not ranks:
        return None
    final = ranks[-1]
    for idx, value in enumerate(ranks):
        if value == final and all(later == final for later in ranks[idx:]):
            return idx
    return None


def plot_convergence(history_path: str | Path, *, stem: str | None = None) -> list[Path]:
    """Plot KKT residual and rank, saving PNG and EPS under results."""
    history = pd.read_csv(history_path)
    if stem is None:
        stem = Path(history_path).stem.replace("_history", "")
    stem = safe_filename(stem)
    out_dir = results_dir()
    iterations = history["iter"]
    fig, ax_left = plt.subplots(figsize=(7.0, 4.5))
    ax_left.semilogy(iterations, history["kkt_res"], color="tab:blue", label="KKT residual")
    iterate_metrics = (
        ("Z_diff", r"$\|Z_{k+1}-Z_k\|_F$"),
    )
    for column, label in iterate_metrics:
        if column in history.columns:
            values = pd.to_numeric(history[column], errors="coerce")
            positive_values = values.where(values > 0.0, np.nan)
            if positive_values.notna().any():
                ax_left.semilogy(iterations, positive_values, color="tab:green", linestyle="-", label=label)
    ax_left.set_xlabel("iteration")
    ax_left.set_ylabel("max KKT residual")
    ax_left.grid(True, which="both", linewidth=0.4)

    ax_right = ax_left.twinx()
    ax_right.plot(iterations, history["rank_X"], color="tab:orange", label="rank(X)")
    ax_right.set_ylabel("rank(X)")
    stable_idx = rank_stabilization_iteration([int(v) for v in history["rank_X"].tolist()])
    if stable_idx is not None:
        ax_left.axvline(float(history["iter"].iloc[stable_idx]), color="red", linestyle="-.", linewidth=1.0)

    left_lines, left_labels = ax_left.get_legend_handles_labels()
    right_lines, right_labels = ax_right.get_legend_handles_labels()
    ax_left.legend(left_lines + right_lines, left_labels + right_labels, loc="best", framealpha=1.0)
    fig.tight_layout()

    png_path = out_dir / f"{stem}_convergence.png"
    eps_path = out_dir / f"{stem}_convergence.eps"
    fig.savefig(png_path, dpi=200)
    fig.savefig(eps_path, format="eps")
    plt.close(fig)
    return [png_path, eps_path]
