#!/usr/bin/env python3
"""
Generate moderately nontrivial SDP instances in SDPA sparse .dat-s format.

Problem convention encoded in the .dat-s files:
    min <C, X>
    s.t. <A_k, X> = b_k, k=1,...,m,
         X >= 0.

SDPA sparse convention used here:
    matno = 0 stores C,
    matno = k stores A_k.

Construction:
    n = r + g + s, with r,g,s >= 1.
    X_* = diag(I_r, 0_g, 0_s),
    S_* = diag(0_r, 0_g, I_s),
    C = S_*.

The constraints span all symmetric matrices on the leading p=(r+g) block.
Each gap-gap constraint also contains a moderate cross coupling between an
X-positive coordinate and an S-positive coordinate. This removes the normal
gap directions, giving primal-dual nondegeneracy, while r+s=n-g<n gives
failure of strict complementarity.

To make the instances less axis-trivial but not ill-conditioned, the core
constraints are mixed by an orthogonal matrix generated from deterministic
Givens rotations. Hence the linear operator is not diagonal in the coordinate
basis, but its core conditioning is benign.
"""

from __future__ import annotations
import argparse
import json
import math
import zipfile
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def fmt(x: float) -> str:
    if abs(x) < 1e-14:
        return "0"
    if abs(x - round(x)) < 1e-12:
        return str(int(round(x)))
    return f"{x:.12e}"


def basis_pairs(p: int) -> List[Tuple[int, int]]:
    return [(i, j) for i in range(1, p + 1) for j in range(i, p + 1)]


def gap_pairs(r: int, g: int) -> List[Tuple[int, int]]:
    return [(r + a, r + b) for a in range(1, g + 1) for b in range(a, g + 1)]


def orthogonal_givens(q: int, sweeps: int = 2, theta: float = 0.35) -> np.ndarray:
    """Deterministic, well-conditioned mixing matrix.

    Returns a q-by-q orthogonal matrix. A few local Givens sweeps are enough to
    destroy the completely coordinate-aligned structure without introducing
    ill conditioning.
    """
    Q = np.eye(q)
    c = math.cos(theta)
    s = math.sin(theta)
    for sweep in range(sweeps):
        # Even/odd adjacent rotations.
        for parity in [0, 1]:
            for i in range(parity, q - 1, 2):
                Qi = Q[i, :].copy()
                Qj = Q[i + 1, :].copy()
                Q[i, :] = c * Qi + s * Qj
                Q[i + 1, :] = -s * Qi + c * Qj
        # A second, longer-range pass avoids purely banded mixing.
        stride = 7 + sweep
        for i in range(0, q - stride, max(3, stride)):
            Qi = Q[i, :].copy()
            Qj = Q[i + stride, :].copy()
            Q[i, :] = c * Qi + s * Qj
            Q[i + stride, :] = -s * Qi + c * Qj
    return Q


def add_entry(entries: Dict[Tuple[int, int, int, int], float], matno: int, block: int, i: int, j: int, val: float) -> None:
    if abs(val) < 1e-14:
        return
    if i > j:
        i, j = j, i
    key = (matno, block, i, j)
    entries[key] = entries.get(key, 0.0) + float(val)


def generate_instance(n: int, m: int, r: int, g: int, s: int, output: Path,
                      eps_gap: float = 0.25, sweeps: int = 2,
                      theta: float = 0.35) -> Dict:
    if r + g + s != n:
        raise ValueError("Require n = r + g + s.")
    if min(r, g, s) < 1:
        raise ValueError("Require r,g,s >= 1.")
    p = r + g
    pairs = basis_pairs(p)
    q = len(pairs)
    gpairs = gap_pairs(r, g)
    dg = len(gpairs)
    if r * s < dg:
        raise ValueError(f"Need r*s >= g(g+1)/2 for distinct gap couplings; got r*s={r*s}, dg={dg}.")
    if m != q:
        raise ValueError(f"For the well-conditioned default construction use m=q=(r+g)(r+g+1)/2={q}. Got m={m}.")

    T = orthogonal_givens(q, sweeps=sweeps, theta=theta)
    pair_to_col = {ij: t for t, ij in enumerate(pairs)}

    # Assign each gap-gap direction to a distinct cross term E_{a,b}+E_{b,a}
    # with a in the X-positive block and b in the S-positive block.
    cross_for_gap: Dict[Tuple[int, int], Tuple[int, int]] = {}
    idx = 0
    for gg in gpairs:
        a = idx % r + 1
        b = p + (idx // r) + 1
        if b > n:
            raise ValueError("Not enough S-positive coordinates for distinct gap couplings.")
        cross_for_gap[gg] = (a, b)
        idx += 1

    entries: Dict[Tuple[int, int, int, int], float] = {}

    # Objective C = S_*.
    for i in range(p + 1, n + 1):
        add_entry(entries, 0, 1, i, i, 1.0)

    bvec: List[float] = []
    for k in range(1, m + 1):
        row = T[k - 1, :]
        # b_k = <A_k, X_*>, and X_* has ones on the first r diagonal entries.
        bk = 0.0
        for i in range(1, r + 1):
            bk += row[pair_to_col[(i, i)]]
        bvec.append(float(bk))

        # Leading p-by-p block, orthogonally mixed.
        for col, (i, j) in enumerate(pairs):
            add_entry(entries, k, 1, i, j, row[col])

        # Moderate cross coupling for gap-gap directions.
        for gg, (a, b) in cross_for_gap.items():
            col = pair_to_col[gg]
            add_entry(entries, k, 1, a, b, eps_gap * row[col])

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as f:
        f.write(f"{m}\n")
        f.write("1\n")
        f.write(f"{n}\n")
        for t in range(0, m, 6):
            f.write(" ".join(fmt(x) for x in bvec[t:t+6]) + "\n")
        for (matno, block, i, j), val in sorted(entries.items()):
            if abs(val) >= 1e-14:
                f.write(f"{matno} {block} {i} {j} {fmt(val)}\n")

    meta = {
        "file": output.name,
        "n": n,
        "m": m,
        "r": r,
        "g": g,
        "s": s,
        "p": p,
        "q": q,
        "rank_X_star": r,
        "rank_S_star": s,
        "rank_sum": r + s,
        "strict_complementarity": False,
        "sc_gap_dimension": g,
        "lambda_min_pos_X_star": 1.0,
        "lambda_min_pos_S_star": 1.0,
        "eps_gap": eps_gap,
        "mixing": "orthogonal deterministic Givens mixing",
        "mixing_condition_number": 1.0,
        "nonzero_entries": sum(1 for v in entries.values() if abs(v) >= 1e-14),
        "guarantee": "unique optimum; primal-dual ND; strict complementarity fails",
    }
    return meta


def default_params(n: int) -> Tuple[int, int, int, int]:
    # m=q=(r+g)(r+g+1)/2. Choose modest p=r+g so the large files stay sparse.
    defaults = {
        10:   (3, 2, 5, 15),
        100:  (12, 4, 84, 136),
        200:  (16, 4, 180, 210),
        500:  (22, 5, 473, 378),
        1000: (30, 5, 965, 630),
    }
    if n in defaults:
        return defaults[n]
    g = 4 if n >= 50 else 2
    r = max(g + 1, int(math.sqrt(n)))
    s = n - r - g
    if s < 1:
        raise ValueError("n too small for automatic parameters; specify r,g,s,m.")
    p = r + g
    m = p * (p + 1) // 2
    return r, g, s, m


def write_readme(outdir: Path, metas: List[Dict]) -> None:
    with (outdir / "README.txt").open("w") as f:
        f.write("Simpler PDND/non-SC SDP instances in SDPA sparse .dat-s format\n")
        f.write("==============================================================\n\n")
        f.write("Convention: min <C,X> s.t. <A_i,X>=b_i, X >= 0. matno=0 stores C.\n\n")
        f.write("Construction: X_*=diag(I_r,0_g,0_s), S_*=diag(0_r,0_g,I_s).\n")
        f.write("The constraints are an orthogonal mixing of a basis of S^{r+g}, with moderate\n")
        f.write("gap-to-range/slack cross couplings. There is no ill-conditioned scaling.\n\n")
        f.write("The optimum is unique. Primal-dual nondegeneracy holds at the optimum.\n")
        f.write("Strict complementarity fails because rank(X_*)+rank(S_*)=n-g<n.\n\n")
        for meta in metas:
            f.write(
                f"{meta['file']}: n={meta['n']}, m={meta['m']}, r={meta['r']}, "
                f"g={meta['g']}, s={meta['s']}, rank_sum={meta['rank_sum']}<n, "
                f"eps_gap={meta['eps_gap']}, nnz={meta['nonzero_entries']}\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, help="matrix size")
    ap.add_argument("--m", type=int, help="number of equality constraints; for default construction m must equal (r+g)(r+g+1)/2")
    ap.add_argument("--r", type=int, help="rank of X_star")
    ap.add_argument("--g", type=int, help="strict-complementarity gap dimension")
    ap.add_argument("--s", type=int, help="rank of S_star")
    ap.add_argument("--output", type=str, help="output .dat-s path")
    ap.add_argument("--outdir", type=str, default="pdnd_nosc_simple_dats")
    ap.add_argument("--all", action="store_true", help="generate the five requested instances")
    ap.add_argument("--eps-gap", type=float, default=0.25)
    ap.add_argument("--sweeps", type=int, default=2)
    ap.add_argument("--theta", type=float, default=0.35)
    args = ap.parse_args()

    if args.all:
        outdir = Path(args.outdir)
        outdir.mkdir(parents=True, exist_ok=True)
        metas: List[Dict] = []
        for n in [10, 100, 200, 500, 1000]:
            r, g, s, m = default_params(n)
            path = outdir / f"pdnd_nosc_simple_n{n}.dat-s"
            meta = generate_instance(n, m, r, g, s, path, eps_gap=args.eps_gap,
                                     sweeps=args.sweeps, theta=args.theta)
            metas.append(meta)
            print(f"wrote {path}  n={n} m={m} r={r} g={g} s={s}")
        with (outdir / "metadata.json").open("w") as f:
            json.dump(metas, f, indent=2)
        write_readme(outdir, metas)
        zip_path = outdir.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as z:
            for file in sorted(outdir.iterdir()):
                z.write(file, arcname=file.name)
        print(f"zip {zip_path}")
        return

    if None in [args.n, args.r, args.g, args.s]:
        ap.error("Either use --all or specify --n --r --g --s. If --m is omitted, m=q.")
    n, r, g, s = args.n, args.r, args.g, args.s
    p = r + g
    q = p * (p + 1) // 2
    m = args.m if args.m is not None else q
    output = Path(args.output) if args.output else Path(f"pdnd_nosc_simple_n{n}.dat-s")
    meta = generate_instance(n, m, r, g, s, output, eps_gap=args.eps_gap,
                             sweeps=args.sweeps, theta=args.theta)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
