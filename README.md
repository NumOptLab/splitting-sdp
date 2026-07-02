# pdhg-sdp

Dense first-order solvers for standard-form semidefinite programs

```text
minimize    <C, X>
subject to  A X = b
            X >= 0
```

with the dual slack convention `S = C - A^* y`.

## Install

```bash
uv sync --extra dev
```

Core dependencies are `numpy`, `scipy`, `h5py`, `pandas`, and `matplotlib`.

## CLI

Solve a dense SDPA sparse file:

```bash
python -m cli solve --input data/foo.dat-s --algorithm pdhg --tau 1e-3 --sigma 1e-3 --max-iters 1000 --tol 1e-6 --adapt none --init zero --period 100
python -m cli solve --input data/foo.dat-s --algorithm admm --rho 1.0 --max-iters 1000 --tol 1e-6 --adapt first-k --adapt-iters 100 --init random --seed 1 --period 100
```

Convert MATLAB data. The output is written next to the input file:

```bash
python -m cli convert-mat --input data/foo.mat
```

Plot and diagnose saved runs:

```bash
python -m cli plot --history results/..._history.csv
python -m cli diagnose --solution results/..._solution.h5 --check-nd true
```

Solver outputs are written under `results`: summary JSON, solution HDF5, history CSV, and convergence plots in both PNG and EPS.

## Conventions

- `--init zero` starts all algorithm variables at the origin.
- `--init random --seed N` uses symmetric Gaussian matrix initial iterates and Gaussian vector dual iterates.
- Solver progress tables are printed to stderr by default; use `--period N` to change the frequency or `--no-verbose` to suppress them.
- ADMM uses `--rho` as the public name for the paper penalty parameter `sigma`.
- PDHG uses `theta=1` by default.
- PDHG dual residual is the squared Frobenius distance from `S = C - A^* y` to the PSD cone, normalized by `1 + ||C||_F`.
- `.dat-s` positive SDPA blocks are assembled as one dense block-diagonal SDP. LP/negative blocks are rejected.
