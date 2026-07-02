"""Input/output for dense SDPA data and MATLAB conversion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy import io as scipy_io
from scipy import sparse

from core.psd import symmetrize

DEFAULT_MAX_DENSE_BYTES = 2 * 1024**3


@dataclass(frozen=True)
class SDPData:
    C: np.ndarray
    A: np.ndarray
    b: np.ndarray
    block_sizes: tuple[int, ...]
    problem_name: str = "problem"
    input_path: str | None = None
    input_format: str = "memory"

    @property
    def n(self) -> int:
        return int(self.C.shape[0])

    @property
    def m(self) -> int:
        return int(self.b.size)


@dataclass(frozen=True)
class SDPMetadata:
    m: int
    block_sizes: tuple[int, ...]
    n: int
    dense_bytes: int


def _clean_tokens(path: Path) -> list[str]:
    tokens: list[str] = []
    with path.open("r", encoding="utf-8", errors="ignore") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped or stripped.startswith(("*", '"', "#")):
                continue
            if '"' in stripped:
                stripped = stripped.split('"', 1)[0].strip()
            if stripped:
                tokens.extend(stripped.split())
    return tokens


def read_dat_s_metadata(path: str | Path) -> SDPMetadata:
    """Read only SDPA sparse header metadata."""
    path_obj = Path(path)
    tokens = _clean_tokens(path_obj)
    if len(tokens) < 3:
        raise ValueError(f"{path_obj} is not a valid SDPA sparse file.")
    cursor = 0
    m = int(float(tokens[cursor])); cursor += 1
    n_blocks = int(float(tokens[cursor])); cursor += 1
    block_sizes = tuple(int(float(tokens[cursor + i])) for i in range(n_blocks))
    if any(size <= 0 for size in block_sizes):
        raise ValueError("Only positive semidefinite SDPA blocks are supported.")
    n = int(sum(block_sizes))
    dense_bytes = (m + 1) * n * n * np.dtype(float).itemsize
    return SDPMetadata(m=m, block_sizes=block_sizes, n=n, dense_bytes=dense_bytes)


def _check_dense_size(metadata: SDPMetadata, max_dense_bytes: int | None) -> None:
    if max_dense_bytes is None:
        return
    if metadata.dense_bytes > max_dense_bytes:
        gb = metadata.dense_bytes / 1024**3
        limit = max_dense_bytes / 1024**3
        raise MemoryError(
            f"Dense data would require about {gb:.2f} GiB, exceeding the {limit:.2f} GiB guard."
        )


def read_dat_s(path: str | Path, *, max_dense_bytes: int | None = DEFAULT_MAX_DENSE_BYTES) -> SDPData:
    """Read an SDPA sparse ``.dat-s`` file into dense standard-form arrays."""
    path_obj = Path(path)
    metadata = read_dat_s_metadata(path_obj)
    _check_dense_size(metadata, max_dense_bytes)

    tokens = _clean_tokens(path_obj)
    cursor = 0
    m = int(float(tokens[cursor])); cursor += 1
    n_blocks = int(float(tokens[cursor])); cursor += 1
    block_sizes = tuple(int(float(tokens[cursor + i])) for i in range(n_blocks)); cursor += n_blocks
    b = np.asarray([float(tokens[cursor + i]) for i in range(m)], dtype=float); cursor += m

    offsets = np.cumsum((0,) + block_sizes)
    C = np.zeros((metadata.n, metadata.n), dtype=float)
    A = np.zeros((m, metadata.n, metadata.n), dtype=float)

    remaining = len(tokens) - cursor
    if remaining % 5 != 0:
        raise ValueError(f"{path_obj} has incomplete SDPA matrix entries.")

    while cursor < len(tokens):
        matno = int(float(tokens[cursor])); cursor += 1
        block = int(float(tokens[cursor])); cursor += 1
        row = int(float(tokens[cursor])); cursor += 1
        col = int(float(tokens[cursor])); cursor += 1
        value = float(tokens[cursor]); cursor += 1
        if block < 1 or block > n_blocks:
            raise ValueError(f"Invalid block index {block}.")
        if matno < 0 or matno > m:
            raise ValueError(f"Invalid matrix index {matno}.")
        base = int(offsets[block - 1])
        i = base + row - 1
        j = base + col - 1
        target = C if matno == 0 else A[matno - 1]
        target[i, j] += value
        if i != j:
            target[j, i] += value

    return SDPData(
        C=symmetrize(C),
        A=np.asarray([symmetrize(Ai) for Ai in A], dtype=float),
        b=b,
        block_sizes=block_sizes,
        problem_name=path_obj.stem.replace(".dat", ""),
        input_path=str(path_obj),
        input_format="dat-s",
    )


def _mat_vector_indices(n: int) -> Iterable[tuple[int, int]]:
    """MATLAB At/C vectors use packed upper-triangular svec order."""
    for j in range(n):
        for i in range(j + 1):
            yield i, j


def _triangular_n(length: int) -> int:
    n_float = (np.sqrt(8 * length + 1) - 1) / 2
    n = int(round(n_float))
    if n * (n + 1) // 2 != length:
        raise ValueError(f"Vector length {length} is not triangular.")
    return n


def write_dat_s(data: SDPData, path: str | Path, *, tol: float = 0.0) -> Path:
    """Write dense data to SDPA sparse format."""
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    with path_obj.open("w", encoding="utf-8") as handle:
        handle.write(f"{data.m}\n")
        handle.write(f"{len(data.block_sizes)}\n")
        handle.write(" ".join(str(size) for size in data.block_sizes) + "\n")
        handle.write(" ".join(f"{float(value):.17g}" for value in data.b) + "\n")
        matrices = [symmetrize(data.C)] + [symmetrize(Ai) for Ai in data.A]
        offsets = np.cumsum((0,) + data.block_sizes)
        for matno, matrix in enumerate(matrices):
            for block_idx, block_size in enumerate(data.block_sizes):
                base = int(offsets[block_idx])
                for j in range(block_size):
                    for i in range(j, block_size):
                        value = float(matrix[base + i, base + j])
                        if abs(value) <= tol:
                            continue
                        handle.write(f"{matno} {block_idx + 1} {i + 1} {j + 1} {value:.17g}\n")
    return path_obj


def _as_vector(value: object) -> np.ndarray:
    if sparse.issparse(value):
        return np.asarray(value.toarray(), dtype=float).reshape(-1)
    return np.asarray(value, dtype=float).reshape(-1)


def _write_vector_dat_s(C_vec: np.ndarray, At: object, b: np.ndarray, output_path: Path) -> Path:
    n = _triangular_n(int(C_vec.size))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    b_vec = np.asarray(b, dtype=float).reshape(-1)
    m = int(b_vec.size)
    indices = list(_mat_vector_indices(n))
    offdiag_scale = 1.0 / float(np.sqrt(2.0))
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(f"{m}\n")
        handle.write("1\n")
        handle.write(f"{n}\n")
        handle.write(" ".join(f"{float(value):.17g}" for value in b_vec) + "\n")
        for idx, value in enumerate(C_vec):
            if float(value) == 0.0:
                continue
            i, j = indices[idx]
            matrix_value = float(value) * (offdiag_scale if i != j else 1.0)
            row, col = max(i, j), min(i, j)
            handle.write(f"0 1 {row + 1} {col + 1} {matrix_value:.17g}\n")
        At_matrix = At if sparse.issparse(At) else np.asarray(At, dtype=float)
        if At_matrix.shape != (C_vec.size, m):
            raise ValueError(f"At must have shape {(C_vec.size, m)}, got {At_matrix.shape}.")
        if sparse.issparse(At_matrix):
            coo = At_matrix.tocoo()
            for idx, constraint, value in zip(coo.row, coo.col, coo.data):
                i, j = indices[int(idx)]
                matrix_value = float(value) * (offdiag_scale if i != j else 1.0)
                row, col = max(i, j), min(i, j)
                handle.write(f"{int(constraint) + 1} 1 {row + 1} {col + 1} {matrix_value:.17g}\n")
        else:
            rows, cols = np.nonzero(At_matrix)
            for idx, constraint in zip(rows, cols):
                value = float(At_matrix[idx, constraint])
                i, j = indices[int(idx)]
                matrix_value = value * (offdiag_scale if i != j else 1.0)
                row, col = max(i, j), min(i, j)
                handle.write(f"{int(constraint) + 1} 1 {row + 1} {col + 1} {matrix_value:.17g}\n")
    return output_path


def mat_to_dat_s(input_path: str | Path) -> Path:
    """Convert a supported MATLAB SDP file to ``.dat-s`` next to the input file."""
    path_obj = Path(input_path)
    raw = scipy_io.loadmat(path_obj, squeeze_me=False, struct_as_record=False)
    data = {key: value for key, value in raw.items() if not key.startswith("__")}
    output_path = path_obj.with_suffix(".dat-s")

    if {"C", "A", "b"}.issubset(data):
        C = np.asarray(data["C"], dtype=float)
        A_raw = np.asarray(data["A"], dtype=float)
        b = _as_vector(data["b"])
        if C.ndim != 2 or C.shape[0] != C.shape[1]:
            raise ValueError("Dense key C must be square.")
        n = C.shape[0]
        if A_raw.ndim != 3:
            raise ValueError("Dense key A must have shape (m,n,n) or (n,n,m).")
        if A_raw.shape[1:] == (n, n):
            A = A_raw
        elif A_raw.shape[:2] == (n, n):
            A = np.moveaxis(A_raw, 2, 0)
        else:
            raise ValueError("Dense key A must have shape (m,n,n) or (n,n,m).")
        dense = SDPData(C=symmetrize(C), A=np.asarray([symmetrize(Ai) for Ai in A]), b=b, block_sizes=(n,), problem_name=path_obj.stem)
        return write_dat_s(dense, output_path)

    if {"At", "C", "b"}.issubset(data):
        return _write_vector_dat_s(_as_vector(data["C"]), data["At"], _as_vector(data["b"]), output_path)

    keys = ", ".join(sorted(data))
    raise ValueError(f"Unsupported .mat structure. Expected C/A/b or At/b/C; found keys: {keys}.")
