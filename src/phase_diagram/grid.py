"""The query grid: one jittered point per cell, nested across four resolutions.

The T axis is linear and the pressure axis is logarithmic, as on the usual phase
diagram. Each cell holds one point drawn uniformly at random, then rounded to two
decimals and four significant figures, so round values appear only by chance. A draw
that lands inside the buffer of a boundary is redrawn within the same cell, so maps have
no holes and no point sits on a coexistence curve.

Levels nest: level 3 (16x16) is a stratified subset of level 2 (32x32), which is a
subset of level 1 (64x64), which is a subset of level 0 (128x128). Refining a model
from one level to the next therefore reuses every answer it already gave.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from . import truth

T_MIN, T_MAX = 170.0, 900.0  # K
LOGP_MIN, LOGP_MAX = -2.0, math.log10(2e10)  # 0.01 Pa to 20 GPa, within the ice VII data
N_FINE = 128
MAX_LEVEL = 3
SIDES = {k: N_FINE >> k for k in range(MAX_LEVEL + 1)}
SEED = 20261031
SIG_FIGS = 4
MAX_TRIES = 400

T_STEP = (T_MAX - T_MIN) / N_FINE
LOGP_STEP = (LOGP_MAX - LOGP_MIN) / N_FINE


def round_sig(x: float, sig: int = SIG_FIGS) -> float:
    return float(f"{x:.{sig - 1}e}")


def _draw_cell(rng: np.random.Generator, i: int, j: int) -> tuple[float, float] | None:
    for _ in range(MAX_TRIES):
        T = round(T_MIN + (i + rng.random()) * T_STEP, 2)
        p = round_sig(10 ** (LOGP_MIN + (j + rng.random()) * LOGP_STEP))
        if not truth.near_boundary(T, p)[0]:
            return T, p
    return None


def build_grid(seed: int = SEED) -> pd.DataFrame:
    """Return one row per query point, with `level` the coarsest level it belongs to."""
    rng = np.random.default_rng(seed)
    points = {}
    for i in range(N_FINE):
        for j in range(N_FINE):
            drawn = _draw_cell(rng, i, j)
            if drawn is not None:
                points[(i, j)] = drawn

    level = {cell: 0 for cell in points}
    members = set(points)
    for k in range(1, MAX_LEVEL + 1):
        blocks: dict[tuple[int, int], list[tuple[int, int]]] = {}
        for cell in sorted(members):
            blocks.setdefault((cell[0] >> k, cell[1] >> k), []).append(cell)
        members = set()
        for block_cells in blocks.values():
            chosen = block_cells[rng.integers(len(block_cells))]
            members.add(chosen)
            level[chosen] = k

    rows = [
        {
            "pid": i * N_FINE + j,
            "i": i,
            "j": j,
            "T_K": T,
            "p_Pa": p,
            "level": level[(i, j)],
        }
        for (i, j), (T, p) in sorted(points.items())
    ]
    df = pd.DataFrame(rows)
    df["truth"] = truth.phase(df["T_K"].to_numpy(), df["p_Pa"].to_numpy())
    return df


def at_level(grid: pd.DataFrame, level: int) -> pd.DataFrame:
    """Points that belong to `level` (a coarser point also belongs to finer levels)."""
    return grid[grid["level"] >= level]


def cell_edges(level: int = 0) -> tuple[np.ndarray, np.ndarray]:
    side = SIDES[level]
    return np.linspace(T_MIN, T_MAX, side + 1), np.linspace(LOGP_MIN, LOGP_MAX, side + 1)
