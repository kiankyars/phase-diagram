"""Grade each model's answers against IAPWS.

Four kinds of score, each answering a different question:

  accuracy         Is the map right?
  boundaries       Where does the model put each curve, relative to the real one?
  coherence        Does the map look like a phase diagram? These scores never see the
                   truth, which is what lets a tidy map be told apart from a right one.
  representation   Does the map survive a change of units, beyond the amount it changes
                   when the identical question is simply asked twice?

Balanced accuracy is the headline, because 61% of the grid is vapor and a model that
always answers "vapor" would otherwise look competent.

Boundaries are measured as offsets from the true curve at each point's own temperature
and pressure, so the jitter inside a grid cell is not mistaken for model error. Along
each isotherm or isobar, the model's boundary is the split that leaves the fewest of
its answers on the wrong side, which keeps one stray answer from moving a curve. Its
error is that split less the split a perfect map gets from the same points, so a
perfect map scores zero at any resolution.
"""

from __future__ import annotations

from functools import lru_cache
from itertools import chain, combinations, combinations_with_replacement
from math import comb

import numpy as np
import pandas as pd
from scipy import ndimage, special, stats

from . import grid, truth
from .framings import BY_KEY, CANONICAL, FRAMINGS
from .registry import Model
from .truth import LIQUID, P_C, P_IH_III, P_TP, SOLID, SUPERCRITICAL, T_C, T_TP, VAPOR

CODES = {name: code for code, name in enumerate(truth.CLASSES)}
INVALID = -1  # a reply with no usable phase word
MISSING = -2  # a point the model was not asked

# How dense each phase is, in order. In an ordinary phase diagram, more pressure never
# moves a substance down this order and more heat never moves it up. Water's ice Ih
# melts under pressure, a real exception that the truth map also scores against itself.
DENSITY = np.array([3, 2, 0, 1])  # solid, liquid, vapor, supercritical

# Everyday water, where a model can lean on remembered facts. Everything else on the
# map has to come from a picture of the whole diagram.
FAMILIAR_T = (250.0, 400.0)
FAMILIAR_P = (1e4, 1e7)

RETEST_FRAMINGS = (CANONICAL, "C_atm")
BOOTSTRAPS = 2000
PAIRED_RESAMPLES = 1_000_000  # the printed interval ends must not move with the seed
EXACT_MODELS = 13  # up to this many models, a trend's every distinct resample can be listed
TREND_RESAMPLES = 1_000_000  # beyond that the Theil-Sen bootstrap is lumpy, so it needs many draws
CHUNK = 20_000

MIN_CLASS_POINTS = 5  # fewer than this and one answer swings a balanced accuracy
REGION_REACH = 1 / 16  # how near three regions must come to meet, as a share of the map

CURVE_WINDOW = 5  # lines a drawn curve is smoothed over, so one line's noise is not drawn

ATMOSPHERE = 101325.0  # Pa
ATMOSPHERE_BOILING = 373.124  # K, from the IAPWS saturation curve
EVERYDAY_ROWS = 1.3  # grid rows either side of one atmosphere that a reading uses
MIN_LIQUID = 3  # liquid answers a boiling reading must rest on, or it only marks where ice ends
HOT_LIQUID_T = (400.0, 640.0)  # kelvin, where the true melting pressure climbs from 3 to 12 GPa

# The parts of the diagram, everyday ones first. A model can reach those with remembered
# facts; the rest take a picture of the whole diagram.
REGION_NAMES = {
    "ordinary_ice": "Ice",
    "ordinary_liquid": "Liquid",
    "ordinary_vapor": "Vapor",
    "cold_vapor": "Vapor below 0 °C",
    "high_pressure_ice": "High-pressure ice",
    "compressed_liquid": "Liquid above the critical pressure",
    "hot_vapor": "Vapor above the critical temperature",
    "supercritical": "Supercritical",
}
SUBLIMATION_HEAT = 51.0e3  # J/mol, near the triple point
GAS_CONSTANT = 8.314  # J/(mol K)


def sketch_phase(T, p) -> np.ndarray:
    """The diagram four remembered facts give, drawn with Clausius-Clapeyron.

    The facts are the triple point, the critical point, ice melting at 0 °C, and water's
    heat of sublimation. Liquid and vapor meet on a straight line in ln p against 1/T
    between the two points, ice ends at a vertical line, so there is no ice above 0 °C
    at any pressure. That layout and the textbook edges of supercritical are built in, so
    this marks what the facts are worth once the shape of the diagram is given.
    """
    T = np.asarray(T, dtype=float)
    p = np.asarray(p, dtype=float)
    to_critical = np.log(P_C / P_TP) / (1 / T_TP - 1 / T_C)
    p_saturation = P_TP * np.exp(to_critical * (1 / T_TP - 1 / T))
    p_sublimation = P_TP * np.exp(SUBLIMATION_HEAT / GAS_CONSTANT * (1 / T_TP - 1 / T))
    out = np.where(p < p_sublimation, VAPOR, SOLID)
    warm = (T >= 273.15) & (T < T_C)
    out = np.where(warm, np.where(p < p_saturation, VAPOR, LIQUID), out)
    return np.where(T >= T_C, np.where(p < P_C, VAPOR, SUPERCRITICAL), out)


def arm_for(model: Model, condition: str) -> str | None:
    """Which arm holds a model's answers under "instant" or "reasoned", if any does."""
    if condition == "instant":
        return None if model.thinks else "direct"
    if model.thinks:
        return "direct"
    return "reasoned" if model.reasoned else None


def attempt(answers: pd.DataFrame, points: pd.DataFrame, framing=CANONICAL, rep=0, arm="direct") -> pd.DataFrame:
    """One pass over the grid: the points asked, with the answer's code as `pred`."""
    chosen = answers[(answers["arm"] == arm) & (answers["framing"] == framing) & (answers["rep"] == rep)]
    frame = points.merge(chosen[["pid", "label", "status"]], on="pid")
    frame["pred"] = frame["label"].map(CODES).fillna(INVALID).astype(np.int8)
    return frame


def at_level(frame: pd.DataFrame, level: int) -> pd.DataFrame:
    return frame[frame["level"] >= level]


def to_image(frame: pd.DataFrame, level: int, column: str = "pred") -> np.ndarray:
    """Answers as an array indexed [temperature, pressure]."""
    side = grid.SIDES[level]
    image = np.full((side, side), MISSING, dtype=np.int8)
    chosen = at_level(frame, level)
    image[chosen["i"].to_numpy() >> level, chosen["j"].to_numpy() >> level] = chosen[column].to_numpy()
    return image


def complete_level(frame: pd.DataFrame) -> int | None:
    """The finest level this pass has fully answered, or None."""
    for level in sorted(grid.SIDES):
        if len(at_level(frame, level)) == grid.SIDES[level] ** 2:
            return level
    return None


def balanced_accuracy(true: np.ndarray, pred: np.ndarray) -> float:
    return float(np.mean([(pred[true == k] == k).mean() for k in np.unique(true)]))


def supported_ba(true: np.ndarray, pred: np.ndarray) -> float:
    """Balanced accuracy, or NaN where a phase has too few points to weigh a third of it."""
    counts = np.bincount(true, minlength=len(truth.CLASSES))
    return float("nan") if counts[counts > 0].min() < MIN_CLASS_POINTS else balanced_accuracy(true, pred)


def macro_f1(true: np.ndarray, pred: np.ndarray) -> float:
    scores = []
    for k in np.unique(true):
        hits = ((true == k) & (pred == k)).sum()
        scores.append(2 * hits / max((true == k).sum() + (pred == k).sum(), 1))
    return float(np.mean(scores))


def _melting_curve() -> tuple[np.ndarray, np.ndarray]:
    curves = truth.boundary_curves(4000)
    pieces = [curves[name] for name in ("melt_ih", "melt_iii", "melt_v", "melt_vi", "melt_vii")]
    T = np.concatenate([piece[0] for piece in pieces])
    logp = np.log10(np.concatenate([piece[1] for piece in pieces]))
    order = np.argsort(logp)
    return logp[order], T[order]


_MELT_LOGP, _MELT_T = _melting_curve()


def t_melt(p) -> np.ndarray:
    """Temperature of the solid's warm edge at pressure p, for p above the triple point."""
    return np.interp(np.log10(np.asarray(p, dtype=float)), _MELT_LOGP, _MELT_T)


_SATURATION_T = np.linspace(T_TP, T_C, 4000)
_SATURATION_LOGP = np.log10(truth.p_sat(_SATURATION_T))


def t_sat(p) -> np.ndarray:
    """Boiling temperature at pressure p, between the triple and critical points."""
    return np.interp(np.log10(np.asarray(p, dtype=float)), _SATURATION_LOGP, _SATURATION_T)


def best_split(offset: np.ndarray, below: np.ndarray) -> float:
    """The offset that leaves the fewest points on the wrong side of it.

    `below` says which points the answer places under the boundary. With no transition
    on the line, the split lands at the end of the line: the model's boundary is at or
    past the edge of the map, and the offset is a lower bound on how far.
    """
    if not len(offset):
        return float("nan")
    order = np.argsort(offset)
    offset, below = offset[order], below[order]
    above_it = np.concatenate([[0], np.cumsum(~below)])
    left_over = below.sum() - np.concatenate([[0], np.cumsum(below)])
    misplaced = above_it + left_over
    edges = np.concatenate([[offset[0]], (offset[:-1] + offset[1:]) / 2, [offset[-1]]])
    return float(np.median(edges[misplaced == misplaced.min()]))


def relative_split(offset: np.ndarray, below: np.ndarray) -> float:
    """A map's split less the split the true labels give on the same points."""
    return best_split(offset, below) - best_split(offset, offset < 0)


def _splits(line: np.ndarray, offset: np.ndarray, below: np.ndarray, side: int) -> np.ndarray:
    """A row of (grid line, offset, found) for each line that crosses the true curve.

    A line that only clips the end of a curve has a handful of points, too few to say
    where a boundary is, so it must hold at least half its points to count. `found` is
    0 where the answers never change side, so that the offset is only a lower bound.
    """
    out = []
    for k in np.unique(line):
        here = line == k
        if here.sum() >= side / 2 and offset[here].min() < 0 < offset[here].max():
            found = below[here].any() and not below[here].all()
            out.append((k, relative_split(offset[here], below[here]), found))
    return np.array(out, dtype=float).reshape(-1, 3)


def boundary_splits(frame: pd.DataFrame, level: int) -> dict[str, np.ndarray]:
    """Signed distance from each true curve to the model's, as rows from `_splits`.

    Offsets are in decades of pressure for the curves read along isotherms, and in
    kelvin for the melting curve, which is read along isobars.
    """
    chosen = at_level(frame, level)
    chosen = chosen[chosen["pred"] >= 0]
    T, p = chosen["T_K"].to_numpy(), chosen["p_Pa"].to_numpy()
    logp, pred = np.log10(p), chosen["pred"].to_numpy()
    isotherm, isobar = chosen["i"].to_numpy() >> level, chosen["j"].to_numpy() >> level
    vapor, solid = pred == VAPOR, pred == SOLID

    warm = (T > T_TP) & (T < T_C)
    cold = T < T_TP
    pressed = p > P_TP
    side = grid.SIDES[level]
    return {
        "saturation": _splits(isotherm[warm], logp[warm] - np.log10(truth.p_sat(T[warm])), vapor[warm], side),
        "sublimation": _splits(isotherm[cold], logp[cold] - np.log10(truth.p_subl(T[cold])), vapor[cold], side),
        "melting": _splits(isobar[pressed], T[pressed] - t_melt(p[pressed]), solid[pressed], side),
    }


def isobar_pressures(isobars: np.ndarray, level: int) -> np.ndarray:
    """The log10 pressure at the middle of each isobar, as numbered in `boundary_splits`."""
    return grid.LOGP_MIN + (isobars + 0.5) * (grid.LOGP_MAX - grid.LOGP_MIN) / grid.SIDES[level]


def boundary_lines(frame: pd.DataFrame, level: int) -> dict[str, np.ndarray]:
    """Where the model puts its boiling and melting curves, as rows of (T, log10 p).

    A line where the map has no boundary at all draws nothing.
    """
    side = grid.SIDES[level]
    splits = {name: rows[rows[:, 2] > 0] for name, rows in boundary_splits(frame, level).items()}

    def smoothed(offset: np.ndarray) -> np.ndarray:
        return pd.Series(offset).rolling(CURVE_WINDOW, center=True, min_periods=1).median().to_numpy()

    saturation_t = grid.T_MIN + (splits["saturation"][:, 0] + 0.5) * (grid.T_MAX - grid.T_MIN) / side
    melting_logp = isobar_pressures(splits["melting"][:, 0], level)
    return {
        "saturation": np.column_stack([saturation_t, np.log10(truth.p_sat(saturation_t)) + smoothed(splits["saturation"][:, 1])]),
        "melting": np.column_stack([t_melt(10**melting_logp) + smoothed(splits["melting"][:, 1]), melting_logp]),
    }


def everyday_points(frame: pd.DataFrame, level: int) -> tuple[float, float]:
    """Where a map has water boiling and freezing at one atmosphere, in kelvin.

    These are the two facts everyone knows, read from every point near one atmosphere
    and measured from the true curves, so a perfect map reads exactly right. Ice counts
    with liquid on the cold side of the boiling split. A boiling reading is NaN unless
    MIN_LIQUID liquid answers sit below that split, because otherwise it marks where the
    ice ends. A freezing reading is NaN where the map has no solid there, and both are
    NaN at the coarsest level, which holds too few points near one atmosphere to say.
    """
    if level == grid.MAX_LEVEL:
        return float("nan"), float("nan")
    chosen = at_level(frame, level)
    chosen = chosen[chosen["pred"] >= 0]
    band = EVERYDAY_ROWS * (grid.LOGP_MAX - grid.LOGP_MIN) / grid.SIDES[level]
    near = np.abs(np.log10(chosen["p_Pa"].to_numpy()) - np.log10(ATMOSPHERE)) <= band
    T, p, pred = chosen["T_K"].to_numpy()[near], chosen["p_Pa"].to_numpy()[near], chosen["pred"].to_numpy()[near]
    to_boil, cold = T - t_sat(p), np.isin(pred, [SOLID, LIQUID])
    if (pred[to_boil < best_split(to_boil, cold)] == LIQUID).sum() < MIN_LIQUID:
        boiling = float("nan")
    else:
        boiling = float(t_sat(ATMOSPHERE)) + relative_split(to_boil, cold)
    if not (pred == SOLID).any():
        freezing = float("nan")
    else:
        freezing = float(t_melt(ATMOSPHERE)) + relative_split(T - t_melt(p), pred == SOLID)
    return boiling, freezing


def ice_ceiling(frame: pd.DataFrame, level: int) -> float:
    """The pressure, in pascals, at which a map turns hot compressed liquid into ice.

    Read as the one pressure that best splits solid answers from the rest within
    HOT_LIQUID_T. A map with no ice there reads at the top of the map.
    """
    chosen = at_level(frame, level)
    chosen = chosen[chosen["pred"] >= 0]
    T, p, pred = chosen["T_K"].to_numpy(), chosen["p_Pa"].to_numpy(), chosen["pred"].to_numpy()
    hot = (T >= HOT_LIQUID_T[0]) & (T <= HOT_LIQUID_T[1]) & (p > P_C)
    return float(10 ** best_split(np.log10(p[hot]), pred[hot] != SOLID))


def meeting_point(image: np.ndarray, phases: tuple[int, int, int]) -> tuple[float, float]:
    """Where a map's regions of three phases come together, as (T, log10 p).

    Each phase is its largest connected region, so a stray answer cannot move the point,
    and the point is the cell nearest all three. Regions that never come within
    REGION_REACH of each other do not meet, which gives NaN.
    """
    side = image.shape[0]
    distances = []
    for phase in phases:
        labels, count = ndimage.label(image == phase)
        if not count:
            return float("nan"), float("nan")
        largest = np.bincount(labels.ravel())[1:].argmax() + 1
        distances.append(ndimage.distance_transform_edt(labels != largest))
    farthest = np.max(distances, axis=0)
    if farthest.min() > REGION_REACH * side:
        return float("nan"), float("nan")
    i, j = np.nonzero(farthest == farthest.min())
    T = grid.T_MIN + (np.median(i) + 0.5) * (grid.T_MAX - grid.T_MIN) / side
    logp = grid.LOGP_MIN + (np.median(j) + 0.5) * (grid.LOGP_MAX - grid.LOGP_MIN) / side
    return float(T), float(logp)


def coherence(image: np.ndarray) -> dict[str, float]:
    """Truth-free scores of how much a map looks like a phase diagram."""
    known = image >= 0
    rank = DENSITY[np.where(known, image, 0)]
    along_p = known[:, 1:] & known[:, :-1]
    along_t = known[1:, :] & known[:-1, :]
    rise_p = rank[:, 1:].astype(int) - rank[:, :-1]
    rise_t = rank[1:, :].astype(int) - rank[:-1, :]
    pairs = along_p.sum() + along_t.sum()
    ordered = ((rise_p[along_p] >= 0).sum() + (rise_t[along_t] <= 0).sum()) / pairs
    same = ((rise_p[along_p] == 0).sum() + (rise_t[along_t] == 0).sum()) / pairs
    changes = pairs * (1 - same)
    physical = ((rise_p[along_p] > 0).sum() + (rise_t[along_t] < 0).sum()) / changes if changes else float("nan")

    in_largest = 0
    for k in range(len(truth.CLASSES)):
        labels, count = ndimage.label(image == k)
        if count:
            in_largest += np.bincount(labels.ravel())[1:].max()
    return {
        "ordered": float(ordered),
        "same_neighbor": float(same),
        "physical_change": float(physical),
        "one_region": float(in_largest / known.sum()),
    }


def regions(T: np.ndarray, p: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Which points fall in each part of the diagram. Together they cover every point."""
    return {
        "ordinary_ice": (true == SOLID) & (p < P_IH_III),
        "ordinary_liquid": (true == LIQUID) & (p < P_C),
        "ordinary_vapor": (true == VAPOR) & (T >= T_TP) & (T < T_C),
        "cold_vapor": (true == VAPOR) & (T < T_TP),
        "high_pressure_ice": (true == SOLID) & (p >= P_IH_III),
        "compressed_liquid": (true == LIQUID) & (p >= P_C),
        "hot_vapor": (true == VAPOR) & (T >= T_C),
        "supercritical": true == SUPERCRITICAL,
    }


def tolerated(T: np.ndarray, p: np.ndarray, true: np.ndarray) -> dict[str, np.ndarray]:
    """Where the word supercritical could be forgiven, narrowest reading first.

    The fluid changes smoothly across the critical temperature and pressure lines, so
    beside those lines the word is a naming miss. The widest reading goes further and
    forgives it on dense liquid far from them, so it is a ceiling on what naming explains.
    """
    hot_gas = (true == VAPOR) & (T >= T_C)
    compressed = (true == LIQUID) & (p >= P_C)
    return {"hot_gas": hot_gas, "hot_fluid": hot_gas | (compressed & (T >= 0.9 * T_C)), "any_fluid": hot_gas | compressed}


def score_map(frame: pd.DataFrame, level: int) -> dict:
    """Every score for one pass over the grid, at one resolution."""
    chosen = at_level(frame, level)
    true, pred = chosen["truth"].to_numpy(), chosen["pred"].to_numpy()
    T, p = chosen["T_K"].to_numpy(), chosen["p_Pa"].to_numpy()
    familiar = (T >= FAMILIAR_T[0]) & (T <= FAMILIAR_T[1]) & (p >= FAMILIAR_P[0]) & (p <= FAMILIAR_P[1])
    out = {
        "level": level,
        "n": len(chosen),
        "invalid": float((pred < 0).mean()),
        "accuracy": float((pred == true).mean()),
        "balanced_accuracy": balanced_accuracy(true, pred),
        "macro_f1": macro_f1(true, pred),
        "ba_familiar": supported_ba(true[familiar], pred[familiar]),
        "ba_unfamiliar": balanced_accuracy(true[~familiar], pred[~familiar]),
    }
    for k, name in enumerate(truth.CLASSES):
        out[f"recall_{name}"] = float((pred[true == k] == k).mean())
        out[f"share_{name}"] = float((pred == k).mean())
        for m, said in enumerate(truth.CLASSES):
            out[f"confusion_{name}_{said}"] = int(((true == k) & (pred == m)).sum())
    for name, here in regions(T, p, true).items():
        out[f"wrong_{name}"] = float((pred[here] != true[here]).mean())
        out[f"misnamed_{name}"] = float(((pred[here] != true[here]) & (pred[here] != SUPERCRITICAL)).mean())
    for name, here in tolerated(T, p, true).items():
        out[f"ba_{name}"] = balanced_accuracy(true, np.where(here & (pred == SUPERCRITICAL), true, pred))
    splits = boundary_splits(frame, level)
    for name, rows in splits.items():
        out[f"{name}_bias"] = float(rows[:, 1].mean())
        out[f"{name}_mae"] = float(np.abs(rows[:, 1]).mean())
        out[f"{name}_found"] = float(rows[:, 2].mean())
    melting = splits["melting"]
    top = isobar_pressures(melting[:, 0], level) >= np.log10(P_IH_III)
    out["melting_bias_low"] = float(melting[~top, 1].mean()) if (~top).any() else float("nan")
    out["melting_bias_high"] = float(melting[top, 1].mean()) if top.any() else float("nan")
    out["boils_at"], out["freezes_at"] = everyday_points(frame, level)
    out["ice_ceiling"] = ice_ceiling(frame, level)
    image = to_image(frame, level)
    true_image = to_image(frame, level, "truth")
    for name, phases, known_T, known_p in (
        ("critical", (LIQUID, VAPOR, SUPERCRITICAL), T_C, P_C),
        ("triple", (SOLID, LIQUID, VAPOR), T_TP, P_TP),
    ):
        T_read, logp_read = meeting_point(image, phases)
        T_true, logp_true = meeting_point(true_image, phases)
        out[f"{name}_T_error"], out[f"{name}_logp_error"] = T_read - T_true, logp_read - logp_true
        out[f"{name}_T"] = known_T + out[f"{name}_T_error"]
        out[f"{name}_logp"] = np.log10(known_p) + out[f"{name}_logp_error"]
    return out | coherence(image)


def score_units(answers: pd.DataFrame, points: pd.DataFrame, level: int, arm: str = "direct", seed: int = 0) -> dict:
    """Agreement across framings against agreement of a prompt with itself."""
    chosen = points[points["level"] >= level]
    passes = {}
    for framing in FRAMINGS:
        frame = attempt(answers, chosen, framing.key, arm=arm).set_index("pid")["pred"]
        if len(frame) == len(chosen):
            passes[framing.key] = frame
    table = pd.DataFrame(passes).reindex(chosen["pid"])
    keys = list(table.columns)
    pairs = list(combinations(keys, 2))
    matrix = table.to_numpy()
    index = {key: k for k, key in enumerate(keys)}

    def compared(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """1 where two answers agree, 0 where they differ, NaN if either is unusable."""
        return np.where((a >= 0) & (b >= 0), (a == b).astype(float), np.nan)

    def kind(a: str, b: str) -> str:
        same_t = BY_KEY[a].t_unit == BY_KEY[b].t_unit
        same_p = BY_KEY[a].p_unit == BY_KEY[b].p_unit
        return "t_unit" if same_p else "p_unit" if same_t else "both"

    equal = np.stack([compared(matrix[:, index[a]], matrix[:, index[b]]) for a, b in pairs], axis=1)
    kinds = np.array([kind(a, b) for a, b in pairs])
    out = {"framings": len(keys), "cross": float(np.nanmean(equal))}
    for name in ("t_unit", "p_unit", "both"):
        out[f"cross_{name}"] = float(np.nanmean(equal[:, kinds == name]))

    repeats = []
    for framing in RETEST_FRAMINGS:
        second = attempt(answers, chosen, framing, rep=1, arm=arm).set_index("pid")["pred"]
        if len(second) == len(chosen):
            repeats.append(compared(table[framing].to_numpy(), second.reindex(chosen["pid"]).to_numpy()))
    if not repeats:
        return out
    repeats = np.stack(repeats, axis=1)
    out["retest"] = float(np.nanmean(repeats))
    out["gap"] = out["retest"] - out["cross"]

    rng = np.random.default_rng(seed)
    n = len(chosen)
    picks = rng.integers(n, size=(BOOTSTRAPS, n))
    gaps = np.nanmean(repeats[picks], axis=(1, 2)) - np.nanmean(equal[picks], axis=(1, 2))
    out["gap_low"], out["gap_high"] = (float(q) for q in np.quantile(gaps, [0.025, 0.975]))
    return out


def score_framings(answers: pd.DataFrame, points: pd.DataFrame, level: int, arm: str = "direct") -> list[dict]:
    """The same map scores in each notation, for the unit main effects."""
    rows = []
    for framing in FRAMINGS:
        frame = attempt(answers, points, framing.key, arm=arm)
        if len(at_level(frame, level)) == grid.SIDES[level] ** 2:
            rows.append({"framing": framing.key, "t_unit": framing.t_unit, "p_unit": framing.p_unit, **score_map(frame, level)})
    return rows


def boundary_distance(points: pd.DataFrame) -> pd.Series:
    """How far each grid point sits from any true boundary, as a share of the map's width."""
    image = to_image(points.assign(pred=points["truth"].astype(np.int8)), 0)
    inside = [ndimage.distance_transform_edt(image == k) for k in range(len(truth.CLASSES))]
    distance = np.max(inside, axis=0) / grid.N_FINE
    return pd.Series(distance[points["i"].to_numpy(), points["j"].to_numpy()], index=points.index)


def thought_focus(answers: pd.DataFrame, points: pd.DataFrame, arm: str, reach: float = 1 / 16) -> dict:
    """Whether a model that decides when to think spends that thought near the boundaries."""
    chosen = answers[(answers["arm"] == arm) & (answers["framing"] == CANONICAL) & (answers["rep"] == 0)]
    frame = points.assign(distance=boundary_distance(points)).merge(chosen[["pid", "label", "reasoning_tokens"]], on="pid")
    near = frame["distance"].to_numpy() <= reach
    thought = frame["reasoning_tokens"].to_numpy() > 0
    right = frame["label"].map(CODES).fillna(INVALID).to_numpy() == frame["truth"].to_numpy()
    return {
        "n": len(frame),
        "thought_near": float(thought[near].mean()),
        "thought_far": float(thought[~near].mean()),
        "right_when_thought": float(right[thought].mean()) if thought.any() else float("nan"),
        "right_when_not": float(right[~thought].mean()) if (~thought).any() else float("nan"),
    }


def shared_level(scores: pd.DataFrame) -> int:
    """The finest level every model has under every condition, so that all compare."""
    return int(scores.groupby(["condition", "model"])["level"].min().max())


def unit_level(units: pd.DataFrame, condition: str) -> int:
    """The finest level at which every model under a condition has its retest."""
    chosen = units[units["condition"] == condition].dropna(subset=["retest"])
    return int(chosen.groupby("model")["level"].min().max())


def paired_effect(first: pd.Series, second: pd.Series, seed: int = 0) -> dict:
    """The mean of first less second over the models that have both, with its interval."""
    diff = (first - second).dropna().to_numpy()
    rng = np.random.default_rng(seed)
    means = [diff[rng.integers(len(diff), size=(CHUNK, len(diff)))].mean(axis=1) for _ in range(PAIRED_RESAMPLES // CHUNK)]
    low, high = np.quantile(np.concatenate(means), [0.025, 0.975])
    return {"n": len(diff), "mean": float(diff.mean()), "low": float(low), "high": float(high), "first_higher": int((diff > 0).sum())}


def median_slopes(x: np.ndarray, y: np.ndarray, picks: np.ndarray) -> np.ndarray:
    """The Theil-Sen slope of each resample, NaN for any with no spread in x."""
    first, second = np.triu_indices(picks.shape[1], 1)
    run = x[picks[:, second]] - x[picks[:, first]]
    rise = y[picks[:, second]] - y[picks[:, first]]
    usable = run != 0
    slopes = np.sort(np.where(usable, rise / np.where(usable, run, 1.0), np.nan), axis=1)
    count = usable.sum(axis=1)
    rows = np.nonzero(count)[0]
    out = np.full(len(picks), np.nan)
    out[rows] = (slopes[rows, (count[rows] - 1) // 2] + slopes[rows, count[rows] // 2]) / 2
    return out


@lru_cache(maxsize=None)
def every_resample(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Every distinct bootstrap resample of n items, as sorted rows, and each one's probability."""
    total = comb(2 * n - 1, n)
    picks = np.fromiter(chain.from_iterable(combinations_with_replacement(range(n), n)), dtype=np.int8, count=total * n).reshape(total, n)
    counts = np.stack([(picks == k).sum(axis=1) for k in range(n)], axis=1)
    arrangements = special.gammaln(n + 1) - special.gammaln(counts + 1).sum(axis=1)
    return picks, np.exp(arrangements - n * np.log(n))


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    """The smallest value whose cumulative weight reaches q."""
    order = np.argsort(values)
    running = np.cumsum(weights[order])
    return float(values[order][np.searchsorted(running, q * running[-1])])


def trend_slopes(x: np.ndarray, y: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap slopes and their weights: exact where the resamples can be listed, so no seed matters."""
    n = len(y)
    if n <= EXACT_MODELS:
        picks, weights = every_resample(n)
        slopes = [median_slopes(x, y, picks[start:start + CHUNK]) for start in range(0, len(picks), CHUNK)]
    else:
        rng = np.random.default_rng(seed)
        slopes = [median_slopes(x, y, rng.integers(n, size=(CHUNK, n))) for _ in range(TREND_RESAMPLES // CHUNK)]
        weights = np.ones(sum(len(part) for part in slopes))
    slopes = np.concatenate(slopes)
    known = ~np.isnan(slopes)
    return slopes[known], weights[known]


def trend(dates: pd.Series, values: pd.Series, seed: int = 0) -> dict:
    """Rank correlation with release date, and the median change per year with its interval."""
    keep = values.notna().to_numpy()
    years = (pd.to_datetime(dates[keep]) - pd.Timestamp("2024-01-01")).dt.days.to_numpy() / 365.25
    y = values[keep].to_numpy()
    if len(y) < 3 or np.ptp(y) == 0:
        return {key: float("nan") for key in ("spearman", "p", "per_year", "per_year_low", "per_year_high")} | {"n": len(y)}
    rho, p = stats.spearmanr(years, y)
    slopes, weights = trend_slopes(years, y, seed)
    return {
        "n": len(y),
        "spearman": float(rho),
        "p": float(p),
        "per_year": float(stats.theilslopes(y, years)[0]),
        "per_year_low": weighted_quantile(slopes, weights, 0.025),
        "per_year_high": weighted_quantile(slopes, weights, 0.975),
    }
