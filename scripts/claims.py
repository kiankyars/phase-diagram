"""Recompute the scores, trends and counts the results page and README quote, and write results/claims.json.

The text is not generated from the data, so this is what keeps it honest. Run it after grading, and
any number in the text that no longer matches its entry here has gone stale.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
from scipy import optimize, stats

from phase_diagram import grade, grid, truth
from phase_diagram.framings import CANONICAL, FRAMINGS, P_SCALE, format_number
from phase_diagram.pipeline import ROOT, load_answers, load_grid, spent
from phase_diagram.registry import BY_ID, MODELS
from phase_diagram.truth import LIQUID, P_C, SOLID, SUPERCRITICAL, T_C, VAPOR

RESULTS = ROOT / "results"
GPA = 1e9
CLOSE_K = 15  # how near the true boiling point counts as within reach
SKETCH_DRAWS = 2000  # resamples of the grid points when comparing a map with the sketch
FINE_DRAWS = 20_000  # the same on a 128 × 128 map, where the interval is quoted to three places
DRAW_CHUNK = 500
STEP_DRAWS = 20_000  # resamples of the points either side of 1.000e6 pascals
STEP_BAND = 0.3  # decades either side of that pressure
STEP_NARROW_BAND = 0.15  # the tighter window the text quotes for megapascals
STEP_LARGE = 0.24  # the smallest extra fall the text calls large, the fourth largest at the switch
PLACEBO_CUTS = (4.2, 4.5, 4.8, 5.1, 5.4, 5.7, 6.3, 6.6, 6.9)  # log10 pascals, where no window crosses a change of form
FAN_TEMPERATURES = (400, 600)
RANDOM_MAPS = 50
CONTRAST_DRAWS = 1_000_000  # pairs of bootstrap slopes, one from each provider
NAMED_FLUID_T = 0.9 * T_C  # the temperature the Scoring text calls beside the critical point
NOTATION_TRENDS = ("balanced_accuracy", "saturation_mae", "melting_mae")


BY_LABEL = {m.label: m for m in MODELS}


def rounded(value):
    if isinstance(value, (float, np.floating)):
        return None if np.isnan(value) else round(float(value), 4)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, dict):
        return {k: rounded(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [rounded(v) for v in value]
    return value


def extent(series: pd.Series) -> dict:
    """The lowest and highest value of a series, each with its model, and the median."""
    series = series.dropna()
    return {
        "n": len(series),
        "low": series.min(),
        "low_model": series.idxmin(),
        "high": series.max(),
        "high_model": series.idxmax(),
        "median": series.median(),
    }


class Tables:
    """The graded tables, indexed the way the text refers to them."""

    def __init__(self) -> None:
        self.scores = pd.read_csv(RESULTS / "scores.csv")
        self.references = pd.read_csv(RESULTS / "references.csv")
        self.units = pd.read_csv(RESULTS / "units.csv")
        self.framings = pd.read_csv(RESULTS / "framings.csv")
        self.notation = pd.read_csv(RESULTS / "notation.csv")
        self.trends = pd.read_csv(RESULTS / "trends.csv")
        self.thought = pd.read_csv(RESULTS / "thought.csv")
        self.shared = grade.shared_level(self.scores)
        self.unit_levels = {c: grade.unit_level(self.units, c) for c in ("instant", "reasoned")}
        self.labels = {m.id: m.label for m in MODELS}

    def at(self, condition: str, level: int | None = None) -> pd.DataFrame:
        level = self.shared if level is None else level
        chosen = self.scores[(self.scores["condition"] == condition) & (self.scores["level"] == level)]
        return chosen.set_index("label")

    def reference(self, name: str, level: int | None = None) -> pd.Series:
        level = self.shared if level is None else level
        chosen = self.references[(self.references["reference"] == name) & (self.references["level"] == level)]
        return chosen.iloc[0]

    def paired(self) -> list[str]:
        """Models with a map under both conditions."""
        return sorted(set(self.at("instant").index) & set(self.at("reasoned").index))

    def fit(self, condition: str, provider: str, metric: str, span: str = "all", framing: str = CANONICAL, level: int | None = None) -> dict:
        level = self.shared if level is None else level
        t = self.trends
        row = t[
            (t["condition"] == condition) & (t["provider"] == provider) & (t["metric"] == metric)
            & (t["span"] == span) & (t["framing"] == framing) & (t["level"] == level)
        ].iloc[0]
        return {"per_year": row["per_year"], "low": row["per_year_low"], "high": row["per_year_high"], "n": row["n"], "spearman": row["spearman"], "p": row["p"]}


def grid_shares(t: Tables, points: pd.DataFrame) -> dict:
    chosen = grade.at_level(points, t.shared)
    T, p = chosen["T_K"].to_numpy(), chosen["p_Pa"].to_numpy()
    true = chosen["truth"].to_numpy()
    coarse = grade.at_level(points, t.unit_levels["instant"])
    familiar = coarse["T_K"].between(*grade.FAMILIAR_T) & coarse["p_Pa"].between(*grade.FAMILIAR_P)
    return {
        "vapor": float((true == VAPOR).mean()),
        "vapor_above_tc": float(((true == VAPOR) & (T >= T_C)).mean()),
        "points": len(chosen),
        "weight_set_by_naming": float(np.mean([((true == LIQUID) & (p >= P_C)).sum() / (true == LIQUID).sum(), ((true == VAPOR) & (T >= T_C)).sum() / (true == VAPOR).sum(), 1, 0])),
        "familiar_points_at_units_grid": int(familiar.sum()),
        "temperature_cell_k": (grid.T_MAX - grid.T_MIN) / grid.SIDES[t.shared],
    }


def headline(t: Tables) -> dict:
    instant, reasoned = t.at("instant"), t.at("reasoned")
    sketch = t.reference("Four-fact sketch")
    paired = t.paired()
    gain = reasoned.loc[paired, "balanced_accuracy"] - instant.loc[paired, "balanced_accuracy"]
    providers = instant.loc[paired, "provider"]
    beat = reasoned.loc[paired, "balanced_accuracy"] > sketch["balanced_accuracy"]
    allowed = reasoned["balanced_accuracy"]
    return {
        "models_instant": len(instant),
        "models_reasoned": len(reasoned),
        "models_paired": len(paired),
        "instant_ba": extent(instant["balanced_accuracy"]),
        "reasoned_ba": extent(allowed),
        "sketch_ba": sketch["balanced_accuracy"],
        "sketch_ba_familiar": sketch["ba_familiar"],
        "sketch_ba_unfamiliar": sketch["ba_unfamiliar"],
        "sketch_melting_mae": sketch["melting_mae"],
        "sketch_saturation_mae": sketch["saturation_mae"],
        "sketch_wrong_high_pressure_ice": sketch["wrong_high_pressure_ice"],
        "sketch_boils_at": sketch["boils_at"],
        "reasoned_beat_sketch": int((allowed > sketch["balanced_accuracy"]).sum()),
        "reasoned_beat_sketch_names": allowed[allowed > sketch["balanced_accuracy"]].sort_values().index.tolist(),
        "paired_beat_sketch": {"all": int(beat.sum()), "anthropic": int((beat & (providers == "anthropic")).sum()), "openai": int((beat & (providers == "openai")).sum()),
                               "anthropic_of": int((providers == "anthropic").sum()), "openai_of": int((providers == "openai").sum())},
        "paired_reach_090": int((reasoned.loc[paired, "balanced_accuracy"] >= 0.9).sum()),
        "near_sketch": allowed[(allowed - sketch["balanced_accuracy"]).abs() < 0.008].sort_values().to_dict(),
        "gain": extent(gain),
        "gain_openai_median": gain[providers == "openai"].median(),
        "gain_anthropic_median": gain[providers == "anthropic"].median(),
        "instant_better_than_sketch_on_melting": instant.index[instant["melting_mae"] < sketch["melting_mae"]].tolist(),
        "instant_better_than_sketch_on_high_ice": int((instant["wrong_high_pressure_ice"] < sketch["wrong_high_pressure_ice"]).sum()),
        "instant_better_than_sketch_on_boiling": int((instant["saturation_mae"] < sketch["saturation_mae"]).sum()),
    }


def random_map_order(level: int) -> float:
    """What the physical-order score gives on maps of random words."""
    rng = np.random.default_rng(0)
    side = grid.SIDES[level]
    return float(np.mean([grade.coherence(rng.integers(len(truth.CLASSES), size=(side, side)))["physical_change"] for _ in range(RANDOM_MAPS)]))


def stray_steps(image: np.ndarray) -> float:
    """The share of a map's unphysical steps that touch a cell none of its four neighbors matches."""
    padded = np.pad(image, 1, constant_values=-1)
    stray = ~((padded[2:, 1:-1] == image) | (padded[:-2, 1:-1] == image) | (padded[1:-1, 2:] == image) | (padded[1:-1, :-2] == image))
    known = image >= 0
    rank = grade.DENSITY[np.where(known, image, 0)].astype(int)
    along_p = known[:, 1:] & known[:, :-1] & (rank[:, 1:] < rank[:, :-1])
    along_t = known[1:, :] & known[:-1, :] & (rank[1:, :] > rank[:-1, :])
    touching = (along_p & (stray[:, 1:] | stray[:, :-1])).sum() + (along_t & (stray[1:, :] | stray[:-1, :])).sum()
    return float(touching / (along_p.sum() + along_t.sum()))


def stray_order(t: Tables, points: pd.DataFrame) -> dict:
    shares = {model.label: stray_steps(grade.to_image(frame, t.shared)) for model, frame in frames("instant", t.shared, points)}
    return extent(pd.Series(shares))


def tidiness(t: Tables) -> dict:
    instant, reasoned = t.at("instant"), t.at("reasoned")
    true = t.reference("IAPWS")
    return {
        "instant_one_region": extent(instant["one_region"]),
        "reasoned_one_region": extent(reasoned["one_region"]),
        "truth_one_region": true["one_region"],
        "instant_physical_change": extent(instant["physical_change"]),
        "random_map_physical_change": random_map_order(t.shared),
        "reasoned_physical_change": extent(reasoned["physical_change"]),
        "truth_physical_change": true["physical_change"],
        "truth_physical_change_by_side": {str(grid.SIDES[level]): t.reference("IAPWS", level)["physical_change"] for level in grid.SIDES},
        "instant_ordered": extent(instant["ordered"]),
        "truth_ordered": true["ordered"],
        "instant_same_neighbor": extent(instant["same_neighbor"]),
        "share_says_liquid": instant["share_liquid"].sort_values().head(7).to_dict(),
        "recall_liquid": instant["recall_liquid"].sort_values().head(7).to_dict(),
        "truth_share_liquid": true["share_liquid"],
        "ordinary_liquid_right": (1 - instant["wrong_ordinary_liquid"]).sort_values().head(8).to_dict(),
        "maps_missing_most_ordinary_liquid": int((instant["wrong_ordinary_liquid"] > 0.5).sum()),
    }


def notations(t: Tables) -> dict:
    level = t.unit_levels["instant"]
    chosen = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == level)]
    table = chosen.pivot(index="label", columns="framing", values="balanced_accuracy")
    sketch = t.reference("Four-fact sketch", level)["balanced_accuracy"]
    mean = table.mean(axis=1)
    best = table.stack()
    canonical_top = table[CANONICAL].max()
    return {
        "level": level,
        "sketch_ba": sketch,
        "maps": int(table.notna().sum().sum()),
        "best_map": list(best.idxmax()) + [best.max()],
        "worst_map": list(best.idxmin()) + [best.min()],
        "maps_above_shared_best": int((table > t.at("instant")["balanced_accuracy"].max()).sum().sum()),
        "models_above_shared_best": int(((table > t.at("instant")["balanced_accuracy"].max()).any(axis=1)).sum()),
        "maps_at_or_above_sketch": int((table >= sketch).sum().sum()),
        "best_of_six_notation_mean": [mean.idxmax(), mean.max()],
        "luna_six_notation_mean": mean.get("GPT-6 Luna"),
        "canonical_best_at_this_level": canonical_top,
        "spread_widest": [(table.max(axis=1) - table.min(axis=1)).idxmax(), (table.max(axis=1) - table.min(axis=1)).max()],
    }


def naming(t: Tables) -> dict:
    instant, reasoned = t.at("instant"), t.at("reasoned")
    sketch = t.reference("Four-fact sketch")
    paired = t.paired()
    rows = {"strict": "balanced_accuracy", "hot_gas": "ba_hot_gas", "hot_fluid": "ba_hot_fluid", "any_fluid": "ba_any_fluid"}
    out = {}
    for name, column in rows.items():
        gain = reasoned.loc[paired, column] - instant.loc[paired, column]
        out[name] = {
            "best_instant": instant[column].max(),
            "best_instant_model": instant[column].idxmax(),
            "median_gain": gain.median(),
            "sketch": sketch[column],
            "instant_beating_sketch": int((instant[column] > sketch[column]).sum()),
        }
    gains = pd.DataFrame({name: reasoned.loc[paired, column] - instant.loc[paired, column] for name, column in rows.items()})
    out["largest_gain"] = {name: [gains[name].idxmax(), gains[name].max()] for name in rows}
    out["haiku_5_5_gain"] = gains.loc["Haiku 5.5"].to_dict()
    out["worst_instant_any_fluid"] = instant["ba_any_fluid"].sort_values().head(6).to_dict()
    out["sonnet_5_instant"] = {name: instant.loc["Sonnet 5", column] for name, column in rows.items()}
    out["sonnet_5_reasoned"] = {name: reasoned.loc["Sonnet 5", column] for name, column in rows.items()}
    out["sonnet_5_instant_wrong_hot_vapor"] = instant.loc["Sonnet 5", "wrong_hot_vapor"]
    out["sonnet_5_reasoned_wrong_hot_vapor"] = reasoned.loc["Sonnet 5", "wrong_hot_vapor"]
    out["sonnet_5_hot_gas_gain_rank_from_smallest"] = int(gains["hot_gas"].rank()["Sonnet 5"])
    out["smallest_hot_gas_gain"] = [gains["hot_gas"].idxmin(), gains["hot_gas"].min()]
    vapor_change = reasoned.loc[paired, "recall_vapor"] - instant.loc[paired, "recall_vapor"]
    out["vapor_recall_falls"] = vapor_change[vapor_change < 0].to_dict()
    out["vapor_recall_next"] = vapor_change.sort_values().head(2).to_dict()
    return out


def boundaries(t: Tables) -> dict:
    instant, reasoned = t.at("instant"), t.at("reasoned")
    paired = t.paired()
    ceiling = instant["ice_ceiling"] / GPA
    below = ceiling[ceiling < 1]
    released = pd.to_datetime(instant["released"])
    before_2026 = released < "2026-01-01"
    level = t.unit_levels["instant"]
    other = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == level)]
    ceilings = other.pivot(index="label", columns="framing", values="ice_ceiling") / GPA
    counts = (ceilings < 1).sum()
    allowed = reasoned["ice_ceiling"].reindex(below.index).dropna() / GPA
    true_ceiling = t.reference("IAPWS")["ice_ceiling"] / GPA
    saturation = other.pivot(index="label", columns="framing", values="saturation_mae")
    finer = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == t.shared)].pivot(index="label", columns="framing", values="saturation_mae")
    return {
        "saturation_mae_instant": extent(instant["saturation_mae"]),
        "saturation_mae_reasoned": extent(reasoned["saturation_mae"]),
        "saturation_picks": instant.loc[["Sonnet 4.5", "GPT-4o mini", "Sonnet 5.5", "GPT-5.5", "GPT-6 Sol"], "saturation_mae"].to_dict(),
        "saturation_across_notations": {name: [saturation.loc[name].min(), saturation.loc[name].max()] for name in ("Sonnet 4.5", "GPT-6 Sol")},
        "saturation_across_notations_finer": {name: finer.loc[name].to_dict() for name in ("Sonnet 4.5", "Sonnet 4.6")},
        "sonnet_4_6_months_after_4_5": (released["Sonnet 4.6"] - released["Sonnet 4.5"]).days / 30.44,
        "melting_mae_instant": extent(instant["melting_mae"]),
        "melting_mae_reasoned": extent(reasoned["melting_mae"]),
        "saturation_bias": instant["saturation_bias"].to_dict(),
        "melting_bias": instant.loc[["GPT-4o mini", "GPT-4o", "GPT-4.1", "GPT-5.4", "GPT-5.6 Sol", "GPT-6 Sol"], "melting_bias"].to_dict(),
        "melting_mae_top_found_share": instant.sort_values("melting_mae", ascending=False)["melting_found"].head(3).to_dict(),
        "paired_saturation_medians": [instant.loc[paired, "saturation_mae"].median(), reasoned.loc[paired, "saturation_mae"].median()],
        "reasoned_at_or_below_best_instant_saturation": int((reasoned["saturation_mae"] <= instant["saturation_mae"].min()).sum()),
        "paired_melting_medians": [instant.loc[paired, "melting_mae"].median(), reasoned.loc[paired, "melting_mae"].median()],
        "paired_melting_improved": int((reasoned.loc[paired, "melting_mae"] < instant.loc[paired, "melting_mae"]).sum()),
        "ice_ceiling_below_1_gpa": {"count": len(below), "of": len(ceiling), "in_2026": int((~before_2026.loc[below.index]).sum()), "released_2026": int((~before_2026).sum()),
                                    "before_2026": int(before_2026.loc[below.index].sum()), "released_before_2026": int(before_2026.sum()), "names": below.sort_values().index.tolist()},
        "ice_ceiling_counts_by_notation": counts.to_dict(),
        "ice_ceiling_below_in_all_six": ceilings.index[(ceilings < 1).all(axis=1)].tolist(),
        "ice_ceiling_true_gpa": true_ceiling,
        "ice_ceiling_reasoned_of_those": allowed.round(2).to_dict(),
        "ice_ceiling_reasoned_above_1": int((allowed > 1).sum()),
        "ice_ceiling_reasoned_near_truth": int(((allowed > 5) & (allowed < 7)).sum()),
        "ice_ceiling_reasoned_have_arm": len(allowed),
    }


def melting_parts(t: Tables, points: pd.DataFrame) -> dict:
    """The melting error split at the Ih-III pressure, which separates two different stories."""
    rows = {}
    for model, frame in frames("instant", t.shared, points):
        split = grade.boundary_splits(frame, t.shared)["melting"]
        logp = grid.LOGP_MIN + (split[:, 0] + 0.5) * (grid.LOGP_MAX - grid.LOGP_MIN) / grid.SIDES[t.shared]
        low = logp < np.log10(truth.P_IH_III)
        rows[model.label] = {"below": np.abs(split[low, 1]).mean(), "above": np.abs(split[~low, 1]).mean(), "n_below": int(low.sum()), "n_above": int((~low).sum()),
                             "found": float(split[:, 2].mean()), "unfound_above": int((split[~low, 2] == 0).sum()), "provider": model.provider, "released": model.released}
    table = pd.DataFrame(rows).T
    limit = 10.0**grid.LOGP_MAX
    size_below = {}
    for provider, part in table.groupby("provider"):
        fitted = grade.trend(part["released"].reset_index(drop=True), part["below"].astype(float).reset_index(drop=True))
        size_below[provider] = {"per_year": fitted["per_year"], "low": fitted["per_year_low"], "high": fitted["per_year_high"]}
    return {
        "size_below_trend": size_below,
        "unfound_above_by_model": table.loc[table["unfound_above"] > 0, "unfound_above"].astype(int).to_dict(),
        "curve_leaves_map_at_k": optimize.brentq(lambda T: float(truth.p_melt_high(np.array([T]))[0]) - limit, truth.T_VI_VII, truth.T_VII_MAX),
        "isobars_below": int(table["n_below"].iloc[0]),
        "isobars_above": int(table["n_above"].iloc[0]),
        "below": extent(table["below"].astype(float)),
        "above": extent(table["above"].astype(float)),
        "unfound_models": table.index[table["found"] < 1].tolist(),
    }


def melting_tilt(t: Tables) -> dict:
    """OpenAI's signed melting error either side of the Ih-III pressure, oldest model first."""
    rows = t.at("instant").reset_index()
    rows = rows[rows["provider"] == "openai"].sort_values("released").reset_index(drop=True)
    return {
        "by_model": rows.set_index("label")[["melting_bias", "melting_bias_low", "melting_bias_high"]].to_dict("index"),
        "low": t.fit("instant", "openai", "melting_bias_low"),
        "high": t.fit("instant", "openai", "melting_bias_high"),
        "low_shared_span": t.fit("instant", "openai", "melting_bias_low", span="shared"),
        "high_shared_span": t.fit("instant", "openai", "melting_bias_high", span="shared"),
        "low_lost_without_one": lost_without_one(rows, "melting_bias_low"),
    }


def lost_without_one(rows: pd.DataFrame, metric: str) -> list[str]:
    """The models whose removal brings zero back into a trend's interval."""
    lost = []
    for name in rows["label"]:
        keep = rows[rows["label"] != name].reset_index(drop=True)
        fitted = grade.trend(keep["released"], keep[metric])
        if fitted["per_year_low"] <= 0 <= fitted["per_year_high"]:
            lost.append(name)
    return lost


def melting_matches(t: Tables, points: pd.DataFrame) -> dict:
    """Isobars where a reasoned map's melting split lands exactly on the true one, and who came closest."""
    reasoned = t.at("reasoned")
    rows = {}
    for model, frame in frames("reasoned", t.shared, points):
        split = grade.boundary_splits(frame, t.shared)["melting"]
        top = grade.isobar_pressures(split[:, 0], t.shared) >= np.log10(truth.P_IH_III)
        rows[model.label] = {"isobars": len(split), "exact": int((split[:, 1] == 0).sum()), "within_a_cell": int((np.abs(split[:, 1]) <= (grid.T_MAX - grid.T_MIN) / grid.SIDES[t.shared]).sum()),
                             "isobars_above": int(top.sum()), "exact_above": int((split[top, 1] == 0).sum())}
    closest = reasoned["melting_mae"].sort_values().head(10)
    released = reasoned["released"]
    return {
        "isobars": rows,
        "maps": len(reasoned),
        "maps_from_2026": int((released >= "2026-01-01").sum()),
        "closest": {name: [reasoned.loc[name, "melting_mae"], reasoned.loc[name, "released"]] for name in closest.index},
        "first_before_2026": reasoned.loc[reasoned["released"] < "2026-01-01", "melting_mae"].idxmin(),
        "rank_of_first_before_2026": int(reasoned["melting_mae"].rank()[reasoned["released"] < "2026-01-01"].min()),
    }


def frames(condition: str, level: int, points: pd.DataFrame):
    for model in sorted(MODELS, key=lambda m: m.released):
        arm = grade.arm_for(model, condition)
        answers = load_answers(model)
        if arm is None or not len(answers):
            continue
        frame = grade.attempt(answers, points, arm=arm)
        finest = grade.complete_level(frame)
        if finest is not None and finest <= level:
            yield model, frame


def boiling_reach(t: Tables) -> dict:
    instant, reasoned = t.at("instant"), t.at("reasoned")
    paired = t.paired()
    off = lambda frame: (frame["boils_at"] - grade.ATMOSPHERE_BOILING).abs()
    coarse = t.at("instant", t.unit_levels["instant"])
    readings = off(instant.loc[paired]).dropna()
    return {
        "paired_instant_within": int((readings <= CLOSE_K).sum()),
        "paired_instant_with_reading": len(readings),
        "paired_reasoned_within": int((off(reasoned.loc[paired]) <= CLOSE_K).sum()),
        "all_instant_within": int((off(instant).dropna() <= CLOSE_K).sum()),
        "all_instant_with_reading": int(off(instant).notna().sum()),
        "all_instant_within_at_32": int((off(coarse).dropna() <= CLOSE_K).sum()),
        "all_instant_with_reading_at_32": int(off(coarse).notna().sum()),
        "all_reasoned_within": int((off(reasoned).dropna() <= CLOSE_K).sum()),
        "all_reasoned_with_reading": int(off(reasoned).notna().sum()),
    }


def trends(t: Tables) -> dict:
    out = {}
    for provider in ("openai", "anthropic"):
        for metric in ("balanced_accuracy", "saturation_mae", "melting_mae"):
            out[f"{provider}_instant_{metric}"] = t.fit("instant", provider, metric)
            out[f"{provider}_instant_{metric}_shared_span"] = t.fit("instant", provider, metric, span="shared")
            out[f"{provider}_reasoned_{metric}"] = t.fit("reasoned", provider, metric)
    level = t.unit_levels["instant"]
    for provider in ("openai", "anthropic"):
        for metric in ("balanced_accuracy", "melting_mae"):
            out[f"{provider}_{metric}_by_notation"] = {f.key: t.fit("instant", provider, metric, framing=f.key, level=level)["per_year"] for f in FRAMINGS}
    slope = out["openai_instant_balanced_accuracy"]
    gain = t.at("reasoned").loc[t.paired(), "balanced_accuracy"] - t.at("instant").loc[t.paired(), "balanced_accuracy"]
    dates = pd.to_datetime(t.at("instant").query("provider == 'openai'")["released"])
    out["years_of_openai_trend_for_the_gain"] = {
        "at_slope": gain.median() / slope["per_year"],
        "at_interval_ends": [gain.median() / slope["high"], gain.median() / slope["low"]],
        "openai_span_years": (dates.max() - dates.min()).days / 365.25,
    }
    boiling = t.fit("instant", "openai", "saturation_mae")
    paired = t.paired()
    fall = t.at("instant").loc[paired, "saturation_mae"] - t.at("reasoned").loc[paired, "saturation_mae"]
    out["boiling_gain_in_years_of_openai_trend"] = fall.median() / abs(boiling["per_year"])
    for provider in ("openai", "anthropic"):
        for metric in ("saturation_bias", "melting_bias", "physical_change", "one_region"):
            out[f"{provider}_instant_{metric}"] = t.fit("instant", provider, metric)
            out[f"{provider}_instant_{metric}_shared_span"] = t.fit("instant", provider, metric, span="shared")
    for name, metric in (("as_scored", "balanced_accuracy"), ("hot_gas", "ba_hot_gas"), ("hot_fluid", "ba_hot_fluid"), ("any_fluid", "ba_any_fluid")):
        out[f"openai_instant_naming_{name}"] = t.fit("instant", "openai", metric)
    return out


def clear_of_zero(fit: dict) -> bool:
    return fit["low"] > 0 or fit["high"] < 0


def curves_by_notation(t: Tables) -> dict:
    """Which instant trends are resolved in each notation at 32 × 32."""
    level = t.unit_levels["instant"]
    out = {}
    for provider in ("openai", "anthropic"):
        for metric in NOTATION_TRENDS:
            fits = {f.key: t.fit("instant", provider, metric, framing=f.key, level=level) for f in FRAMINGS}
            shared = {f.key: t.fit("instant", provider, metric, framing=f.key, level=level, span="shared") for f in FRAMINGS}
            out[f"{provider}_{metric}"] = {"resolved": sum(clear_of_zero(fit) for fit in fits.values()), "slopes": {k: fit["per_year"] for k, fit in fits.items()},
                                           "resolved_in": [k for k, fit in fits.items() if clear_of_zero(fit)],
                                           "resolved_in_shared_span": [k for k, fit in shared.items() if clear_of_zero(fit)]}
    fine = {provider: {metric: clear_of_zero(t.fit("instant", provider, metric)) for metric in NOTATION_TRENDS} for provider in ("openai", "anthropic")}
    return out | {"resolved_at_64_in_k_mpa": fine}


def openai_accuracy_by_notation(t: Tables) -> dict:
    """OpenAI's instant accuracy trend in each notation at 32 × 32, and how sturdy each one is."""
    level = t.unit_levels["instant"]
    chosen = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == level) & (t.framings["provider"] == "openai")]
    out = {}
    for framing in FRAMINGS:
        rows = chosen[chosen["framing"] == framing.key].sort_values("released").reset_index(drop=True)
        out[framing.key] = {"all": t.fit("instant", "openai", "balanced_accuracy", framing=framing.key, level=level),
                            "shared": t.fit("instant", "openai", "balanced_accuracy", span="shared", framing=framing.key, level=level),
                            "lost_without_one": lost_without_one(rows, "balanced_accuracy")}
    return out


def provider_contrasts(t: Tables) -> dict:
    """OpenAI's yearly change less Anthropic's: one interval clearing zero is not a difference."""
    instant = t.at("instant").reset_index()
    out = {}
    for metric in ("saturation_mae", "melting_mae", "balanced_accuracy", "saturation_bias"):
        drawn = {}
        rng = np.random.default_rng(0)
        for provider in ("openai", "anthropic"):
            rows = instant[instant["provider"] == provider].sort_values("released")
            years = (pd.to_datetime(rows["released"]) - pd.Timestamp("2024-01-01")).dt.days.to_numpy() / 365.25
            slopes, weights = grade.trend_slopes(years, rows[metric].to_numpy(), 0)
            drawn[provider] = rng.choice(slopes, size=CONTRAST_DRAWS, p=weights / weights.sum())
        low, high = np.quantile(drawn["openai"] - drawn["anthropic"], [0.025, 0.975])
        out[metric] = {"difference": t.fit("instant", "openai", metric)["per_year"] - t.fit("instant", "anthropic", metric)["per_year"], "low": low, "high": high}
    return out


def reasoning_drift(t: Tables) -> dict:
    """How the reasoning models use moves with release date, which a reasoned trend cannot separate."""
    reasoned = t.at("reasoned")
    out = {}
    for provider in ("anthropic", "openai"):
        rows = reasoned[reasoned["provider"] == provider]
        days = (pd.to_datetime(rows["released"]) - pd.Timestamp("2024-01-01")).dt.days
        for metric in ("share_thought", "reasoning_tokens"):
            rho, p = stats.spearmanr(days, rows[metric])
            out[f"{provider}_{metric}"] = {"spearman": rho, "p": p, "n": len(rows)}
    return out


def interval_counts(t: Tables) -> dict:
    """How many intervals were examined, since the text quotes only some of them."""
    keys = ["condition", "provider", "metric", "framing", "level", "per_year", "per_year_low", "per_year_high"]
    distinct = t.trends.drop_duplicates(keys)
    distinct = distinct[distinct["per_year_low"].notna()]
    clear = distinct[(distinct["per_year_low"] > 0) | (distinct["per_year_high"] < 0)]
    openai_instant = clear[(clear["provider"] == "openai") & (clear["condition"] == "instant")]
    contrasts = t.notation[t.notation["low"].notna()]
    clear_contrasts = contrasts[(contrasts["low"] > 0) | (contrasts["high"] < 0)]
    return {
        "trends": len(distinct),
        "trends_clear": len(clear),
        "trends_clear_openai_instant": len(openai_instant),
        "trends_clear_openai_instant_whole_series": int((openai_instant["span"] == "all").sum()),
        "trends_clear_anthropic_instant": int(((clear["provider"] == "anthropic") & (clear["condition"] == "instant")).sum()),
        "trends_anthropic_instant": int(((distinct["provider"] == "anthropic") & (distinct["condition"] == "instant")).sum()),
        "contrasts": len(contrasts),
        "contrasts_clear": len(clear_contrasts),
        "contrasts_clear_instant": int((clear_contrasts["condition"] == "instant").sum()),
    }


def openai_without(t: Tables) -> dict:
    """OpenAI's instant accuracy trend leaving out each model in turn."""
    rows = t.at("instant").reset_index()
    rows = rows[rows["provider"] == "openai"].sort_values("released").reset_index(drop=True)
    out = {}
    for name in rows["label"]:
        keep = rows[rows["label"] != name].reset_index(drop=True)
        fitted = grade.trend(keep["released"], keep["balanced_accuracy"])
        out[name] = {"per_year": fitted["per_year"], "low": fitted["per_year_low"], "high": fitted["per_year_high"]}
    return {
        "models": out,
        "slopes": [min(v["per_year"] for v in out.values()), max(v["per_year"] for v in out.values())],
        "lose_the_interval": [name for name, v in out.items() if v["low"] <= 0],
    }


def small_models(t: Tables) -> dict:
    instant = t.at("instant")
    ranks = instant[["balanced_accuracy", "saturation_mae", "melting_mae"]].copy()
    ranks["balanced_accuracy"] = ranks["balanced_accuracy"].rank(ascending=False)
    ranks["saturation_mae"] = ranks["saturation_mae"].rank()
    ranks["melting_mae"] = ranks["melting_mae"].rank()
    picks = ["GPT-6 Luna", "Haiku 5.5", "Haiku 4.5", "GPT-4o mini"]
    return {
        "scores": instant.loc[picks, ["balanced_accuracy", "saturation_mae", "melting_mae"]].to_dict("index"),
        "ranks": ranks.loc[picks].to_dict("index"),
    }


def liquid_lines(t: Tables, points: pd.DataFrame, label: str) -> dict:
    """How many isotherms crossing the boiling curve hold any liquid answer on one instant map."""
    model = next(m for m in MODELS if m.label == label)
    frame = grade.at_level(grade.attempt(load_answers(model), points, arm=grade.arm_for(model, "instant")), t.shared)
    warm = frame[(frame["T_K"] > truth.T_TP) & (frame["T_K"] < T_C)]
    lines = warm["i"].to_numpy() >> t.shared
    pred = warm["pred"].to_numpy()
    return {"isotherms": len(np.unique(lines)), "with_liquid": len(np.unique(lines[pred == LIQUID]))}


def sketch_margins(t: Tables, points: pd.DataFrame) -> dict:
    """Which reasoned maps are clearly above or below the sketch, resampling grid points."""
    chosen = grade.at_level(points, t.shared).sort_values("pid")
    true = chosen["truth"].to_numpy()
    sketch = grade.sketch_phase(chosen["T_K"], chosen["p_Pa"])
    picks = np.random.default_rng(0).integers(len(true), size=(SKETCH_DRAWS, len(true)))
    out = {}

    def balanced(answers: np.ndarray) -> np.ndarray:
        right = answers[picks] == true[picks]
        kinds = [true[picks] == k for k in range(len(truth.CLASSES))]
        return np.mean([(right & kind).sum(axis=1) / kind.sum(axis=1) for kind in kinds], axis=0)

    baseline = balanced(sketch)
    for model, frame in frames("reasoned", t.shared, points):
        pred = grade.at_level(frame, t.shared).set_index("pid").reindex(chosen["pid"])["pred"].to_numpy()
        low, high = np.quantile(balanced(pred) - baseline, [0.025, 0.975])
        out[model.label] = [low, high]
    return {
        "clearly_above": sorted(k for k, (low, _) in out.items() if low > 0),
        "clearly_below": sorted(k for k, (_, high) in out.items() if high < 0),
        "unresolved": {k: v for k, v in out.items() if v[0] <= 0 <= v[1]},
    }


def sketch_on_finest(points: pd.DataFrame) -> dict:
    """GPT-6 Luna and GPT-5.1 against the sketch on all the points of their 128 × 128 maps."""
    chosen = grade.at_level(points, 0).sort_values("pid")
    true = chosen["truth"].to_numpy()
    sketch = grade.sketch_phase(chosen["T_K"], chosen["p_Pa"])

    def balanced(answers: np.ndarray, picks: np.ndarray) -> np.ndarray:
        right = answers[picks] == true[picks]
        kinds = [true[picks] == k for k in range(len(truth.CLASSES))]
        return np.mean([(right & kind).sum(axis=1) / kind.sum(axis=1) for kind in kinds], axis=0)

    out = {}
    for label in ("GPT-6 Luna", "GPT-5.1"):
        model = BY_LABEL[label]
        frame = grade.attempt(load_answers(model), points, arm=grade.arm_for(model, "reasoned"))
        pred = grade.at_level(frame, 0).set_index("pid").reindex(chosen["pid"])["pred"].to_numpy()
        rng = np.random.default_rng(0)
        lead = []
        for _ in range(FINE_DRAWS // DRAW_CHUNK):
            picks = rng.integers(len(true), size=(DRAW_CHUNK, len(true)))
            lead.append(balanced(pred, picks) - balanced(sketch, picks))
        out[label] = {"score": grade.balanced_accuracy(true, pred), "lead": grade.balanced_accuracy(true, pred) - grade.balanced_accuracy(true, sketch),
                      "interval": [float(q) for q in np.quantile(np.concatenate(lead), [0.025, 0.975])]}
    return out


def ice_wall(points: pd.DataFrame, level: int) -> dict:
    """Where solid gives way to the rest below the triple-point pressure, for the models that wall it."""
    out = {}
    for name in ("claude-sonnet-4-5-20250929", "claude-opus-4-7", "claude-haiku-4-5-20251001", "gpt-4o-mini-2024-07-18", "gpt-4.1-2025-04-14"):
        model = BY_ID[name]
        frame = grade.at_level(grade.attempt(load_answers(model), points, arm="direct"), level)
        low = frame[frame["p_Pa"] < truth.P_TP]
        low = low[low["pred"] >= 0]
        out[model.label] = grade.best_split(low["T_K"].to_numpy(), low["pred"].to_numpy() == SOLID)
    return out


def melting_without(t: Tables) -> dict:
    """Anthropic's reasoned melting trend with whole families left out."""
    reasoned = t.at("reasoned").reset_index()
    rows = reasoned[reasoned["provider"] == "anthropic"].sort_values("released")
    fit = lambda keep: grade.trend(keep["released"].reset_index(drop=True), keep["melting_mae"].reset_index(drop=True))
    out = {}
    for name, keep in (("all", rows), *((f"without_{family}", rows[rows["family"] != family]) for family in ("fable", "haiku", "opus", "sonnet"))):
        fitted = fit(keep)
        out[name] = {"per_year": fitted["per_year"], "low": fitted["per_year_low"], "high": fitted["per_year_high"], "n": fitted["n"]}
    alone = {name: fit(rows[rows["label"] != name]) for name in rows["label"]}
    out["lose_the_interval_without_one"] = [name for name, fitted in alone.items() if fitted["per_year_high"] >= 0]
    out["models"] = len(rows)
    return out


def crossings(line: np.ndarray, T: float) -> np.ndarray:
    """Every pressure in GPa where a melting line, followed up in pressure, crosses T."""
    rows = line[np.argsort(line[:, 1])]
    above = rows[:, 0] >= T
    turns = np.nonzero(above[1:] != above[:-1])[0]
    at = [rows[k, 1] + (T - rows[k, 0]) * (rows[k + 1, 1] - rows[k, 1]) / (rows[k + 1, 0] - rows[k, 0]) for k in turns]
    return 10 ** np.array(([rows[0, 1]] if above[0] else []) + at) / GPA


def line_medians(first: pd.Series) -> float | None:
    """The median first crossing, counting a line that never reaches T as above every line that does."""
    middle = first.fillna(np.inf).median()
    return float(middle) if np.isfinite(middle) else None


def fans(t: Tables, points: pd.DataFrame) -> dict:
    """Where the instant melting lines sit at 400 and 600 K, the temperatures the page quotes."""
    rows = []
    for model, frame in frames("instant", t.shared, points):
        line = grade.boundary_lines(frame, t.shared)["melting"]
        for T in FAN_TEMPERATURES:
            found = crossings(line, T)
            rows.append({"label": model.label, "T": T, "recent": model.released >= "2026-01-01", "first": found[0] if len(found) else np.nan, "last": found[-1] if len(found) else np.nan})
    table = pd.DataFrame(rows)
    out = {"true_gpa_at": {T: float(truth.p_melt_high(np.array([float(T)]))[0]) / GPA for T in FAN_TEMPERATURES},
           "by_model": {f"at_{T}_k": part.set_index("label")[["first", "last"]].to_dict("index") for T, part in table.groupby("T")}}
    for T, part in table.groupby("T"):
        recent, older = part[part["recent"]], part[~part["recent"]]
        out[f"at_{T}_k"] = {
            "lines_2026": len(recent),
            "lines_2026_that_reach": int(recent["first"].notna().sum()),
            "lines_before": len(older),
            "lines_before_that_reach": int(older["first"].notna().sum()),
            "highest_crossing_2026": float(recent["last"].max()),
            "lines_2026_at_or_below_2_gpa_throughout": int((recent["last"] <= 2).sum()),
            "median_2026": line_medians(recent["first"]),
            "median_before": line_medians(older["first"]),
        }
    return out


def regions(t: Tables) -> dict:
    instant, reasoned = t.at("instant"), t.at("reasoned")
    paired = t.paired()
    openai_recent = instant[(instant["provider"] == "openai") & (instant["released"] >= "2025-12-01")]
    columns = [f"wrong_{name}" for name in grade.REGION_NAMES]
    return {
        "means": {"instant": instant[columns].mean().to_dict(), "reasoned": reasoned[columns].mean().to_dict()},
        "means_without_supercritical_answers": {
            "instant": instant[[f"misnamed_{n}" for n in grade.REGION_NAMES]].mean().to_dict(),
            "reasoned": reasoned[[f"misnamed_{n}" for n in grade.REGION_NAMES]].mean().to_dict(),
        },
        "paired_means": {"instant": instant.loc[paired, columns].mean().to_dict(), "reasoned": reasoned.loc[paired, columns].mean().to_dict()},
        "paired_means_without_supercritical_answers": {
            "instant": instant.loc[paired, [f"misnamed_{n}" for n in grade.REGION_NAMES]].mean().to_dict(),
            "reasoned": reasoned.loc[paired, [f"misnamed_{n}" for n in grade.REGION_NAMES]].mean().to_dict(),
        },
        "cold_vapor_worst": instant["wrong_cold_vapor"].sort_values(ascending=False).head(6).to_dict(),
        "ice_lost_from_gpt_5_2": openai_recent[["wrong_ordinary_ice", "wrong_cold_vapor", "sublimation_bias"]].to_dict("index"),
    }


def ice_lost(t: Tables, points: pd.DataFrame) -> dict:
    """What recent OpenAI models call the ordinary ice they miss."""
    out = {}
    for label in ("GPT-5.2", "GPT-5.4", "GPT-5.5", "GPT-5.6 Sol", "GPT-6 Sol", "GPT-6.1 Terra"):
        model = BY_LABEL[label]
        frame = grade.at_level(grade.attempt(load_answers(model), points, arm=grade.arm_for(model, "instant")), t.shared)
        T, p, true, pred = frame["T_K"].to_numpy(), frame["p_Pa"].to_numpy(), frame["truth"].to_numpy(), frame["pred"].to_numpy()
        said = pred[grade.regions(T, p, true)["ordinary_ice"]]
        out[label] = {"points": len(said), "missed": int((said != SOLID).sum()), "vapor": int((said == VAPOR).sum()), "liquid": int((said == LIQUID).sum()),
                      "supercritical": int((said == SUPERCRITICAL).sum()), "liquid_share_of_missed": float((said == LIQUID).sum() / (said != SOLID).sum())}
    return out


def naming_tiers(t: Tables, points: pd.DataFrame) -> dict:
    """Where the word supercritical falls in the two regions it costs most, averaged over maps."""
    chosen = grade.at_level(points, t.shared).sort_values("pid")
    T, p, true = chosen["T_K"].to_numpy(), chosen["p_Pa"].to_numpy(), chosen["truth"].to_numpy()
    parts = grade.regions(T, p, true)
    liquid, hot = parts["compressed_liquid"], parts["hot_vapor"]
    out = {"compressed_liquid_points": int(liquid.sum()), "hot_gas_points": int(hot.sum()), "points_below_named_t": int((liquid & (T < NAMED_FLUID_T)).sum()),
           "named_t": NAMED_FLUID_T}
    for condition in ("instant", "reasoned"):
        rows = []
        for model, frame in frames(condition, t.shared, points):
            pred = grade.at_level(frame, t.shared).set_index("pid").reindex(chosen["pid"])["pred"].to_numpy()
            said = pred == SUPERCRITICAL
            share_of = lambda here, whole: (said & here).sum() / whole.sum()
            rows.append({
                "liquid_wrong": (pred[liquid] != LIQUID).mean(),
                "liquid_said": share_of(liquid, liquid),
                "liquid_said_beside_tc": share_of(liquid & (T >= NAMED_FLUID_T), liquid),
                "liquid_said_further": share_of(liquid & (T < NAMED_FLUID_T), liquid),
                "liquid_said_below_500": share_of(liquid & (T < 500), liquid),
                "liquid_said_below_400": share_of(liquid & (T < 400), liquid),
                "hot_wrong": (pred[hot] != VAPOR).mean(),
                "hot_said": share_of(hot, hot),
                "hot_said_over_a_decade_below_pc": share_of(hot & (p < P_C / 10), hot),
            })
        out[condition] = pd.DataFrame(rows).mean().to_dict()
    return out


def compressed_liquid_words(t: Tables, points: pd.DataFrame) -> dict:
    """What OpenAI's instant maps say on liquid above the critical pressure, oldest first."""
    out = {}
    for model, frame in frames("instant", t.shared, points):
        if model.provider != "openai":
            continue
        at = grade.at_level(frame, t.shared)
        said = at.loc[(at["truth"] == LIQUID) & (at["p_Pa"] >= P_C), "pred"].to_numpy()
        out[model.label] = {"supercritical": float((said == SUPERCRITICAL).mean()), "liquid": float((said == LIQUID).mean()), "solid": float((said == SOLID).mean())}
    return out


def boiling_line_meaning(t: Tables, points: pd.DataFrame) -> dict:
    """What a model says just above its boiling line, and how far off the line is where that is liquid."""
    side = grid.SIDES[t.shared]
    out = {}
    for label in ("GPT-4o mini", "GPT-4o", "GPT-4.1", "Haiku 5.5"):
        model = BY_LABEL[label]
        frame = grade.at_level(grade.attempt(load_answers(model), points, arm=grade.arm_for(model, "instant")), t.shared)
        frame = frame[frame["pred"] >= 0]
        T, p, pred = frame["T_K"].to_numpy(), frame["p_Pa"].to_numpy(), frame["pred"].to_numpy()
        warm = (T > truth.T_TP) & (T < T_C)
        line, offset, said = frame["i"].to_numpy()[warm] >> t.shared, np.log10(p[warm]) - np.log10(truth.p_sat(T[warm])), pred[warm]
        next_words, errors = [], []
        for k in np.unique(line):
            here = line == k
            if here.sum() >= side / 2 and offset[here].min() < 0 < offset[here].max():
                order = np.argsort(offset[here])
                ranked, words = offset[here][order], said[here][order]
                split = grade.best_split(ranked, words == VAPOR)
                after = words[(ranked > split) & (words != VAPOR)]
                next_words.append(truth.CLASSES[after[0]] if len(after) else "none")
                errors.append(abs(grade.relative_split(ranked, words == VAPOR)))
        next_words, errors = np.array(next_words), np.array(errors)
        out[label] = {"lines": len(errors), "error": errors.mean(), **{word: int((next_words == word).sum()) for word in (*truth.CLASSES, "none")},
                      "error_where_liquid_is_next": errors[next_words == "liquid"].mean() if (next_words == "liquid").any() else None}
    return out


def high_pressure_bias(t: Tables, points: pd.DataFrame) -> dict:
    """Signed melting error above the Ih-III pressure by notation: positive is ice too hot."""
    level = t.unit_levels["instant"]
    rows = {}
    for model in sorted(MODELS, key=lambda m: m.released):
        arm = grade.arm_for(model, "instant")
        answers = load_answers(model)
        if arm is None or not len(answers):
            continue
        for framing in FRAMINGS:
            frame = grade.attempt(answers, points, framing=framing.key, arm=arm)
            if len(grade.at_level(frame, level)) == grid.SIDES[level] ** 2:
                split = grade.boundary_splits(frame, level)["melting"]
                logp = grid.LOGP_MIN + (split[:, 0] + 0.5) * (grid.LOGP_MAX - grid.LOGP_MIN) / grid.SIDES[level]
                rows.setdefault(model.label, {})[framing.key] = split[logp >= np.log10(truth.P_IH_III), 1].mean()
    table = pd.DataFrame(rows).T
    released = pd.Series({m.label: m.released for m in MODELS})
    recent = table.index[released[table.index] >= "2026-01-01"]
    return {
        "named": table.loc[["GPT-4o mini", "GPT-4o", "GPT-4.1", "Haiku 4.5", "Haiku 5.5"]].round(0).to_dict("index"),
        "too_cold_in_all_six": table.index[(table < 0).all(axis=1)].tolist(),
        "too_hot_in_all_six_2026": [name for name in recent if (table.loc[name] > 0).all()],
        "models_2026": len(recent),
    }


def pile_ups(t: Tables, points: pd.DataFrame) -> dict:
    """The three places instant errors gather, and how they fare with reasoning allowed."""
    level = t.shared
    paired = set(t.paired())
    out = {}
    chosen = grade.at_level(points, level)
    T, p, true = chosen["T_K"].to_numpy(), chosen["p_Pa"].to_numpy(), chosen["truth"].to_numpy()
    under_pc = (true == VAPOR) & (p < P_C) & (p >= P_C / 10)
    places = {
        "compressed_liquid": (true == LIQUID) & (p >= P_C),
        "cold_vapor": (true == VAPOR) & (T < truth.T_TP),
        "vapor_within_a_decade_below_pc": under_pc,
    }
    anywhere = np.any(list(places.values()), axis=0)
    hot_near_pc = (true == VAPOR) & (T >= T_C) & (p >= 2e6) & (p < P_C)
    words = {}
    for condition in ("instant", "reasoned"):
        wrong = {name: [] for name in places}
        pooled, whole = [], []
        supercritical_on_hot, wrong_on_hot = 0, 0
        said_by_model, total_wrong, paired_wrong = {}, 0, 0
        hot_wrong, region_wrong = {}, {name: 0 for name in grade.REGION_NAMES}
        paired_region_wrong = {name: 0 for name in grade.REGION_NAMES}
        for model, frame in frames(condition, level, points):
            at = grade.at_level(frame, level).set_index("pid").reindex(chosen["pid"])
            pred = at["pred"].to_numpy()
            miss = pred != true
            if model.label in paired:
                for name, here in places.items():
                    wrong[name].append(miss[here].mean())
                pooled.append(miss[anywhere].mean())
                whole.append(miss.mean())
            wrong_on_hot += miss[hot_near_pc].sum()
            supercritical_on_hot += (pred[hot_near_pc] == SUPERCRITICAL).sum()
            total_wrong += miss.sum()
            for name, here in grade.regions(T, p, true).items():
                region_wrong[name] += miss[here].sum()
                if model.label in paired:
                    paired_region_wrong[name] += miss[here].sum()
            paired_wrong += miss.sum() if model.label in paired else 0
            hot_wrong[model.label] = int(miss[(true == VAPOR) & (T >= T_C)].sum())
        words[condition] = {
            "paired_share_wrong": {name: float(np.mean(v)) for name, v in wrong.items()},
            "paired_three_places": float(np.mean(pooled)),
            "paired_whole_map": float(np.mean(whole)),
            "hot_gas_just_below_pc_wrong": wrong_on_hot,
            "hot_gas_just_below_pc_supercritical_share": supercritical_on_hot / wrong_on_hot if wrong_on_hot else None,
            "region_share_of_all_wrong": {k: v / total_wrong for k, v in region_wrong.items()},
            "paired_region_share_of_wrong": {k: v / paired_wrong for k, v in paired_region_wrong.items()},
            "total_wrong": total_wrong,
            "hot_vapor_top_models": dict(sorted(hot_wrong.items(), key=lambda kv: -kv[1])[:5]),
        }
    out["by_condition"] = words
    out["points"] = {name: int(here.sum()) for name, here in places.items()}
    return out


def unit_control(t: Tables, points: pd.DataFrame) -> dict:
    instant_level, reasoned_level = t.unit_levels["instant"], t.unit_levels["reasoned"]
    u = t.units
    instant = u[(u["condition"] == "instant") & (u["level"] == instant_level)].set_index("label")
    reasoned = u[(u["condition"] == "reasoned") & (u["level"] == reasoned_level)].set_index("label")
    n = t.notation
    effect = lambda condition, metric, first, second: n[(n["condition"] == condition) & (n["metric"] == metric) & (n["first"] == first) & (n["second"] == second)].iloc[0][["n", "mean", "low", "high", "first_higher"]].to_dict()
    level_1 = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == instant_level)]
    cold = level_1.pivot(index="label", columns="framing", values="wrong_cold_vapor")
    finer = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == t.shared)]
    cold_finer = finer.pivot(index="label", columns="framing", values="wrong_cold_vapor")
    return {
        "levels": t.unit_levels,
        "instant_gap": extent(instant["gap"]),
        "instant_gap_all_positive": bool((instant["gap_low"] > 0).all()),
        "instant_cross": extent(instant["cross"]),
        "instant_retest": instant.loc[["GPT-4o", "Sonnet 4.6"], ["cross", "retest", "gap"]].to_dict("index"),
        "instant_gaps": instant["gap"].sort_values(ascending=False).round(3).to_dict(),
        "reasoned_gap": extent(reasoned["gap"]),
        "reasoned_gap_ci_includes_zero": int(((reasoned["gap_low"] <= 0) & (reasoned["gap_high"] >= 0)).sum()),
        "reasoned_models": len(reasoned),
        "effects": {
            "ba_c_minus_k": effect("instant", "balanced_accuracy", "C", "K"),
            "familiar_atm_minus_pa": effect("instant", "ba_familiar", "atm", "Pa"),
            "cold_vapor_c_minus_k": effect("instant", "wrong_cold_vapor", "C", "K"),
            "ordinary_ice_c_minus_k": effect("instant", "wrong_ordinary_ice", "C", "K"),
        },
        "cold_vapor_misses": {name: cold.loc[name].round(3).to_dict() for name in ("Sonnet 4.5", "Sonnet 4.6")},
        "cold_vapor_misses_finer": cold_finer.loc["Sonnet 4.6"].round(3).to_dict(),
        "opus_5_hot_vapor": opus_5_words(points, instant_level),
        "opus_5_hot_vapor_finer": opus_5_words(points, t.shared),
        "opus_5_gap_with_words_forgiven": forgiven_gap(points, instant_level),
    }


def opus_5_words(points: pd.DataFrame, level: int) -> dict:
    model = BY_ID["claude-opus-5"]
    answers = load_answers(model)
    out = {}
    for framing in FRAMINGS:
        frame = grade.at_level(grade.attempt(answers, points, framing=framing.key), level)
        hot = (frame["truth"] == VAPOR) & (frame["T_K"] >= T_C)
        out[framing.key] = float((frame.loc[hot, "pred"] == SUPERCRITICAL).mean())
    return out


def forgiven_gap(points: pd.DataFrame, level: int) -> dict:
    """Opus 5's gap once a supercritical answer on fluid is replaced by the true word."""
    model = BY_ID["claude-opus-5"]
    answers = load_answers(model)
    place = points.set_index("pid").loc[answers["pid"]]
    T, p, true = place["T_K"].to_numpy(), place["p_Pa"].to_numpy(), place["truth"].to_numpy()
    fluid = grade.tolerated(T, p, true)["any_fluid"]
    mended = answers.assign(label=np.where(fluid & (answers["label"] == "supercritical"), np.array(truth.CLASSES)[true], answers["label"]))
    return {"strict": grade.score_units(answers, points, level)["gap"], "forgiven": grade.score_units(mended, points, level)["gap"]}


def number_form(t: Tables, points: pd.DataFrame) -> dict:
    """Where a unit writes pressures in scientific notation, and what crossing into it does.

    Above and below 1.000e6 pascals every point is vapor, so the true answer does not
    change, and megapascals and atmospheres say what the same pressures do without the switch.
    """
    chosen = grade.at_level(points, t.shared).sort_values("pid")
    p = chosen["p_Pa"].to_numpy()
    shares = {unit: float(np.mean(["e" in format_number(value / scale) for value in p])) for unit, scale in P_SCALE.items()}
    logp, T = np.log10(p), chosen["T_K"].to_numpy()

    def either_side(cut: float, band: float = STEP_BAND) -> tuple[np.ndarray, np.ndarray]:
        top = 10 ** (cut + band)
        vapor_throughout = (chosen["truth"].to_numpy() == VAPOR) & (T >= float(grade.t_sat(top))) & (np.abs(logp - cut) <= band)
        return np.nonzero(vapor_throughout & (logp < cut))[0], np.nonzero(vapor_throughout & (logp >= cut))[0]

    def interval(rng: np.random.Generator, gap: np.ndarray, low: np.ndarray, high: np.ndarray) -> list[float]:
        """The change in a per-point value across a cut, resampling the points on each side."""
        drawn = gap[high[rng.integers(len(high), size=(STEP_DRAWS, len(high)))]].mean(axis=1) - gap[low[rng.integers(len(low), size=(STEP_DRAWS, len(low)))]].mean(axis=1)
        return [float(q) for q in np.quantile(drawn, [0.025, 0.975])]

    below, above = either_side(6)
    narrow_below, narrow_above = either_side(6, STEP_NARROW_BAND)
    placebos = {cut: either_side(cut) for cut in PLACEBO_CUTS}
    placebo_rows = []
    steps = {}
    for model in sorted(MODELS, key=lambda m: m.released):
        arm = grade.arm_for(model, "instant")
        answers = load_answers(model)
        if arm is None or not len(answers):
            continue
        said = {}
        for unit in P_SCALE:
            per_framing = []
            for framing in FRAMINGS:
                if framing.p_unit == unit:
                    frame = grade.attempt(answers, chosen, framing=framing.key, arm=arm)
                    if len(frame) == len(chosen):
                        per_framing.append((frame.set_index("pid").reindex(chosen["pid"])["pred"].to_numpy() == VAPOR).astype(float))
            if len(per_framing) == 2:
                said[unit] = np.mean(per_framing, axis=0)
        if len(said) < len(P_SCALE):
            continue
        rng = np.random.default_rng(0)
        change = lambda unit, low=below, high=above: said[unit][high].mean() - said[unit][low].mean()
        beside = said["Pa"] - (said["MPa"] + said["atm"]) / 2
        extra_low, extra_high = interval(rng, beside, below, above)
        against_mpa = interval(rng, said["Pa"] - said["MPa"], below, above)
        against_atm = interval(rng, said["Pa"] - said["atm"], below, above)
        steps[model.label] = {"Pa": change("Pa"), "MPa": change("MPa"), "atm": change("atm"), "extra": change("Pa") - (change("MPa") + change("atm")) / 2,
                              "low": extra_low, "high": extra_high, "against_mpa_low": against_mpa[0], "against_mpa_high": against_mpa[1],
                              "against_atm_low": against_atm[0], "against_atm_high": against_atm[1],
                              "mpa_near_the_cut": change("MPa", narrow_below, narrow_above)}
        for cut, (low, high) in placebos.items():
            placebo_low, placebo_high = interval(rng, beside, low, high)
            past_mpa, past_atm = interval(rng, said["Pa"] - said["MPa"], low, high)[1], interval(rng, said["Pa"] - said["atm"], low, high)[1]
            placebo_rows.append({"model": model.label, "cut": cut, "extra": beside[high].mean() - beside[low].mean(), "clear": placebo_low > 0 or placebo_high < 0,
                                 "falls_past_both": placebo_high < 0 and past_mpa < 0 and past_atm < 0})
    table = pd.DataFrame(steps).T.sort_values("extra")
    placebo = pd.DataFrame(placebo_rows)
    falls_past_both = table[(table["high"] < 0) & (table["against_mpa_high"] < 0) & (table["against_atm_high"] < 0)]
    return {
        "scientific_share": shares,
        "points": [len(below), len(above)],
        "points_within_narrow_band": [len(narrow_below), len(narrow_above)],
        "models": len(table),
        "clear_fall": table.index[table["high"] < 0].tolist(),
        "clear_rise": table.index[table["low"] > 0].tolist(),
        "clear_of_both_units": falls_past_both.index.tolist(),
        "large_at_the_switch": int((table["extra"].abs() >= STEP_LARGE).sum()),
        "placebo": {"pairs": len(placebo), "clear": int(placebo["clear"].sum()), "clear_per_cut": placebo.groupby("cut")["clear"].sum().to_dict(),
                    "mean_clear_per_cut": float(placebo.groupby("cut")["clear"].sum().mean()), "large": int((placebo["extra"].abs() >= STEP_LARGE).sum()),
                    "falls_past_both": int(placebo["falls_past_both"].sum()), "mean_falls_past_both_per_cut": float(placebo.groupby("cut")["falls_past_both"].sum().mean())},
        "steps": table.round(3).to_dict("index"),
    }


def boiling_readings(t: Tables, points: pd.DataFrame) -> dict:
    level = t.unit_levels["instant"]
    chosen = grade.at_level(points, level)
    band = grade.EVERYDAY_ROWS * (grid.LOGP_MAX - grid.LOGP_MIN) / grid.SIDES[level]
    near = np.abs(np.log10(chosen["p_Pa"]) - np.log10(grade.ATMOSPHERE)) <= band
    coarse = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == level)]
    table = coarse.pivot(index="label", columns="framing", values="boils_at")
    fine = t.framings[(t.framings["condition"] == "instant") & (t.framings["level"] == t.shared)]
    finer = fine.pivot(index="label", columns="framing", values="boils_at")
    spread = (table.max(axis=1) - table.min(axis=1)).sort_values(ascending=False)
    moved = (table.reindex_like(finer) - finer).abs().stack().dropna()
    readings = table.notna().sum(axis=1)
    return {
        "points_in_band": int(near.sum()),
        "band_atm": [10**(-band), 10**band],
        "readings_total": int(readings.sum()),
        "readings_per_model": readings.to_dict(),
        "widest": {name: [table.loc[name].min(), table.loc[name].max(), spread[name]] for name in spread.index[:4]},
        "gpt_6_sol": [table.loc["GPT-6 Sol"].min(), table.loc["GPT-6 Sol"].max(), (table.loc["GPT-6 Sol"] - grade.ATMOSPHERE_BOILING).abs().max()],
        "steadiest_at_finer_grid": {
            name: [row.max() - row.min(), (row - grade.ATMOSPHERE_BOILING).abs().max()]
            for name, row in finer[finer.notna().sum(axis=1) == len(FRAMINGS)].loc[lambda f: (f.max(axis=1) - f.min(axis=1)).sort_values().index[:4]].iterrows()
        },
        "finer_moves_median": moved.median(),
        "finer_moves_over_20": [int((moved > 20).sum()), len(moved)],
        "finer_moves_max": moved.max(),
        "sonnet_4_6_finer": finer.loc["Sonnet 4.6"].to_dict(),
    }


def repeat_moves(t: Tables, points: pd.DataFrame) -> dict:
    level = t.unit_levels["instant"]
    moves = []
    for model in MODELS:
        arm = grade.arm_for(model, "instant")
        answers = load_answers(model)
        if arm is None or not len(answers):
            continue
        for framing in grade.RETEST_FRAMINGS:
            first = grade.attempt(answers, points, framing=framing, rep=0, arm=arm)
            second = grade.attempt(answers, points, framing=framing, rep=1, arm=arm)
            if len(grade.at_level(second, level)) == grid.SIDES[level] ** 2:
                a = grade.everyday_points(first, level)[0]
                b = grade.everyday_points(second, level)[0]
                if not np.isnan(a) and not np.isnan(b):
                    moves.append(abs(a - b))
    moves = np.array(moves)
    return {"pairs": len(moves), "median": float(np.median(moves)), "over_20": int((moves > 20).sum()), "max": float(moves.max())}


def thoughts(t: Tables) -> dict:
    reasoned = t.at("reasoned")
    focus = t.thought.set_index("label")
    selective = reasoned[reasoned["share_thought"] <= 0.95].sort_values("share_thought")
    rows = {name: {"share": selective.loc[name, "share_thought"], "near": focus.loc[name, "thought_near"], "far": focus.loc[name, "thought_far"], "n": focus.loc[name, "n"]} for name in selective.index}
    return {
        "models": len(selective),
        "shares": [selective["share_thought"].min(), selective["share_thought"].max()],
        "more_near": int(sum(r["near"] > r["far"] for r in rows.values())),
        "rows": rows,
        "next_share": reasoned.loc[reasoned["share_thought"] > 0.95, "share_thought"].sort_values().head(1).to_dict(),
    }


def thought_detail(t: Tables, points: pd.DataFrame) -> dict:
    """Reasoning by pressure band, and on true curves alone, for each model that often skips it."""
    distance = grade.boundary_distance(points)
    near = distance <= 1 / 16
    reasoned = t.at("reasoned")
    out = {}
    for label in reasoned.index[reasoned["share_thought"] <= 0.95]:
        model = BY_LABEL[label]
        answers = load_answers(model)
        arm = grade.arm_for(model, "reasoned")
        chosen = answers[(answers["arm"] == arm) & (answers["framing"] == CANONICAL) & (answers["rep"] == 0)]
        frame = points.assign(near=near).merge(chosen[["pid", "reasoning_tokens"]], on="pid")
        thought = frame["reasoning_tokens"] > 0
        low = frame["p_Pa"] < 1e8
        high = frame["p_Pa"] >= 1e9
        lower_band = (frame["p_Pa"] >= 2e8) & ~high
        corner = (frame["T_K"] > T_C) & high & ~frame["near"]
        coldest = (frame["T_K"] < 250) & high
        spent_tokens = frame["reasoning_tokens"]
        out[label] = {
            "points": len(frame),
            "tokens_near_and_far": [spent_tokens[frame["near"]].mean(), spent_tokens[~frame["near"]].mean()],
            "below_100_mpa": [thought[low & frame["near"]].mean(), thought[low & ~frame["near"]].mean()],
            "above_1_gpa": [thought[high & frame["near"]].mean(), thought[high & ~frame["near"]].mean()],
            "at_or_above_1_gpa": thought[high].mean(),
            "from_200_mpa_to_1_gpa": thought[lower_band].mean(),
            "coldest_above_1_gpa": thought[coldest].mean(),
            "hot_corner_points": int(corner.sum()),
            "hot_corner_reasoned": float(thought[corner].mean()),
        }
    return out


def reasoning_and_errors(t: Tables, points: pd.DataFrame) -> dict:
    """Whether a model reasons more on the points it gets wrong instantly, and whether that holds locally."""
    reasoned = t.at("reasoned")
    out = {}
    for label in reasoned.index[reasoned["share_thought"] <= 0.95]:
        model = BY_LABEL[label]
        if grade.arm_for(model, "instant") is None:
            out[label] = None
            continue
        answers = load_answers(model)
        instant = grade.at_level(grade.attempt(answers, points, arm=grade.arm_for(model, "instant")), t.shared)
        chosen = answers[(answers["arm"] == grade.arm_for(model, "reasoned")) & (answers["framing"] == CANONICAL) & (answers["rep"] == 0)]
        frame = instant.merge(chosen[["pid", "reasoning_tokens"]], on="pid").assign(wrong=lambda f: f["pred"] != f["truth"], thought=lambda f: f["reasoning_tokens"] > 0)
        cells = frame.assign(i16=frame["i"].to_numpy() >> grid.MAX_LEVEL, j16=frame["j"].to_numpy() >> grid.MAX_LEVEL).groupby(["i16", "j16"])
        wrong, right = cells["wrong"].sum(), cells["wrong"].size() - cells["wrong"].sum()
        reasoned_wrong = cells.apply(lambda f: f.loc[f["wrong"], "thought"].mean() if f["wrong"].any() else 0.0, include_groups=False)
        reasoned_right = cells.apply(lambda f: f.loc[~f["wrong"], "thought"].mean() if (~f["wrong"]).any() else 0.0, include_groups=False)
        weight = wrong * right / (wrong + right)
        near = grade.boundary_distance(points).reindex(frame["pid"].map(points.reset_index().set_index("pid")["index"])).to_numpy() <= 1 / 16
        hot_gas = frame["wrong"].to_numpy() & (frame["T_K"].to_numpy() > truth.T_C) & (frame["p_Pa"].to_numpy() < truth.P_C)
        out[label] = {
            "points": len(frame),
            "instant_error_near_and_far": [frame.loc[near, "wrong"].mean(), frame.loc[~near, "wrong"].mean()],
            "errors": int(frame["wrong"].sum()),
            "hot_gas_errors": int(hot_gas.sum()),
            "hot_gas_errors_far_from_a_line": int((hot_gas & ~near).sum()),
            "reasons_on_wrong": frame.loc[frame["wrong"], "thought"].mean(),
            "reasons_on_right": frame.loc[~frame["wrong"], "thought"].mean(),
            "difference_within_16_cells": float((weight * (reasoned_wrong - reasoned_right)).sum() / weight.sum()),
        }
    return out


def tokens(t: Tables) -> dict:
    reasoned = t.at("reasoned")
    return {
        "astra_reasoning_tokens": reasoned.loc["GPT-6 Astra", "reasoning_tokens"],
        "astra_ba": reasoned.loc["GPT-6 Astra", "balanced_accuracy"],
        "fable_5_1_share": reasoned.loc["Fable 5.1", "share_thought"],
    }


def curve_margins(points: pd.DataFrame) -> dict:
    """How near any grid point comes to the ice VII melting curve, the least certain of the curves."""
    T, p = points["T_K"].to_numpy(), points["p_Pa"].to_numpy()
    on_curve = (T >= truth.T_VI_VII) & (T <= truth.T_VII_MAX)
    ratio = p[on_curve] / truth.p_melt_high(T[on_curve])
    return {"nearest_below": ratio[ratio < 1].max(), "nearest_above": ratio[ratio > 1].min()}


def cost() -> dict:
    """What the run cost, from the tokens each API reported, at the prices paid."""
    paid = pd.DataFrame([{"label": m.label, "provider": m.provider, "usd": spent(m, load_answers(m))} for m in MODELS])
    by_provider = paid.groupby("provider")["usd"].sum()
    return {
        "total": paid["usd"].sum(),
        "anthropic": by_provider["anthropic"],
        "openai": by_provider["openai"],
        "dearest": paid.sort_values("usd", ascending=False).head(5).set_index("label")["usd"].to_dict(),
    }


def escapes() -> dict:
    """Replies outside the schema. A bare phase word counts as that word; the rest count as wrong."""
    counts = {"total": 0, "outside_schema": 0, "read_as_word": 0, "cut_off": 0, "refused": 0}
    rows = {}
    for model in MODELS:
        answers = load_answers(model)
        if not len(answers):
            continue
        counts["total"] += len(answers)
        outside = answers[answers["status"] != "schema"]
        unread = outside[outside["label"].isna()]
        cut_off = unread["stop"].isin(["max_tokens", "length"])
        counts["outside_schema"] += len(outside)
        counts["read_as_word"] += int(outside["label"].notna().sum())
        counts["cut_off"] += int(cut_off.sum())
        counts["refused"] += int((~cut_off).sum())
        if len(outside):
            rows[model.label] = {"outside": len(outside), "cut_off": int(cut_off.sum()), "refused": int((~cut_off).sum())}
    return counts | {"counted_wrong": counts["cut_off"] + counts["refused"], "by_model": rows}


def main() -> None:
    t = Tables()
    points = load_grid()
    claims = {
        "headline": headline(t),
        "tidiness": tidiness(t),
        "stray_order": stray_order(t, points),
        "notations": notations(t),
        "naming": naming(t),
        "boundaries": boundaries(t),
        "melting_parts": melting_parts(t, points),
        "melting_tilt": melting_tilt(t),
        "melting_matches": melting_matches(t, points),
        "boiling_reach": boiling_reach(t),
        "trends": trends(t),
        "curves_by_notation": curves_by_notation(t),
        "openai_accuracy_by_notation": openai_accuracy_by_notation(t),
        "provider_contrasts": provider_contrasts(t),
        "reasoning_drift": reasoning_drift(t),
        "interval_counts": interval_counts(t),
        "openai_without": openai_without(t),
        "melting_without": melting_without(t),
        "small_models": small_models(t),
        "haiku_5_5_liquid_lines": liquid_lines(t, points, "Haiku 5.5"),
        "sketch_margins": sketch_margins(t, points),
        "sketch_on_finest": sketch_on_finest(points),
        "ice_wall": ice_wall(points, t.shared),
        "grid": grid_shares(t, points),
        "curve_margins": curve_margins(points),
        "fans": fans(t, points),
        "regions": regions(t),
        "ice_lost": ice_lost(t, points),
        "naming_tiers": naming_tiers(t, points),
        "compressed_liquid_words": compressed_liquid_words(t, points),
        "boiling_line_meaning": boiling_line_meaning(t, points),
        "high_pressure_bias": high_pressure_bias(t, points),
        "pile_ups": pile_ups(t, points),
        "units": unit_control(t, points),
        "number_form": number_form(t, points),
        "boiling_readings": boiling_readings(t, points),
        "repeat_moves": repeat_moves(t, points),
        "thoughts": thoughts(t),
        "thought_detail": thought_detail(t, points),
        "reasoning_and_errors": reasoning_and_errors(t, points),
        "tokens": tokens(t),
        "cost": cost(),
        "escapes": escapes(),
    }
    (RESULTS / "claims.json").write_text(json.dumps(rounded(claims), indent=1, default=str) + "\n")
    print(f"results/claims.json: {sum(len(v) for v in claims.values())} entries")


if __name__ == "__main__":
    main()
