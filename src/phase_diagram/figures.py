"""Draw the results: the atlas, the units poster and the charts behind the findings."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from . import grade, grid, truth
from .framings import BY_KEY, CANONICAL, FRAMINGS
from .pipeline import ROOT, load_answers, load_grid
from .registry import BY_ID, MODELS, Model

FIGURES = ROOT / "results" / "figures"

# Blue, green, yellow and magenta keep every pair of phases apart for colorblind readers,
# which a phase diagram needs because every phase touches every other.
SOLID, LIQUID, VAPOR, SUPERCRITICAL = "#2a78d6", "#008300", "#eda100", "#e87ba4"
NO_ANSWER = "#c3c2b7"
PHASE_MAP = ListedColormap(["#ffffff", NO_ANSWER, SOLID, LIQUID, VAPOR, SUPERCRITICAL])
PHASE_LEGEND = (("Solid", SOLID), ("Liquid", LIQUID), ("Vapor", VAPOR), ("Supercritical", SUPERCRITICAL))

SURFACE = "#fcfcfb"
INK = "#0b0b0b"
SECONDARY = "#52514e"
MUTED = "#898781"
HAIRLINE = "#e1e0d9"
ANTHROPIC, OPENAI = "#2a78d6", "#eb6834"
AGE_OLD, AGE_NEW = "#b7d3f5", "#0d366b"
PROVIDER_COLOR = {"anthropic": ANTHROPIC, "openai": OPENAI}
PROVIDER_NAME = {"anthropic": "Anthropic", "openai": "OpenAI"}
CONDITION_TITLE = {"instant": "Answering instantly", "reasoned": "With reasoning allowed"}

PRESSURE_TICKS = {0: "1 Pa", 3: "1 kPa", 6: "1 MPa", 9: "1 GPa"}
HEADER_HEIGHT = 0.12
NO_OFF_SWITCH = "* Cannot turn reasoning off."
UNITS_LIMIT = 0.12  # both conditions share this scale, so a small deviation looks small

ATLAS_NOTES = {
    "instant": "Fable 5, Fable 5.1, Opus 5.5, o3, GPT-6 Astra and GPT-6.1 Sol cannot turn reasoning off, so they appear only in the other atlas.",
    "reasoned": f"{NO_OFF_SWITCH} GPT-4o, GPT-4o mini, GPT-4.1 and Opus 4.7 did not reason, so they appear only in the instant atlas.",
}


def how_to_read(side: int) -> str:
    return (
        "Each square is one model's phase diagram, built only from its one-word answers. Black lines are the IAPWS phase boundaries\n"
        "and the critical temperature and pressure. Temperature runs left to right, pressure bottom to top. Scores are balanced accuracy\n"
        f"in kelvin and megapascals, on the same {side} × {side} points for every model, even where a map is drawn finer: 0.25 is guessing, 1 is perfect.\n"
    )

plt.rcParams.update(
    {
        "font.family": ["Helvetica Neue", "Arial", "DejaVu Sans"],
        "font.size": 10,
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "axes.edgecolor": "#c3c2b7",
        "axes.labelcolor": SECONDARY,
        "text.color": INK,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "xtick.labelcolor": SECONDARY,
        "ytick.labelcolor": SECONDARY,
    }
)


def true_curves() -> list[tuple[np.ndarray, np.ndarray]]:
    """The IAPWS boundaries in map coordinates, with the critical lines."""
    curves = [(t, np.log10(p)) for t, p in truth.boundary_curves().values()]
    log_pc = np.log10(truth.P_C)
    t_top = truth.p_melt_high(np.array([truth.T_C]))[0]
    curves.append((np.array([truth.T_C, truth.T_C]), np.array([log_pc, np.log10(t_top)])))
    curves.append((np.array([truth.T_C, grid.T_MAX]), np.array([log_pc, log_pc])))
    return curves


CURVES = true_curves()


def draw_map(ax, image: np.ndarray, line_width: float = 0.55) -> None:
    ax.imshow(
        image.T,
        origin="lower",
        cmap=PHASE_MAP,
        vmin=-2,
        vmax=3,
        aspect="auto",
        interpolation="nearest",
        extent=[grid.T_MIN, grid.T_MAX, grid.LOGP_MIN, grid.LOGP_MAX],
    )
    for t, logp in CURVES:
        ax.plot(t, logp, color=INK, lw=line_width, solid_capstyle="round")
    ax.set_xlim(grid.T_MIN, grid.T_MAX)
    ax.set_ylim(grid.LOGP_MIN, grid.LOGP_MAX)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def named(model: Model) -> str:
    return f"{model.label} *" if model.thinks else model.label


def label_axes(ax) -> None:
    ax.set_xticks([200, 400, 600, 800], ["200", "400", "600", "800 K"], fontsize=8)
    ax.set_yticks(list(PRESSURE_TICKS), list(PRESSURE_TICKS.values()), fontsize=8)
    ax.tick_params(length=2, width=0.5, pad=2)


def model_image(model: Model, points: pd.DataFrame, answers: pd.DataFrame, condition: str, framing: str = CANONICAL):
    """The finest complete map a model has, or None."""
    arm = grade.arm_for(model, condition)
    if arm is None or not len(answers):
        return None
    frame = grade.attempt(answers, points, framing=framing, arm=arm)
    level = grade.complete_level(frame)
    return None if level is None else grade.to_image(frame, level)


def atlas(condition: str, scores: pd.DataFrame, references: pd.DataFrame) -> None:
    """Every model's diagram, IAPWS first, in release order within each provider."""
    points = load_grid()
    level = grade.shared_level(scores)
    chosen = scores[(scores["condition"] == condition) & (scores["level"] == level)]
    ba = chosen.set_index("model")["balanced_accuracy"].to_dict()
    sketch_ba = sketch_score(references, level)
    answers = {m.id: load_answers(m) for m in MODELS}
    groups = {}
    for provider in ("anthropic", "openai"):
        chosen = sorted((m for m in MODELS if m.provider == provider), key=lambda m: m.released)
        pairs = [(m, model_image(m, points, answers[m.id], condition)) for m in chosen]
        groups[provider] = [(m, image) for m, image in pairs if image is not None]

    columns = 7
    tile = 1.72
    rows_for = {p: int(np.ceil(len(g) / columns)) for p, g in groups.items()}
    heights = [1.0]
    for provider in ("anthropic", "openai"):
        heights += [HEADER_HEIGHT] + [1.0] * rows_for[provider]
    head, foot = 2.2, 0.3
    height = sum(heights) * (tile + 0.34) + head + foot
    fig = plt.figure(figsize=(columns * tile + 1.15, height))
    outer = fig.add_gridspec(
        len(heights), columns, height_ratios=heights, left=0.062, right=0.985,
        top=1 - head / height, bottom=foot / height, hspace=0.3, wspace=0.08,
    )

    fig.text(0.062, 1 - 0.42 / height, f"How language models picture water: {CONDITION_TITLE[condition].lower()}",
             fontsize=21, fontweight="bold", va="top")
    note = how_to_read(grid.SIDES[level]) + ATLAS_NOTES[condition]
    fig.text(0.062, 1 - 0.86 / height, note, fontsize=11, color=SECONDARY, va="top", linespacing=1.45)

    ax = fig.add_subplot(outer[0, 0])
    draw_map(ax, grade.to_image(points.assign(pred=points["truth"].astype(np.int8)), 0))
    label_axes(ax)
    ax.set_title("IAPWS truth", loc="left", fontsize=11, fontweight="bold", pad=4)

    ax = fig.add_subplot(outer[0, 1])
    sketch = points.assign(pred=grade.sketch_phase(points["T_K"], points["p_Pa"]).astype(np.int8))
    draw_map(ax, grade.to_image(sketch, 0))
    ax.set_title("Four-fact sketch", loc="left", fontsize=11, fontweight="bold", pad=4)
    ax.text(1.0, 1.02, f"{sketch_ba:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)

    legend_ax = fig.add_subplot(outer[0, 2:])
    legend_ax.axis("off")
    handles = [Patch(facecolor=color, label=name) for name, color in PHASE_LEGEND]
    handles.append(Patch(facecolor=NO_ANSWER, label="No answer"))
    handles.append(Line2D([0], [0], color=INK, lw=1, label="IAPWS boundary or critical line"))
    legend_ax.legend(handles=handles, loc="upper left", frameon=False, fontsize=11, handlelength=1.1, handleheight=1.1,
                     borderaxespad=0, labelspacing=0.55, bbox_to_anchor=(0.02, 1.0))
    note = (
        "The sketch uses four remembered facts: the triple\n"
        "point, the critical point, ice melting at 0 °C and\n"
        "one latent heat, joined by the Clausius-Clapeyron\n"
        "equation, plus a built-in layout and the textbook\n"
        "definition of supercritical. A model below this\n"
        "score gets less of the map right than that."
    )
    legend_ax.text(0.5, 1.0, note, transform=legend_ax.transAxes, va="top", fontsize=11, color=SECONDARY, linespacing=1.5)

    row = 1
    for provider in ("anthropic", "openai"):
        header = fig.add_subplot(outer[row, :])
        header.axis("off")
        header.text(0, 0, PROVIDER_NAME[provider], transform=header.transAxes, fontsize=13, fontweight="bold",
                    color=SECONDARY, va="bottom")
        row += 1
        for k, (model, image) in enumerate(groups[provider]):
            ax = fig.add_subplot(outer[row + k // columns, k % columns])
            draw_map(ax, image)
            ax.set_title(named(model), loc="left", fontsize=11, fontweight="bold", pad=4)
            value = ba.get(model.id)
            if value is not None:
                ax.text(1.0, 1.02, f"{value:.2f}", transform=ax.transAxes, ha="right", va="bottom", fontsize=11)
        row += rows_for[provider]
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"atlas_{condition}.png", dpi=170)
    plt.close(fig)


def preview_card() -> None:
    """The image a shared link to the results page unfurls into: the instant atlas, unlabeled."""
    points = load_grid()
    images = [
        grade.to_image(points.assign(pred=points["truth"].astype(np.int8)), 0),
        grade.to_image(points.assign(pred=grade.sketch_phase(points["T_K"], points["p_Pa"]).astype(np.int8)), 0),
    ]
    for model in sorted(MODELS, key=lambda m: m.released):
        image = model_image(model, points, load_answers(model), "instant")
        if image is not None:
            images.append(image)

    columns = 8
    rows = int(np.ceil(len(images) / columns))
    fig = plt.figure(figsize=(12, 6.3))
    tiles = fig.add_gridspec(rows, columns, left=0.042, right=0.958, top=0.685, bottom=0.05, hspace=0.05, wspace=0.05)
    fig.text(0.042, 0.93, "What does a language model think water does?", fontsize=31, fontweight="bold", va="top")
    fig.text(0.042, 0.81, f"The IAPWS reference, a four-fact sketch, then {len(images) - 2} models answering instantly, in release order.",
             fontsize=15, color=SECONDARY, va="top")
    for k, image in enumerate(images):
        draw_map(fig.add_subplot(tiles[k // columns, k % columns]), image, line_width=0.45)
    fig.savefig(ROOT / "results" / "site" / "preview.png", dpi=200)
    plt.close(fig)


def sketch_score(references: pd.DataFrame, level: int) -> float:
    chosen = references[(references["reference"] == "Four-fact sketch") & (references["level"] == level)]
    return float(chosen["balanced_accuracy"].iloc[0])


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES / f"{name}.png", dpi=170, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def style_chart(ax) -> None:
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color("#c3c2b7")
    ax.tick_params(length=0)
    ax.grid(axis="x", color=HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)


def reasoning_effect(scores: pd.DataFrame, references: pd.DataFrame) -> None:
    """Each model with and without reasoning, on the same points."""
    level = grade.shared_level(scores)
    chosen = scores[scores["level"] == level]
    table = chosen.pivot(index="model", columns="condition", values="balanced_accuracy")
    sketch = sketch_score(references, level)
    fig, axes = plt.subplots(1, 2, figsize=(11, 6.2), sharex=True)
    for ax, provider in zip(axes, ("anthropic", "openai")):
        models = sorted((m for m in MODELS if m.provider == provider and m.id in table.index), key=lambda m: m.released)
        for y, model in enumerate(models):
            instant = table.loc[model.id].get("instant", np.nan)
            reasoned = table.loc[model.id].get("reasoned", np.nan)
            ax.plot([instant, reasoned], [y, y], color="#c3c2b7", lw=2, solid_capstyle="round", zorder=1)
            ax.scatter(instant, y, s=62, color=SURFACE, edgecolor=MUTED, linewidth=1.6, zorder=2)
            ax.scatter(reasoned, y, s=62, color=PROVIDER_COLOR[provider], edgecolor=SURFACE, linewidth=1.2, zorder=3)
        ax.set_yticks(range(len(models)), [named(m) for m in models], fontsize=10.5, color=INK)
        ax.set_ylim(len(models) - 0.4, -0.9)
        ax.axvline(sketch, color=SECONDARY, lw=1)
        ax.text(sketch - 0.006, -0.72, f"Four-fact sketch {sketch:.2f}", ha="right", va="center", fontsize=9, color=SECONDARY)
        ax.set_title(PROVIDER_NAME[provider], loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlim(0.5, 1.01)
        ax.set_xlabel("Balanced accuracy")
        style_chart(ax)
    handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.6, markersize=8, label="Answering instantly"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=ANTHROPIC, markeredgecolor=SURFACE, markersize=9, label="Reasoning allowed, Anthropic"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=OPENAI, markeredgecolor=SURFACE, markersize=9, label="Reasoning allowed, OpenAI"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.05))
    fig.text(0.5, -0.075, NO_OFF_SWITCH, ha="center", fontsize=9.5, color=SECONDARY)
    fig.tight_layout()
    save(fig, "reasoning_effect")


GENERATION_PANELS = (
    ("balanced_accuracy", "Balanced accuracy", (0.5, 1.02)),
    ("saturation_mae", "Boiling curve error, decades of pressure", None),
    ("melting_mae", "Melting curve error, kelvin", None),
)
FAMILY_NAME = {"opus": "Opus", "sonnet": "Sonnet", "haiku": "Haiku", "fable": "Fable", "gpt": "GPT", "small": "Small GPT", "o-series": "o3"}


def spread(values: list[float], gap: float) -> np.ndarray:
    """Move label positions apart until neighbors are at least gap away."""
    order = np.argsort(values)
    placed = np.asarray(values, dtype=float)[order]
    for _ in range(100):
        overlaps = gap - np.diff(placed)
        if (overlaps <= 1e-9).all():
            break
        for k, overlap in enumerate(overlaps):
            if overlap > 0:
                placed[k] -= overlap / 2
                placed[k + 1] += overlap / 2
    result = np.empty(len(placed))
    result[order] = placed
    return result


def generations(scores: pd.DataFrame, references: pd.DataFrame) -> None:
    """Accuracy and boundary error by release date. Lines only join a model to its own family."""
    fig, axes = plt.subplots(len(GENERATION_PANELS), 2, figsize=(11.5, 9.6), sharex=True, sharey="row")
    level = grade.shared_level(scores)
    for c, condition in enumerate(("instant", "reasoned")):
        chosen = scores[(scores["condition"] == condition) & (scores["level"] == level)]
        for r, (metric, name, limits) in enumerate(GENERATION_PANELS):
            ax = axes[r, c]
            ends, alone = [], []
            for (provider, family), group in chosen.groupby(["provider", "family"]):
                group = group.sort_values("released")
                dates = pd.to_datetime(group["released"])
                color = PROVIDER_COLOR[provider]
                ax.plot(dates, group[metric], color=color, lw=1.6, alpha=0.85, zorder=2)
                ax.scatter(dates, group[metric], s=40, color=color, edgecolor=SURFACE, linewidth=1, zorder=3)
                if len(group) > 1 or dates.iloc[-1] > pd.Timestamp("2026-01-01"):
                    ends.append((FAMILY_NAME[family], dates.iloc[-1], group[metric].iloc[-1]))
                else:
                    alone.append((FAMILY_NAME[family], dates.iloc[-1], group[metric].iloc[-1]))
            if metric == "balanced_accuracy":
                sketch = sketch_score(references, level)
                ax.axhline(sketch, color=SECONDARY, lw=1)
                ax.text(pd.Timestamp("2024-07-01"), sketch + 0.008, f"Four-fact sketch {sketch:.2f}", fontsize=9, color=SECONDARY)
                ax.set_title(CONDITION_TITLE[condition], loc="left", fontsize=12, fontweight="bold", pad=8)
            if limits:
                ax.set_ylim(*limits)
            else:
                ax.set_ylim(bottom=0)
            low, high = ax.get_ylim()
            if condition == "instant" or metric != "saturation_mae":
                heights = spread([(value - low) / (high - low) for _, _, value in ends], 0.052)
                heights -= max(heights.max() - 0.97, 0) + min(heights.min() - 0.04, 0)
                for (family, date, value), height in zip(ends, heights):
                    ax.annotate(family, (date, value), xytext=(pd.Timestamp("2026-11-15"), height), textcoords=("data", "axes fraction"),
                                fontsize=8.5, color=SECONDARY, va="center", bbox={"facecolor": SURFACE, "edgecolor": "none", "pad": 0.8},
                                arrowprops={"arrowstyle": "-", "color": MUTED, "lw": 0.5, "shrinkA": 1, "shrinkB": 3})
                for family, date, value in alone:
                    ax.annotate(family, (date, value), xytext=(0, 7), textcoords="offset points", fontsize=8.5, color=SECONDARY, ha="center")
            ax.set_xlim(pd.Timestamp("2024-06-15"), pd.Timestamp("2027-02-15"))
            ax.set_xticks([pd.Timestamp("2025-01-01"), pd.Timestamp("2026-01-01")], ["2025", "2026"])
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.tick_params(length=0)
            ax.grid(axis="y", color=HAIRLINE, lw=0.6)
            ax.set_axisbelow(True)
            if c == 0:
                ax.set_ylabel(name, fontsize=9.5)
    handles = [Line2D([0], [0], color=color, lw=2, marker="o", markersize=6, label=PROVIDER_NAME[p]) for p, color in PROVIDER_COLOR.items()]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=10, bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout()
    save(fig, "generations")


def graded_frames(condition: str, level: int):
    """Each model's pass at this level under this condition, oldest model first."""
    points = load_grid()
    for model in sorted(MODELS, key=lambda m: m.released):
        arm = grade.arm_for(model, condition)
        answers = load_answers(model)
        if arm is None or not len(answers):
            continue
        frame = grade.attempt(answers, points, arm=arm)
        finest = grade.complete_level(frame)
        if finest is not None and finest <= level:
            yield model, grade.at_level(frame, level)


def boundary_fan(scores: pd.DataFrame) -> None:
    """Every model's boiling and melting curves on one diagram, the newest drawn last."""
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.4), sharex=True, sharey=True)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("age", [AGE_OLD, ANTHROPIC, AGE_NEW])
    dates = pd.to_datetime([m.released for m in MODELS])
    span = (dates.max() - dates.min()).days
    level = grade.shared_level(scores)
    for ax, condition in zip(axes, ("instant", "reasoned")):
        for model, frame in graded_frames(condition, level):
            color = ramp((pd.Timestamp(model.released) - dates.min()).days / span)
            for curve in grade.boundary_lines(frame, level).values():
                ax.plot(curve[:, 0], curve[:, 1], color=color, lw=1.1, alpha=0.85, solid_joinstyle="round")
        for t, logp in CURVES:
            ax.plot(t, logp, color=INK, lw=1.5, zorder=5)
        ax.set_xlim(grid.T_MIN, grid.T_MAX)
        ax.set_ylim(grid.LOGP_MIN, grid.LOGP_MAX)
        ax.set_title(f"{CONDITION_TITLE[condition]} ({grid.SIDES[level]} × {grid.SIDES[level]})", loc="left", fontsize=12, fontweight="bold", pad=8)
        label_axes(ax)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    bar = fig.colorbar(plt.cm.ScalarMappable(cmap=ramp), ax=list(axes), fraction=0.02, pad=0.015, ticks=[0, 1])
    bar.ax.set_yticklabels([dates.min().strftime("%b %Y"), dates.max().strftime("%b %Y")], fontsize=9)
    bar.set_label("Release date", fontsize=9)
    bar.outline.set_visible(False)
    fig.text(0.05, -0.03, "Black lines: IAPWS. Each model's lines mark where it stops saying vapor or solid, smoothed over five grid lines.", fontsize=9.5, color=SECONDARY)
    save(fig, "boundary_fan")


def error_share(scores: pd.DataFrame) -> None:
    """How many models get each point wrong."""
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.2), sharex=True, sharey=True)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", ["#f4f3ee", "#86b6ef", "#2a78d6", "#0d366b"])
    level = grade.shared_level(scores)
    for ax, condition in zip(axes, ("instant", "reasoned")):
        wrong = [grade.to_image(frame.assign(wrong=(frame["pred"] != frame["truth"]).astype(np.int8)), level, "wrong") for _, frame in graded_frames(condition, level)]
        shown = ax.imshow(np.mean(wrong, axis=0).T, origin="lower", cmap=ramp, vmin=0, vmax=1, aspect="auto", interpolation="nearest",
                          extent=[grid.T_MIN, grid.T_MAX, grid.LOGP_MIN, grid.LOGP_MAX])
        for t, logp in CURVES:
            ax.plot(t, logp, color=INK, lw=0.8)
        ax.set_xlim(grid.T_MIN, grid.T_MAX)
        ax.set_ylim(grid.LOGP_MIN, grid.LOGP_MAX)
        ax.set_title(f"{CONDITION_TITLE[condition]}, {len(wrong)} models, {grid.SIDES[level]} × {grid.SIDES[level]}", loc="left", fontsize=12, fontweight="bold", pad=8)
        label_axes(ax)
        for spine in ax.spines.values():
            spine.set_visible(False)
    bar = fig.colorbar(shown, ax=list(axes), fraction=0.022, pad=0.015, ticks=[0, 0.5, 1])
    bar.ax.set_yticklabels(["0", "50%", "All"], fontsize=9)
    bar.set_label("Share of models wrong", fontsize=9)
    bar.outline.set_visible(False)
    save(fig, "error_share")


def familiar_vs_unfamiliar(scores: pd.DataFrame) -> None:
    """Accuracy near everyday conditions against accuracy everywhere else."""
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 5), sharex=True, sharey=True)
    level = grade.shared_level(scores)
    for ax, condition in zip(axes, ("instant", "reasoned")):
        chosen = scores[(scores["condition"] == condition) & (scores["level"] == level)]
        ax.plot([0.4, 1.02], [0.4, 1.02], color="#c3c2b7", lw=1)
        for provider, group in chosen.groupby("provider"):
            ax.scatter(group["ba_familiar"], group["ba_unfamiliar"], s=58, color=PROVIDER_COLOR[provider], edgecolor=SURFACE,
                       linewidth=1.2, label=PROVIDER_NAME[provider], zorder=3)
        ax.set_xlim(0.4, 1.02)
        ax.set_ylim(0.4, 1.02)
        ax.set_aspect("equal")
        ax.set_title(CONDITION_TITLE[condition], loc="left", fontsize=12, fontweight="bold", pad=8)
        ax.set_xlabel("Everyday water (250–400 K, 10 kPa – 10 MPa)")
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(color=HAIRLINE, lw=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Everywhere else")
    axes[0].legend(frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    fig.text(0.07, -0.02, "Everyday water has no supercritical points, so chance is 0.33 on the x-axis and 0.25 on the y-axis.", fontsize=9.5, color=SECONDARY)
    save(fig, "familiar_vs_unfamiliar")


def region_matrix(scores: pd.DataFrame) -> None:
    """How often each model is wrong in each part of the diagram."""
    level = grade.shared_level(scores)
    models = sorted(MODELS, key=lambda m: (m.provider, m.released))
    columns = [f"wrong_{name}" for name in grade.REGION_NAMES]
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 0.33 * len(models) + 2.3), sharey=True)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", ["#f4f3ee", "#86b6ef", "#2a78d6", "#0d366b"])
    for ax, condition in zip(axes, ("instant", "reasoned")):
        chosen = scores[(scores["condition"] == condition) & (scores["level"] == level)].set_index("model")
        table = chosen.reindex([m.id for m in models])[columns].to_numpy(dtype=float)
        ax.imshow(table, cmap=ramp, vmin=0, vmax=1, aspect="auto")
        for (r, c), value in np.ndenumerate(table):
            if not np.isnan(value):
                ax.text(c, r, f"{100 * value:.0f}", ha="center", va="center", fontsize=8.5, color="#ffffff" if value > 0.45 else INK)
        ax.set_xticks(range(len(columns)), grade.REGION_NAMES.values(), rotation=38, ha="left", fontsize=9, color=INK)
        ax.set_yticks(range(len(models)), [named(m) for m in models], fontsize=10, color=INK)
        ax.xaxis.tick_top()
        ax.tick_params(length=0)
        ax.set_title(CONDITION_TITLE[condition], loc="left", fontsize=12, fontweight="bold", pad=8, y=-0.05)
        ax.set_xticks(np.arange(-0.5, len(columns)), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models)), minor=True)
        ax.grid(which="minor", color=SURFACE, lw=1.5)
        ax.tick_params(which="minor", length=0)
        for spine in ax.spines.values():
            spine.set_visible(False)
    fig.tight_layout()
    fig.text(0.13, -0.03, f"Percent of each region's points answered wrong, {grid.SIDES[level]} × {grid.SIDES[level]}. A blank row means the model has no map in that condition. {NO_OFF_SWITCH}",
             fontsize=9.5, color=SECONDARY)
    save(fig, "region_matrix")


def plausible_vs_accurate(scores: pd.DataFrame, references: pd.DataFrame) -> None:
    """How tidy a map looks, never using the truth, against how right it is."""
    level = grade.shared_level(scores)
    chosen = scores[scores["level"] == level]
    fig, ax = plt.subplots(figsize=(7.2, 5.4))
    for provider in ("anthropic", "openai"):
        for condition, face in (("instant", SURFACE), ("reasoned", PROVIDER_COLOR[provider])):
            group = chosen[(chosen["provider"] == provider) & (chosen["condition"] == condition)]
            ax.scatter(group["one_region"], group["balanced_accuracy"], s=56, facecolor=face, edgecolor=PROVIDER_COLOR[provider],
                       linewidth=1.5, zorder=3)
    truth_row = references[(references["reference"] == "IAPWS") & (references["level"] == level)].iloc[0]
    ax.scatter(truth_row["one_region"], truth_row["balanced_accuracy"], marker="*", s=170, color=INK, zorder=4)
    ax.annotate("IAPWS", (truth_row["one_region"], truth_row["balanced_accuracy"]), xytext=(-8, -2), textcoords="offset points", ha="right", fontsize=10)
    ax.set_xlabel("Share of points in the largest connected region of their phase (never sees the truth)")
    ax.set_ylabel("Balanced accuracy")
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(color=HAIRLINE, lw=0.6)
    ax.set_axisbelow(True)
    handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.5, markersize=8, label="Instant (hollow)"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=MUTED, markersize=8, label="Reasoning allowed (filled)"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=ANTHROPIC, markeredgecolor=ANTHROPIC, markersize=8, label="Anthropic"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=OPENAI, markeredgecolor=OPENAI, markersize=8, label="OpenAI"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="upper left")
    fig.tight_layout()
    save(fig, "plausible_vs_accurate")


def provider_handles() -> list[Line2D]:
    return [Line2D([0], [0], marker="o", ls="", markerfacecolor=c, markeredgecolor=SURFACE, markersize=9, label=PROVIDER_NAME[p]) for p, c in PROVIDER_COLOR.items()]


def footnote(fig, condition: str) -> None:
    if condition == "reasoned":
        fig.text(0.02, -0.02, NO_OFF_SWITCH, fontsize=9.5, color=SECONDARY)


def units_gap(units: pd.DataFrame, condition: str, level: int) -> None:
    """Agreement across unit systems against agreement of a prompt with itself."""
    chosen = units[(units["condition"] == condition) & (units["level"] == level)]
    if "retest" not in chosen or chosen["retest"].isna().all():
        return
    chosen = chosen.dropna(subset=["retest"])
    chosen = chosen.sort_values("gap")
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(chosen) + 1.5))
    for y, row in enumerate(chosen.itertuples()):
        color = PROVIDER_COLOR[row.provider]
        ax.plot([row.cross, row.retest], [y, y], color="#c3c2b7", lw=2, solid_capstyle="round", zorder=1)
        ax.scatter(row.cross, y, s=58, color=SURFACE, edgecolor=color, linewidth=1.6, zorder=2)
        ax.scatter(row.retest, y, s=58, color=color, edgecolor=SURFACE, linewidth=1.2, zorder=3)
    ax.set_yticks(range(len(chosen)), [named(BY_ID[m]) for m in chosen["model"]], fontsize=10, color=INK)
    ax.set_title(CONDITION_TITLE[condition], loc="left", fontsize=12, fontweight="bold", pad=8)
    ax.set_xlabel(f"Share of points answered the same way, {grid.SIDES[level]} × {grid.SIDES[level]}")
    style_chart(ax)
    handles = [
        Line2D([0], [0], marker="o", ls="", markerfacecolor=SURFACE, markeredgecolor=MUTED, markeredgewidth=1.6, markersize=8, label="Same point, different units"),
        Line2D([0], [0], marker="o", ls="", markerfacecolor=MUTED, markeredgecolor=SURFACE, markersize=9, label="Same prompt, asked twice"),
        *provider_handles(),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=10, loc="lower left")
    fig.tight_layout()
    footnote(fig, condition)
    save(fig, f"units_gap_{condition}")


def units_boiling(framings: pd.DataFrame, condition: str, level: int) -> None:
    """Where each model has water boiling at one atmosphere, in each notation."""
    chosen = framings[(framings["condition"] == condition) & (framings["level"] == level)].dropna(subset=["boils_at"])
    spans = chosen.groupby("model")["boils_at"].agg(["min", "max"])
    order = (spans["max"] - spans["min"]).sort_values().index
    fig, ax = plt.subplots(figsize=(8.2, 0.34 * len(order) + 1.5))
    for y, model in enumerate(order):
        color = PROVIDER_COLOR[BY_ID[model].provider]
        ax.plot(spans.loc[model], [y, y], color="#c3c2b7", lw=2, solid_capstyle="round", zorder=1)
        ax.scatter(chosen[chosen["model"] == model]["boils_at"], np.full((chosen["model"] == model).sum(), y), s=38, color=color, edgecolor=SURFACE, linewidth=0.9, zorder=2)
    ax.axvline(grade.ATMOSPHERE_BOILING, color=SECONDARY, lw=1)
    ax.text(grade.ATMOSPHERE_BOILING + 4, -0.75, "True, 373 K", fontsize=9, color=SECONDARY, va="center")
    ax.set_yticks(range(len(order)), [named(BY_ID[m]) for m in order], fontsize=10, color=INK)
    ax.set_ylim(-1.1, len(order) - 0.5)
    ax.set_xlabel(f"Boiling point at 1 atm implied by each notation with a reading, kelvin, {grid.SIDES[level]} × {grid.SIDES[level]}")
    style_chart(ax)
    ax.legend(handles=provider_handles(), frameon=False, fontsize=10, loc="lower right")
    fig.tight_layout()
    save(fig, f"units_boiling_{condition}")


def units_matrix(framings: pd.DataFrame, condition: str, level: int) -> None:
    """How far each notation sits from a model's own average."""
    chosen = framings[(framings["condition"] == condition) & (framings["level"] == level)]
    if chosen.empty:
        return
    table = chosen.pivot(index="label", columns="framing", values="balanced_accuracy")
    order = chosen.sort_values(["provider", "released"]).drop_duplicates("label")["label"]
    table = table.reindex(order)[[f.key for f in FRAMINGS]]
    deviation = table.sub(table.mean(axis=1), axis=0)
    fig, ax = plt.subplots(figsize=(6.6, 0.3 * len(table) + 1.7))
    diverging = matplotlib.colors.LinearSegmentedColormap.from_list("poles", ["#c8322f", "#f0efec", "#1c5cab"])
    image = ax.imshow(deviation.to_numpy(), cmap=diverging, vmin=-UNITS_LIMIT, vmax=UNITS_LIMIT, aspect="auto")
    ax.set_xticks(range(len(FRAMINGS)), [BY_KEY[f.key].label for f in FRAMINGS], fontsize=9.5)
    by_label = {m.label: m for m in MODELS}
    ax.set_yticks(range(len(table)), [named(by_label[label]) for label in table.index], fontsize=10, color=INK)
    ax.xaxis.tick_top()
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks(np.arange(-0.5, len(FRAMINGS)), minor=True)
    ax.set_yticks(np.arange(-0.5, len(table)), minor=True)
    ax.grid(which="minor", color=SURFACE, lw=1.5)
    ax.tick_params(which="minor", length=0)
    bar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.03)
    bar.set_label("Balanced accuracy, compared with the model's own average", fontsize=9)
    bar.outline.set_visible(False)
    ax.set_title(f"{CONDITION_TITLE[condition]}, {grid.SIDES[level]} × {grid.SIDES[level]}", loc="left", fontsize=12, fontweight="bold", pad=10)
    fig.tight_layout()
    footnote(fig, condition)
    save(fig, f"units_matrix_{condition}")


def units_poster(models: list[Model], condition: str, agreement: pd.DataFrame, agreement_level: int, name: str = "units_poster") -> None:
    """One row per model, one map per notation, every map at the same resolution."""
    points = load_grid()
    passes = []
    for model in models:
        answers = load_answers(model)
        arm = grade.arm_for(model, condition)
        passes.append([grade.attempt(answers, points, framing=framing.key, arm=arm) for framing in FRAMINGS])
    level = max(grade.complete_level(frame) for row in passes for frame in row)
    rows = [(model, [grade.to_image(frame, level) for frame in row]) for model, row in zip(models, passes)]
    fig, axes = plt.subplots(len(rows), 6, figsize=(13.2, 2.02 * len(rows) + 0.5), squeeze=False)
    scores = agreement.set_index("model")
    for r, (model, images) in enumerate(rows):
        for c, image in enumerate(images):
            ax = axes[r, c]
            draw_map(ax, image, line_width=0.5)
            if r == 0:
                ax.set_title(FRAMINGS[c].label, fontsize=11, pad=6)
        axes[r, 0].text(-0.05, 0.56, named(model), transform=axes[r, 0].transAxes, ha="right", va="bottom", fontsize=11, fontweight="bold")
        agreeing = f"{scores.loc[model.id, 'cross']:.0%} across units\n{scores.loc[model.id, 'retest']:.0%} asked twice"
        axes[r, 0].text(-0.05, 0.54, agreeing, transform=axes[r, 0].transAxes, ha="right", va="top", fontsize=10, color=SECONDARY)
    fig.tight_layout(h_pad=0.6, w_pad=0.4)
    fig.text(axes[0, 0].get_position().x0, 0.0, "Blue solid, green liquid, orange vapor, pink supercritical. Black lines are IAPWS boundaries and critical lines.\n"
             f"Temperature runs left to right, pressure bottom to top. Maps at {grid.SIDES[level]} × {grid.SIDES[level]}. Agreement at {grid.SIDES[agreement_level]} × {grid.SIDES[agreement_level]}.",
             fontsize=10, color=SECONDARY, va="top", linespacing=1.4)
    save(fig, name)


def thought_map(models: list[Model], points: pd.DataFrame) -> None:
    """Where models that choose when to think spend their reasoning tokens."""
    panels = []
    for model in models:
        answers = load_answers(model)
        arm = grade.arm_for(model, "reasoned")
        chosen = answers[(answers["arm"] == arm) & (answers["framing"] == CANONICAL) & (answers["rep"] == 0)]
        frame = points.merge(chosen[["pid", "reasoning_tokens"]], on="pid")
        level = grade.complete_level(frame.assign(pred=np.int8(0)))
        if level is None:
            continue
        side = grid.SIDES[level]
        image = np.full((side, side), np.nan)
        at = frame[frame["level"] >= level]
        image[at["i"].to_numpy() >> level, at["j"].to_numpy() >> level] = at["reasoning_tokens"].to_numpy()
        panels.append((model, image))
    if not panels:
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(3.2 * len(panels), 3.6), squeeze=False)
    limit = max(np.nanquantile(image, 0.98) for _, image in panels)
    ramp = matplotlib.colors.LinearSegmentedColormap.from_list("ramp", ["#f4f3ee", "#86b6ef", "#2a78d6", "#0d366b"])
    for ax, (model, image) in zip(axes[0], panels):
        shown = ax.imshow(image.T, origin="lower", cmap=ramp, vmin=0, vmax=limit, aspect="auto", interpolation="nearest",
                          extent=[grid.T_MIN, grid.T_MAX, grid.LOGP_MIN, grid.LOGP_MAX])
        for t, logp in CURVES:
            ax.plot(t, logp, color=INK, lw=0.6)
        ax.set_xlim(grid.T_MIN, grid.T_MAX)
        ax.set_ylim(grid.LOGP_MIN, grid.LOGP_MAX)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(named(model), loc="left", fontsize=11, fontweight="bold", pad=4)
    label_axes(axes[0][0])
    bar = fig.colorbar(shown, ax=list(axes[0]), fraction=0.02, pad=0.015, ticks=[0, limit])
    bar.ax.set_yticklabels(["0", f"{limit:.0f}+"], fontsize=9)
    bar.set_label("Reasoning tokens", fontsize=9)
    bar.outline.set_visible(False)
    footnote(fig, "reasoned")
    save(fig, "thought_map")
