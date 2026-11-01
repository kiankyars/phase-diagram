import re

import numpy as np
import pytest

from phase_diagram import framings, grid, truth
from phase_diagram.framings import FRAMINGS, format_number, render


@pytest.fixture(scope="module")
def points():
    return grid.build_grid()


def test_grid_is_deterministic(points):
    assert grid.build_grid().equals(points)


def test_levels_nest_and_stratify(points):
    for level, side in grid.SIDES.items():
        subset = grid.at_level(points, level)
        blocks = set(zip(subset["i"].to_numpy() >> level, subset["j"].to_numpy() >> level))
        assert len(blocks) == len(subset), "at most one point per block"
        assert len(subset) >= 0.98 * side**2, "almost every block is filled"


def test_no_point_is_near_a_boundary(points):
    near = truth.near_boundary(points["T_K"].to_numpy(), points["p_Pa"].to_numpy())
    assert not near.any()


def test_every_class_is_represented_at_every_level(points):
    for level in grid.SIDES:
        assert set(grid.at_level(points, level)["truth"]) == {0, 1, 2, 3}


def test_round_temperatures_appear_only_by_chance(points):
    hundredths = (points["T_K"] * 100).round().astype(int)
    assert (hundredths % 100 == 0).mean() < 0.015
    assert ((hundredths - 27315) % 100 == 0).mean() < 0.015


@pytest.mark.parametrize(
    "x, expected",
    [
        (2.138, "2.138"),
        (2.0, "2.000"),
        (21.1, "21.10"),
        (0.002138, "0.002138"),
        (2138.0, "2138"),
        (213800.0, "213800"),
        (999999.0, "1.000e6"),
        (3.162e-7, "3.162e-7"),
        (1.995e10, "1.995e10"),
    ],
)
def test_format_number(x, expected):
    assert format_number(x) == expected


def test_prompt_matches_the_experiment_wording():
    prompt = render(450.0, 2e6, framings.BY_KEY["K_MPa"])
    assert prompt == (
        "Pure water at equilibrium, at 450.00 kelvin and 2.000 megapascals: "
        "solid, liquid, vapor, or supercritical? Do not say anything else."
    )


def _displayed_condition(prompt: str) -> tuple[float, float]:
    match = re.search(r"at (-?[\d.]+) (kelvin|degrees Celsius) and ([\d.e-]+) (\w+):", prompt)
    value_t, unit_t, value_p, unit_p = match.groups()
    T = float(value_t) + (framings.C_OFFSET if unit_t != "kelvin" else 0.0)
    scale = {"pascals": 1.0, "megapascals": 1e6, "atmospheres": framings.ATM_PA}[unit_p]
    return T, float(value_p) * scale


def test_every_framing_shows_the_same_physical_point(points):
    """What the model reads, converted back, must give the truth label of the point."""
    sample = points.sample(3000, random_state=1)
    for framing in FRAMINGS:
        for row in sample.itertuples():
            T, p = _displayed_condition(render(row.T_K, row.p_Pa, framing))
            assert abs(T - row.T_K) < 1e-9
            assert abs(np.log10(p / row.p_Pa)) < 2e-3
            assert truth.phase(T, p)[0] == row.truth
