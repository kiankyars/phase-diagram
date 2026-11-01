import numpy as np
import pandas as pd
import pytest
from scipy import stats

from phase_diagram import grade, grid, truth
from phase_diagram.framings import CANONICAL, FRAMINGS
from phase_diagram.truth import CLASSES, LIQUID, SOLID, SUPERCRITICAL, VAPOR

LEVEL = 1


@pytest.fixture(scope="module")
def points():
    return grid.build_grid()


def with_pred(points, pred):
    frame = points.copy()
    frame["pred"] = np.asarray(pred, dtype=np.int8)
    return frame


def shifted(points, decades):
    """A model whose whole diagram is displaced along the pressure axis."""
    return with_pred(points, truth.phase(points["T_K"].to_numpy(), points["p_Pa"].to_numpy() / 10**decades))


def test_a_perfect_map_scores_perfectly(points):
    score = grade.score_map(with_pred(points, points["truth"]), LEVEL)
    assert score["balanced_accuracy"] == 1.0
    assert score["macro_f1"] == 1.0
    assert score["ba_familiar"] == 1.0
    assert score["saturation_mae"] == 0
    assert score["sublimation_mae"] == 0
    assert score["melting_mae"] == 0
    assert score["saturation_found"] == 1
    assert score["critical_T_error"] == 0
    assert score["critical_logp_error"] == 0
    assert score["triple_T_error"] == 0
    assert score["triple_logp_error"] == 0
    assert score["critical_T"] == truth.T_C
    assert score["triple_T"] == truth.T_TP


def test_the_regions_cover_every_point_exactly_once(points):
    T, p, true = points["T_K"].to_numpy(), points["p_Pa"].to_numpy(), points["truth"].to_numpy()
    assert set(grade.regions(T, p, true)) == set(grade.REGION_NAMES)
    assert (np.sum(list(grade.regions(T, p, true).values()), axis=0) == 1).all()


def test_a_region_score_reads_only_its_own_points(points):
    T, p = points["T_K"], points["p_Pa"]
    ice_wall = np.where(T < 273.15, SOLID, points["truth"])
    score = grade.score_map(with_pred(points, ice_wall), LEVEL)
    assert score["wrong_cold_vapor"] == 1.0
    assert score["wrong_ordinary_vapor"] == 0.0


def test_the_confusion_table_counts_every_answered_point(points):
    score = grade.score_map(with_pred(points, np.full(len(points), VAPOR)), LEVEL)
    assert sum(score[f"confusion_{name}_vapor"] for name in CLASSES) == grid.SIDES[LEVEL] ** 2
    assert score["confusion_solid_solid"] == 0
    perfect = grade.score_map(with_pred(points, points["truth"]), LEVEL)
    assert perfect["confusion_solid_liquid"] == 0
    assert perfect["confusion_liquid_liquid"] == (grid.at_level(points, LEVEL)["truth"] == LIQUID).sum()


def test_always_vapor_earns_chance_not_its_raw_accuracy(points):
    score = grade.score_map(with_pred(points, np.full(len(points), VAPOR)), LEVEL)
    assert score["accuracy"] > 0.55
    assert score["balanced_accuracy"] == pytest.approx(0.25)


@pytest.mark.parametrize("decades", [-1.5, 1.0, 2.0])
def test_a_displaced_curve_is_measured_in_decades(points, decades):
    score = grade.score_map(shifted(points, decades), LEVEL)
    assert score["saturation_bias"] == pytest.approx(decades, abs=0.05)
    assert score["critical_logp_error"] == pytest.approx(decades, abs=0.3)


def test_a_warmer_solid_is_measured_in_kelvin(points):
    T, p = points["T_K"].to_numpy(), points["p_Pa"].to_numpy()
    warmer = truth.phase(T - 30.0, p)
    warmer[(T > 500) & (warmer == SOLID)] = truth.phase(T, p)[(T > 500) & (warmer == SOLID)]
    score = grade.score_map(with_pred(points, warmer), LEVEL)
    assert score["melting_bias"] == pytest.approx(30.0, abs=8.0)


def test_the_melting_bias_splits_at_the_ice_iii_pressure(points):
    T, p = points["T_K"].to_numpy(), points["p_Pa"].to_numpy()
    tilted = truth.phase(T, p)
    top = grade.isobar_pressures(points["j"].to_numpy() >> LEVEL, LEVEL) >= np.log10(truth.P_IH_III)
    tilted[top] = truth.phase(T[top] - 30.0, p[top])
    score = grade.score_map(with_pred(points, tilted), LEVEL)
    assert score["melting_bias_low"] == 0
    assert score["melting_bias_high"] == pytest.approx(30.0, abs=12.0)


def test_one_stray_answer_does_not_move_a_boundary():
    offset = np.linspace(-2, 2, 41)
    below = offset < 0
    assert grade.best_split(offset, below) == pytest.approx(0, abs=0.1)
    below[35] = True
    assert grade.best_split(offset, below) == pytest.approx(0, abs=0.1)


def test_melting_curve_passes_through_the_iapws_anchors():
    assert grade.t_melt(101325.0) == pytest.approx(273.15, abs=0.02)
    assert grade.t_melt(208.566e6) == pytest.approx(251.165, abs=0.01)
    assert grade.t_melt(632.4e6) == pytest.approx(273.31, abs=0.01)
    assert grade.t_melt(2216e6) == pytest.approx(355.0, abs=0.01)


def test_coherence_never_needs_the_truth_and_rewards_shape_not_rightness(points):
    right = grade.coherence(grade.to_image(with_pred(points, points["truth"]), LEVEL))
    tidy_but_wrong = grade.coherence(grade.to_image(shifted(points, 3.0), LEVEL))
    noise = np.random.default_rng(0).integers(0, 4, len(points))
    scrambled = grade.coherence(grade.to_image(with_pred(points, noise), LEVEL))
    assert right["one_region"] == 1.0
    assert right["ordered"] > 0.995
    assert tidy_but_wrong["ordered"] > 0.995
    assert scrambled["ordered"] < 0.8
    assert scrambled["one_region"] < 0.3


def test_physical_change_looks_only_at_neighbors_that_differ(points):
    right = grade.coherence(grade.to_image(with_pred(points, points["truth"]), LEVEL))
    noise = np.random.default_rng(0).integers(0, 4, len(points))
    scrambled = grade.coherence(grade.to_image(with_pred(points, noise), LEVEL))
    one_color = grade.coherence(grade.to_image(with_pred(points, np.full(len(points), VAPOR)), LEVEL))
    assert right["physical_change"] > 0.9
    assert scrambled["physical_change"] == pytest.approx(0.5, abs=0.05)
    assert one_color["ordered"] == 1.0 and np.isnan(one_color["physical_change"])


def answer_table(points, level, predict):
    """Answers in every framing at a level, for `predict(framing_key, rep) -> codes`."""
    chosen = points[points["level"] >= level]
    rows = []
    for framing in FRAMINGS:
        for rep in (0, 1):
            labels = [CLASSES[code] for code in predict(framing.key, rep, chosen)]
            rows.append(
                pd.DataFrame({"arm": "direct", "framing": framing.key, "pid": chosen["pid"], "rep": rep, "label": labels, "status": "schema"})
            )
    return pd.concat(rows, ignore_index=True)


def test_units_gap_is_zero_for_a_model_that_ignores_notation(points):
    answers = answer_table(points, 2, lambda framing, rep, chosen: chosen["truth"].to_numpy())
    score = grade.score_units(answers, points, 2)
    assert score["cross"] == 1.0
    assert score["retest"] == 1.0
    assert score["gap"] == 0.0


def test_units_gap_separates_a_notation_effect_from_plain_noise(points):
    def noisy(seed_offset):
        def predict(framing, rep, chosen):
            rng = np.random.default_rng(abs(hash((framing, rep, seed_offset))) % 2**32)
            flip = rng.random(len(chosen)) < 0.1
            return np.where(flip, (chosen["truth"].to_numpy() + 1) % 4, chosen["truth"].to_numpy())

        return predict

    just_noise = grade.score_units(answer_table(points, 2, noisy(0)), points, 2)
    assert abs(just_noise["gap"]) < 0.03

    def confused_by_celsius(framing, rep, chosen):
        codes = chosen["truth"].to_numpy()
        return (codes + 1) % 4 if framing.startswith("C") else codes

    confused = grade.score_units(answer_table(points, 2, confused_by_celsius), points, 2)
    assert confused["gap"] == pytest.approx(0.6)
    assert confused["gap_low"] > 0.5
    assert confused["cross_p_unit"] == 1.0
    assert confused["cross_t_unit"] == 0.0


def test_unit_scores_read_the_arm_they_are_asked_for(points):
    steady = answer_table(points, 2, lambda framing, rep, chosen: chosen["truth"].to_numpy())
    swayed = answer_table(points, 2, lambda framing, rep, chosen: (chosen["truth"].to_numpy() + framing.startswith("C")) % 4)
    answers = pd.concat([steady, swayed.assign(arm="reasoned")], ignore_index=True)
    assert grade.score_units(answers, points, 2, arm="direct")["gap"] == 0.0
    assert grade.score_units(answers, points, 2, arm="reasoned")["gap"] > 0.5
    assert len(grade.score_framings(answers, points, 2, arm="reasoned")) == 6


def test_a_perfect_map_draws_the_true_curves(points):
    lines = grade.boundary_lines(with_pred(points, points["truth"]), LEVEL)
    t, logp = lines["saturation"].T
    assert logp == pytest.approx(np.log10(truth.p_sat(t)))
    t, logp = lines["melting"].T
    assert t == pytest.approx(grade.t_melt(10**logp))


def test_a_map_with_no_boundary_on_a_line_draws_nothing_there_and_says_so(points):
    frame = with_pred(points, np.full(len(points), VAPOR))
    assert not len(grade.boundary_lines(frame, LEVEL)["saturation"])
    assert grade.score_map(frame, LEVEL)["saturation_found"] == 0


def test_an_error_of_one_cell_is_not_hidden_by_the_cell_size(points):
    frame = with_pred(points, points["truth"])
    rows = grade.boundary_splits(frame.assign(pred=truth.phase(frame["T_K"].to_numpy(), frame["p_Pa"].to_numpy() / 10**0.4)), LEVEL)
    assert np.abs(rows["saturation"][:, 1]).mean() == pytest.approx(0.4, abs=0.1)


def test_a_perfect_map_boils_and_freezes_where_water_does(points):
    boiling, freezing = grade.everyday_points(with_pred(points, points["truth"]), LEVEL)
    assert boiling == pytest.approx(373.12, abs=0.01)
    assert freezing == pytest.approx(273.15, abs=0.01)


def test_a_boiling_reading_needs_liquid_below_its_split(points):
    near = np.abs(np.log10(points["p_Pa"]) - np.log10(grade.ATMOSPHERE)) < 0.2
    chosen = grade.at_level(points, LEVEL).index
    liquid = points.index[near & (points["T_K"] > 300) & (points["T_K"] < 370) & points.index.isin(chosen)]
    too_few = np.where(points["T_K"] < 300, SOLID, VAPOR)
    too_few[liquid[:2]] = LIQUID
    assert np.isnan(grade.everyday_points(with_pred(points, too_few), LEVEL)[0])
    enough = np.where(points["T_K"] < 300, SOLID, VAPOR)
    enough[liquid[:3]] = LIQUID
    assert grade.everyday_points(with_pred(points, enough), LEVEL)[0] > 300


def test_the_word_supercritical_is_forgiven_only_on_fluid(points):
    said = np.where(np.isin(points["truth"], [VAPOR, LIQUID]) & (points["p_Pa"] > 1e6), SUPERCRITICAL, points["truth"])
    scores = grade.score_map(with_pred(points, said), LEVEL)
    assert scores["balanced_accuracy"] < scores["ba_hot_gas"] <= scores["ba_hot_fluid"] <= scores["ba_any_fluid"]
    perfect = grade.score_map(with_pred(points, points["truth"]), LEVEL)
    assert perfect["ba_hot_gas"] == perfect["ba_any_fluid"] == perfect["balanced_accuracy"] == 1.0
    assert scores["misnamed_compressed_liquid"] == 0 < scores["wrong_compressed_liquid"]


def test_everyday_points_stay_unread_where_the_map_cannot_say(points):
    assert np.isnan(grade.everyday_points(with_pred(points, points["truth"]), grid.MAX_LEVEL)).all()
    boiling, freezing = grade.everyday_points(with_pred(points, np.full(len(points), VAPOR)), LEVEL)
    assert np.isnan(boiling) and np.isnan(freezing)
    only_ice = grade.everyday_points(with_pred(points, np.where(points["T_K"] < 300, SOLID, VAPOR)), LEVEL)
    assert np.isnan(only_ice[0]) and only_ice[1] == pytest.approx(300, abs=10)


def test_an_empty_line_has_no_split():
    assert np.isnan(grade.best_split(np.array([]), np.array([], dtype=bool)))


def test_regions_that_never_touch_do_not_meet():
    image = np.full((64, 64), SOLID, dtype=np.int8)
    image[:8, :] = LIQUID
    image[-8:, :] = VAPOR
    assert np.isnan(grade.meeting_point(image, (SOLID, LIQUID, VAPOR))).all()
    assert np.isnan(grade.meeting_point(image, (SOLID, LIQUID, truth.SUPERCRITICAL))).all()


def test_a_stray_answer_cannot_move_a_meeting_point(points):
    image = grade.to_image(with_pred(points, points["truth"]), LEVEL)
    before = grade.meeting_point(image, (LIQUID, VAPOR, truth.SUPERCRITICAL))
    image[2, 60] = truth.SUPERCRITICAL
    image[3, 3] = LIQUID
    assert grade.meeting_point(image, (LIQUID, VAPOR, truth.SUPERCRITICAL)) == before


def test_a_phase_too_thin_to_weigh_gives_no_balanced_accuracy():
    assert np.isnan(grade.supported_ba(np.array([0] * 20 + [2]), np.zeros(21, dtype=int)))
    assert grade.supported_ba(np.array([0] * 20 + [2] * 5), np.zeros(25, dtype=int)) == 0.5


def test_two_unusable_replies_do_not_count_as_agreement(points):
    def fails_everywhere(framing, rep, chosen):
        return chosen["truth"].to_numpy()

    answers = answer_table(points, 2, fails_everywhere)
    first = points[points["level"] >= 2]["pid"].iloc[:40].to_numpy()
    answers.loc[answers["pid"].isin(first), "label"] = None
    score = grade.score_units(answers, points, 2)
    assert score["cross"] == 1.0
    assert score["retest"] == 1.0
    wrong_in_one = answers.copy()
    wrong_in_one.loc[wrong_in_one["framing"] == CANONICAL, "label"] = "vapor"
    assert grade.score_units(wrong_in_one, points, 2)["cross"] < 1.0


def test_thought_focus_compares_points_near_and_far_from_the_boundaries(points):
    distance = grade.boundary_distance(points)
    near = distance <= 1 / 16
    answers = pd.DataFrame(
        {
            "arm": "direct",
            "framing": CANONICAL,
            "rep": 0,
            "pid": points["pid"],
            "label": [CLASSES[code] for code in points["truth"]],
            "reasoning_tokens": np.where(near, 50, 0),
        }
    )
    focus = grade.thought_focus(answers, points, "direct")
    assert focus["thought_near"] == 1.0
    assert focus["thought_far"] == 0.0
    assert focus["right_when_thought"] == 1.0


def test_attempt_marks_unusable_replies_as_invalid(points):
    chosen = points[points["level"] >= 3]
    answers = pd.DataFrame(
        {"arm": "direct", "framing": CANONICAL, "pid": chosen["pid"], "rep": 0, "label": None, "status": "invalid"}
    )
    frame = grade.attempt(answers, points)
    assert len(frame) == len(chosen)
    assert (frame["pred"] == grade.INVALID).all()
    assert grade.complete_level(frame) == 3


def test_the_four_fact_sketch_is_good_near_home_and_blind_to_ice_above_0_c(points):
    frame = with_pred(points, grade.sketch_phase(points["T_K"], points["p_Pa"]))
    score = grade.score_map(frame, LEVEL)
    assert 0.75 < score["balanced_accuracy"] < 0.98
    assert score["ba_familiar"] > 0.95
    assert score["recall_solid"] < 0.8
    assert score["recall_vapor"] > 0.95
    assert grade.sketch_phase(np.array([373.15]), np.array([2e5]))[0] == LIQUID
    assert grade.sketch_phase(np.array([373.15]), np.array([5e4]))[0] == VAPOR


def test_ice_ceiling_reads_a_flat_lid_and_the_truth_falls_within_the_climb(points):
    assert 2.8e9 < grade.ice_ceiling(with_pred(points, points["truth"]), LEVEL) < 12.4e9
    lid = np.where(points["p_Pa"] > 5e8, SOLID, LIQUID)
    assert grade.ice_ceiling(with_pred(points, lid), LEVEL) == pytest.approx(5e8, rel=0.3)


def test_ice_ceiling_sits_at_the_top_of_a_map_with_no_ice(points):
    assert grade.ice_ceiling(with_pred(points, np.full(len(points), LIQUID)), LEVEL) > 1e10


def test_paired_effect_reads_a_shift_across_models():
    second = pd.Series([0.5, 0.6, 0.7, 0.8])
    result = grade.paired_effect(second + 0.1, second)
    assert result["mean"] == pytest.approx(0.1)
    assert result["low"] == pytest.approx(0.1) and result["high"] == pytest.approx(0.1)
    assert result["first_higher"] == result["n"] == 4


def test_median_slopes_match_theil_sen_on_each_resample():
    x = np.array([0.0, 0.3, 0.9, 1.4, 2.0])
    y = np.array([1.0, 0.7, 1.9, 1.1, 2.6])
    picks = np.random.default_rng(0).integers(5, size=(200, 5))
    spread_out = [pick for pick in picks if np.ptp(x[pick]) > 0]
    expected = [stats.theilslopes(y[pick], x[pick])[0] for pick in spread_out]
    slopes = grade.median_slopes(x, y, picks)
    assert slopes[~np.isnan(slopes)] == pytest.approx(expected)
    assert np.isnan(slopes).sum() == len(picks) - len(expected)


def test_every_resample_lists_each_multiset_once_with_its_probability():
    picks, weights = grade.every_resample(5)
    assert len(picks) == 126 and len({tuple(row) for row in picks}) == 126
    assert weights.sum() == pytest.approx(1)
    one_item_five_times = 1 / 5**5
    assert weights[0] == pytest.approx(one_item_five_times)
    assert weights[picks.tolist().index([0, 1, 2, 3, 4])] == pytest.approx(120 / 5**5)


def test_the_listed_bootstrap_agrees_with_drawing_resamples():
    x = np.array([0.0, 0.3, 0.9, 1.4, 2.0, 2.2, 2.9])
    y = np.array([1.0, 0.7, 1.9, 1.1, 2.6, 2.1, 3.3])
    slopes, weights = grade.trend_slopes(x, y, 0)
    drawn = grade.median_slopes(x, y, np.random.default_rng(1).integers(7, size=(400_000, 7)))
    for q in (0.025, 0.5, 0.975):
        assert grade.weighted_quantile(slopes, weights, q) == pytest.approx(np.nanquantile(drawn, q), abs=0.03)


def test_weighted_quantile_takes_the_first_value_to_reach_the_share():
    values, weights = np.array([3.0, 1.0, 2.0]), np.array([0.5, 0.25, 0.25])
    assert grade.weighted_quantile(values, weights, 0.2) == 1.0
    assert grade.weighted_quantile(values, weights, 0.5) == 2.0
    assert grade.weighted_quantile(values, weights, 0.75) == 3.0


def test_trend_reads_the_direction_of_a_series():
    dates = pd.Series(pd.date_range("2025-01-01", periods=8, freq="90D").astype(str))
    rising = pd.Series(np.linspace(0.4, 0.8, 8))
    result = grade.trend(dates, rising)
    assert result["spearman"] == pytest.approx(1.0)
    assert result["per_year"] > 0
    assert result["per_year_low"] <= result["per_year"] <= result["per_year_high"]
    assert LIQUID == 1 and SOLID == 0


def test_the_shared_level_is_the_finest_every_model_has_in_both_conditions():
    scores = pd.DataFrame(
        {
            "condition": ["instant", "instant", "instant", "reasoned", "reasoned"],
            "model": ["a", "a", "b", "a", "a"],
            "level": [0, 1, 1, 2, 3],
        }
    )
    assert grade.shared_level(scores) == 2
