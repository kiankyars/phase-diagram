import numpy as np
import pytest
from iapws import IAPWS95
from iapws._iapws import _Melting_Pressure, _Sublimation_Pressure

from phase_diagram import truth
from phase_diagram.truth import LIQUID, SOLID, SUPERCRITICAL, VAPOR


# R14-08(2011) Table 3, computer-program verification values.
@pytest.mark.parametrize(
    "curve, T, p_MPa, digits",
    [
        (truth.p_melt_ih, 260.0, 138.268, 3),
        (truth.p_melt_iii, 254.0, 268.685, 3),
        (truth.p_melt_v, 265.0, 479.640, 3),
        (truth.p_melt_vi, 320.0, 1356.76, 2),
        (truth.p_melt_vii, 550.0, 6308.71, 2),
    ],
)
def test_melting_curves_match_iapws_table_3(curve, T, p_MPa, digits):
    assert round(float(curve(T)) / 1e6, digits) == pytest.approx(p_MPa, abs=10.0**-digits)


def test_sublimation_matches_iapws_table_3():
    assert float(truth.p_subl(230.0)) / 1e6 == pytest.approx(8.94735e-6, rel=1e-5)


def test_saturation_matches_iapws95_equilibrium_solve():
    for T in (280.0, 373.15, 450.0, 550.0, 640.0):
        assert float(truth.p_sat(T)) == pytest.approx(IAPWS95(T=T, x=0).P * 1e6, rel=5e-4)


def test_curves_match_independent_implementation():
    assert float(truth.p_melt_vi(300.0)) / 1e6 == pytest.approx(
        _Melting_Pressure(300.0, ice="VI"), rel=1e-6
    )
    assert float(truth.p_subl(200.0)) / 1e6 == pytest.approx(
        _Sublimation_Pressure(200.0), rel=1e-6
    )


def test_curves_meet_at_triple_points():
    assert float(truth.p_subl(truth.T_TP)) == pytest.approx(truth.P_TP, rel=1e-6)
    assert float(truth.p_sat(truth.T_TP)) == pytest.approx(truth.P_TP, rel=1e-4)
    assert float(truth.p_melt_ih(truth.T_IH_III)) == pytest.approx(truth.P_IH_III, rel=1e-5)
    assert float(truth.p_melt_v(truth.T_III_V)) == pytest.approx(truth.P_III_V, rel=1e-5)
    assert float(truth.p_melt_vi(truth.T_V_VI)) == pytest.approx(truth.P_V_VI, rel=1e-5)
    assert float(truth.p_melt_vii(truth.T_VI_VII)) == pytest.approx(truth.P_VI_VII, rel=1e-5)


@pytest.mark.parametrize(
    "T, p, expected",
    [
        (450.0, 2e6, LIQUID),  # a worked example: 450 K at 2 MPa
        (373.15, 0.5e5, VAPOR),
        (373.15, 2e5, LIQUID),
        (300.0, 1e5, LIQUID),
        (260.0, 1e5, SOLID),
        (260.0, 200e6, LIQUID),  # the liquid wedge above ice Ih
        (260.0, 500e6, SOLID),  # ice V
        (200.0, 1e-2, VAPOR),
        (200.0, 1.0, SOLID),
        (500.0, 1e10, SOLID),  # ice VII
        (640.0, 30e6, LIQUID),  # compressed liquid: p > pc but T < Tc
        (700.0, 10e6, VAPOR),  # T > Tc but p < pc
        (700.0, 30e6, SUPERCRITICAL),
        (700.0, 1.99e10, SOLID),  # ice VII above the critical temperature
        (890.0, 1.99e10, SUPERCRITICAL),
    ],
)
def test_phase_labels(T, p, expected):
    assert truth.phase(T, p)[0] == expected


def test_near_boundary_flags_coexistence_and_clears_interior():
    assert truth.near_boundary(373.15, 101325.0)[0]
    assert truth.near_boundary(truth.T_C + 0.5, 5e7)[0]
    assert truth.near_boundary(800.0, truth.P_C * 1.01)[0]
    assert not truth.near_boundary(300.0, 1e6)[0]


def test_phase_is_vectorized():
    T = np.array([300.0, 300.0])
    p = np.array([1e5, 1e1])
    assert truth.phase(T, p).tolist() == [LIQUID, VAPOR]


def test_the_triple_point_temperature_is_vapor_below_the_triple_pressure():
    assert truth.phase(truth.T_TP, 100.0)[0] == truth.VAPOR
    assert truth.phase(truth.T_TP, 1e5)[0] == truth.LIQUID
