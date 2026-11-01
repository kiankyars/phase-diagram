"""Ground-truth equilibrium phase of pure water from IAPWS reference equations.

Sources
-------
R14-08(2011)   Melting curves of ice Ih, III, V, VI, VII and the sublimation curve.
               Equation numbers below refer to that release.
SR1-86(1992)   Saturation vapor pressure (Wagner & Pruss), consistent with IAPWS-95.
R2-83(1992)    Critical point constants, as adopted by IAPWS-95.

Label convention
----------------
solid          above the sublimation pressure or above a melting curve (any ice polymorph)
liquid         between saturation and melting curves, including T < Tc with p > pc
vapor          below sublimation/saturation, including T > Tc with p < pc
supercritical  T > Tc and p > pc, below the ice VII melting curve

These are the conventional textbook regions. Only the solid/liquid/vapor
boundaries are true coexistence curves; the Tc and pc lines that bound
"supercritical" are a naming convention, and are buffered in the same way.
"""

from __future__ import annotations

import numpy as np

SOLID, LIQUID, VAPOR, SUPERCRITICAL = 0, 1, 2, 3
CLASSES = ("solid", "liquid", "vapor", "supercritical")

T_C = 647.096  # K, critical temperature
P_C = 22.064e6  # Pa, critical pressure
T_TP = 273.16  # K, solid-liquid-vapor triple point
P_TP = 611.657  # Pa

# Triple points with the liquid, R14-08 Table 1.
T_IH_III, P_IH_III = 251.165, 208.566e6
T_III_V, P_III_V = 256.164, 350.1e6
T_V_VI, P_V_VI = 273.31, 632.4e6
T_VI_VII, P_VI_VII = 355.0, 2216e6
T_VII_MAX = 715.0  # highest temperature with measurements on the ice VII curve

# The buffer around every boundary. 0.03 decades is +7.2% and -6.7% in pressure, against
# the largest uncertainty IAPWS states for any of these curves (7%, ice VII). No grid
# point falls in the small gap on the low side: see curve_margins in scripts/claims.py.
BUFFER_T = 2.0  # K
BUFFER_LOGP = 0.03  # decades of pressure


def p_melt_ih(T):
    """Eq. (1): melting pressure of ice Ih, 251.165 K to 273.16 K."""
    th = np.asarray(T, dtype=float) / T_TP
    pi = (
        1
        + 0.119539337e7 * (1 - th**3)
        + 0.808183159e5 * (1 - th**25.75)
        + 0.333826860e4 * (1 - th**103.75)
    )
    return P_TP * pi


def p_melt_iii(T):
    """Eq. (2): melting pressure of ice III, 251.165 K to 256.164 K."""
    th = np.asarray(T, dtype=float) / T_IH_III
    return P_IH_III * (1 - 0.299948 * (1 - th**60))


def p_melt_v(T):
    """Eq. (3): melting pressure of ice V, 256.164 K to 273.31 K."""
    th = np.asarray(T, dtype=float) / T_III_V
    return P_III_V * (1 - 1.18721 * (1 - th**8))


def p_melt_vi(T):
    """Eq. (4): melting pressure of ice VI, 273.31 K to 355 K."""
    th = np.asarray(T, dtype=float) / T_V_VI
    return P_V_VI * (1 - 1.07476 * (1 - th**4.6))


def p_melt_vii(T):
    """Eq. (5): melting pressure of ice VII, 355 K to 715 K."""
    th = np.asarray(T, dtype=float) / T_VI_VII
    ln_pi = (
        0.173683e1 * (1 - 1 / th)
        - 0.544606e-1 * (1 - th**5)
        + 0.806106e-7 * (1 - th**22)
    )
    return P_VI_VII * np.exp(ln_pi)


def p_subl(T):
    """Eq. (6): sublimation pressure of ice Ih, 50 K to 273.16 K."""
    th = np.asarray(T, dtype=float) / T_TP
    ln_pi = (
        -0.212144006e2 * th**0.333333333e-2
        + 0.273203819e2 * th**0.120666667e1
        - 0.610598130e1 * th**0.170333333e1
    ) / th
    return P_TP * np.exp(ln_pi)


def p_sat(T):
    """Saturation vapor pressure, 273.16 K to Tc (Wagner & Pruss, SR1-86)."""
    T = np.asarray(T, dtype=float)
    tau = 1 - T / T_C
    ln_pi = (T_C / T) * (
        -7.85951783 * tau
        + 1.84408259 * tau**1.5
        - 11.7866497 * tau**3
        + 22.6807411 * tau**3.5
        - 15.9618719 * tau**4
        + 1.80122502 * tau**7.5
    )
    return P_C * np.exp(ln_pi)


def p_melt_high(T):
    """Pressure above which a high-pressure ice is stable, for T >= 251.165 K."""
    T = np.asarray(T, dtype=float)
    out = np.full(T.shape, np.inf)
    for lo, hi, curve in (
        (T_IH_III, T_III_V, p_melt_iii),
        (T_III_V, T_V_VI, p_melt_v),
        (T_V_VI, T_VI_VII, p_melt_vi),
        (T_VI_VII, T_VII_MAX, p_melt_vii),
    ):
        mask = (T >= lo) & (T < hi)
        out[mask] = curve(T[mask])
    return out


def phase(T, p):
    """Equilibrium phase code at temperature T (K) and pressure p (Pa)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=float))
    T, p = np.broadcast_arrays(T, p)
    out = np.full(T.shape, SOLID, dtype=np.int8)

    cold = T <= T_TP
    out[cold & (p < p_subl(np.where(cold, T, T_TP)))] = VAPOR

    wedge = (T >= T_IH_III) & (T <= T_TP)
    t_w = np.where(wedge, T, T_TP)
    out[wedge & (p > p_melt_ih(t_w)) & (p <= p_melt_high(t_w))] = LIQUID

    warm = (T > T_TP) & (T < T_C)
    t_h = np.where(warm, T, T_TP + 1)
    out[warm & (p <= p_melt_high(t_h))] = LIQUID
    out[warm & (p < p_sat(t_h))] = VAPOR

    hot = T >= T_C
    below_ice = p <= p_melt_high(T)
    out[hot & below_ice] = SUPERCRITICAL
    out[hot & below_ice & (p < P_C)] = VAPOR
    return out


def near_boundary(T, p, buffer_T=BUFFER_T, buffer_logp=BUFFER_LOGP):
    """True if the phase changes anywhere in the buffer box around (T, p)."""
    T = np.atleast_1d(np.asarray(T, dtype=float))
    p = np.atleast_1d(np.asarray(p, dtype=float))
    base = phase(T, p)
    near = np.zeros(base.shape, dtype=bool)
    for dT in (-buffer_T, 0.0, buffer_T):
        for dlogp in (-buffer_logp, 0.0, buffer_logp):
            near |= phase(T + dT, p * 10**dlogp) != base
    return near


def boundary_curves(n=400):
    """Boundary polylines as {name: (T array, p array)} for drawing overlays."""
    curves = {
        "sublimation": (t := np.linspace(120.0, T_TP, n), p_subl(t)),
        "saturation": (t := np.linspace(T_TP, T_C, n), p_sat(t)),
        "melt_ih": (t := np.linspace(T_IH_III, T_TP, n), p_melt_ih(t)),
        "melt_iii": (t := np.linspace(T_IH_III, T_III_V, n), p_melt_iii(t)),
        "melt_v": (t := np.linspace(T_III_V, T_V_VI, n), p_melt_v(t)),
        "melt_vi": (t := np.linspace(T_V_VI, T_VI_VII, n), p_melt_vi(t)),
        "melt_vii": (t := np.linspace(T_VI_VII, T_VII_MAX, n), p_melt_vii(t)),
    }
    return curves
