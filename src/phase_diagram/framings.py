"""How one physical condition is written down: the units control, and the prompt.

Every point is asked in a 2x3 factorial of units. Temperatures carry two decimals and
pressures four significant figures in every unit, so no framing carries more precision
than another. Pressures use the notation a person would write: plain decimals in the
middle of the range, scientific notation at the extremes.
"""

from __future__ import annotations

from dataclasses import dataclass

ATM_PA = 101325.0
C_OFFSET = 273.15

# Closed with the sentence the world-map experiment used: https://outsidetext.substack.com/p/how-does-a-blind-model-see-the-earth
PROMPT = (
    "Pure water at equilibrium, at {temperature} and {pressure}: "
    "solid, liquid, vapor, or supercritical? Do not say anything else."
)

T_WORDS = {"K": "kelvin", "C": "degrees Celsius"}
P_WORDS = {"Pa": "pascals", "MPa": "megapascals", "atm": "atmospheres"}
P_SCALE = {"Pa": 1.0, "MPa": 1e6, "atm": ATM_PA}


@dataclass(frozen=True)
class Framing:
    key: str
    t_unit: str
    p_unit: str

    @property
    def label(self) -> str:
        return f"{'K' if self.t_unit == 'K' else '°C'}, {self.p_unit}"


# The first framing is canonical: it is the wording of the README's example question,
# and it is the one refined to the finest level.
FRAMINGS = (
    Framing("K_MPa", "K", "MPa"),
    Framing("C_MPa", "C", "MPa"),
    Framing("K_Pa", "K", "Pa"),
    Framing("C_Pa", "C", "Pa"),
    Framing("K_atm", "K", "atm"),
    Framing("C_atm", "C", "atm"),
)
BY_KEY = {f.key: f for f in FRAMINGS}
CANONICAL = FRAMINGS[0].key


def format_number(x: float, sig: int = 4) -> str:
    """Four significant figures; scientific notation outside [0.001, 1e6)."""
    mantissa, exponent = f"{x:.{sig - 1}e}".split("e")
    exponent = int(exponent)
    if -3 <= exponent < 6:
        return f"{float(f'{mantissa}e{exponent}'):.{max(sig - 1 - exponent, 0)}f}"
    return f"{mantissa}e{exponent}"


def format_temperature(T_K: float, unit: str) -> str:
    value = T_K if unit == "K" else T_K - C_OFFSET
    return f"{value:.2f} {T_WORDS[unit]}"


def format_pressure(p_Pa: float, unit: str) -> str:
    return f"{format_number(p_Pa / P_SCALE[unit])} {P_WORDS[unit]}"


def render(T_K: float, p_Pa: float, framing: Framing) -> str:
    return PROMPT.format(
        temperature=format_temperature(T_K, framing.t_unit),
        pressure=format_pressure(p_Pa, framing.p_unit),
    )
