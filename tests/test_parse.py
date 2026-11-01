import pytest

from phase_diagram.parse import parse_answer


@pytest.mark.parametrize(
    "text, label, status",
    [
        ('{"phase":"liquid"}', "liquid", "schema"),
        ('{"phase": "Supercritical"}', "supercritical", "schema"),
        ('{"phase": "plasma"}', None, "invalid"),
        ('{"phase": "liq', None, "invalid"),
        ("Liquid", "liquid", "exact"),
        ("  vapor.\n", "vapor", "exact"),
        ("**Supercritical**", "supercritical", "exact"),
        ("SOLID", "solid", "exact"),
        ("Gas", "vapor", "synonym"),
        ("Vapour", "vapor", "synonym"),
        ("Ice", "solid", "synonym"),
        ("The water is a liquid at this point.", "liquid", "embedded"),
        ("Supercritical fluid", "supercritical", "embedded"),
        ("It is either solid or liquid", None, "invalid"),
        ("I cannot answer that", None, "invalid"),
        ("", None, "invalid"),
        (None, None, "invalid"),
    ],
)
def test_parse_answer(text, label, status):
    assert parse_answer(text) == (label, status)
