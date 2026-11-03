"""Tests for Data Bazaar v0. See CLAUDE.md, section "Tests".

Run from the project folder with:  python3 -m pytest
"""

import pytest

from bazaar.model import make_agent
from bazaar.simulate import lost_deal_formula, run_scenario

SCENARIOS = [("strong", "strong"), ("weak", "weak"), ("weak", "strong"), ("strong", "weak")]

# Step A expected results: (consumer agent, firm agent) -> (beta, p, CS, FS)
STEP_A_EXPECTED = {
    ("strong", "strong"): (1 / 2, 7, 3, 3),
    ("weak", "weak"): (1 / 2, 7, 3, 3),
    ("weak", "strong"): (1 / 3, 6, 2, 4),
    ("strong", "weak"): (2 / 3, 8, 4, 2),
}


@pytest.mark.parametrize("kind_c, kind_f", SCENARIOS)
def test_step_a_exact(kind_c: str, kind_f: str) -> None:
    """Step A (sigma = 0, beta from b) gives exactly the table in CLAUDE.md."""
    agent_c = make_agent(kind_c, sigma=0.0)
    agent_f = make_agent(kind_f, sigma=0.0)
    row = run_scenario(f"{kind_c}/{kind_f}", agent_c, agent_f, n_draws=1).iloc[0]

    beta, p, cs, fs = STEP_A_EXPECTED[(kind_c, kind_f)]
    assert row["beta"] == beta
    assert row["p"] == p
    assert row["CS"] == cs
    assert row["FS"] == fs
    assert row["outcome"] == "good deal"


@pytest.mark.parametrize("kind_c, kind_f", SCENARIOS)
def test_identity_every_draw(kind_c: str, kind_f: str) -> None:
    """CS + FS + lost value = v - r = 6 on every single draw (Step B settings)."""
    agent_c = make_agent(kind_c, b=1.0)
    agent_f = make_agent(kind_f, b=1.0)
    df = run_scenario(f"{kind_c}/{kind_f}", agent_c, agent_f, n_draws=10_000)

    total = df["CS"] + df["FS"] + df["lost_value"]
    # Allow a tiny floating-point rounding error (1e-9 dollars).
    assert (abs(total - 6.0) < 1e-9).all()


@pytest.mark.parametrize("kind_c, kind_f", SCENARIOS)
def test_step_b_lost_rate_matches_formula(kind_c: str, kind_f: str) -> None:
    """Step B simulated lost deal rate is within 0.5 percentage points of the formula."""
    agent_c = make_agent(kind_c, b=1.0)
    agent_f = make_agent(kind_f, b=1.0)
    df = run_scenario(f"{kind_c}/{kind_f}", agent_c, agent_f, n_draws=100_000)

    simulated = (df["outcome"] == "lost deal").mean()
    formula = lost_deal_formula(4.0, 10.0, agent_c.sigma, agent_f.sigma)
    assert abs(simulated - formula) < 0.005
