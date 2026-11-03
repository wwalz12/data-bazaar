"""Run deals: one at a time, or many in a row for a scenario.

See CLAUDE.md, sections "The model" and "Experiments".
"""

import numpy as np
import pandas as pd
from scipy.stats import norm

from bazaar.model import Agent, Bazaar, Consumer, Firm, score


def run_one_deal(
    consumer: Consumer,
    firm: Firm,
    agent_c: Agent,
    agent_f: Agent,
    bazaar: Bazaar,
    rng: np.random.Generator,
    verbose: bool = False,
) -> dict:
    """Run one deal and return the results log row (the modeler's view).

    Steps: each agent reports its principal's number with noise; the bazaar
    sees only the reports and bargaining strengths and returns a price or
    no deal; then the deal is scored with the true r and v. If verbose is
    True, every step is printed.
    """
    # 1. Each agent reports its principal's number, with noise.
    r_hat = agent_c.report(consumer.r, rng)
    v_hat = agent_f.report(firm.v, rng)

    # 2. The bazaar sees ONLY r_hat, v_hat, b_c, b_f.
    p = bazaar.set_price(r_hat, v_hat, agent_c.b, agent_f.b)
    beta = bazaar.beta(agent_c.b, agent_f.b)

    # 3. The modeler scores the deal with the true numbers.
    result = score(consumer.r, firm.v, p)

    log = {
        "r": consumer.r,
        "v": firm.v,
        "sigma_c": agent_c.sigma,
        "sigma_f": agent_f.sigma,
        "r_hat": r_hat,
        "v_hat": v_hat,
        "b_c": agent_c.b,
        "b_f": agent_f.b,
        "beta": beta,
        "deal": p is not None,
        "p": np.nan if p is None else p,
        **result,
    }

    if verbose:
        price_text = "no deal" if p is None else f"${p:.2f}"
        print(f"  consumer: true r = ${consumer.r:.2f}, agent reports r_hat = ${r_hat:.2f}")
        print(f"  firm:     true v = ${firm.v:.2f}, agent reports v_hat = ${v_hat:.2f}")
        print(f"  bazaar:   beta = {beta:.3f}, price p = {price_text}")
        print(f"  score:    CS = ${result['CS']:.2f}, FS = ${result['FS']:.2f}, "
              f"lost value = ${result['lost_value']:.2f}")
        print(f"  outcome:  {result['outcome']}")

    return log


def run_scenario(
    name: str,
    agent_c: Agent,
    agent_f: Agent,
    n_draws: int,
    seed: int = 0,
    consumer: Consumer | None = None,
    firm: Firm | None = None,
) -> pd.DataFrame:
    """Run the same deal n_draws times and return the results log as a DataFrame.

    Each row is one draw: the scenario name, the draw number, then every
    number from run_one_deal. A fixed seed means the same draws every time.
    """
    rng = np.random.default_rng(seed)
    consumer = consumer if consumer is not None else Consumer()
    firm = firm if firm is not None else Firm()
    bazaar = Bazaar()

    rows = []
    for draw in range(n_draws):
        log = run_one_deal(consumer, firm, agent_c, agent_f, bazaar, rng)
        rows.append({"scenario": name, "draw": draw, **log})
    return pd.DataFrame(rows)


def lost_deal_formula(r: float, v: float, sigma_c: float, sigma_f: float) -> float:
    """Theoretical lost deal rate: Phi(-(v - r) / sqrt(sigma_c^2 + sigma_f^2)).

    A deal is lost when r_hat > v_hat, i.e. when eps_c - eps_f > v - r. The
    difference eps_c - eps_f is Normal with variance sigma_c^2 + sigma_f^2,
    and Phi (the standard normal CDF) gives the chance it exceeds v - r.
    """
    total_sigma = np.sqrt(sigma_c**2 + sigma_f**2)
    if total_sigma == 0:
        # No noise: reports are exact, so a deal is lost only if r > v.
        return 0.0 if r <= v else 1.0
    return float(norm.cdf(-(v - r) / total_sigma))


def summarize(df: pd.DataFrame) -> dict:
    """Summarize one scenario's results log into the Step B numbers.

    Rates are shares of all draws. Means are over all draws, counting 0
    for CS and FS when there was no deal. Also returns mean lost value
    (needed for the surplus split figure) and the formula lost deal rate.
    """
    return {
        "scenario": df["scenario"].iloc[0],
        "lost_rate": (df["outcome"] == "lost deal").mean(),
        "lost_rate_formula": lost_deal_formula(
            df["r"].iloc[0], df["v"].iloc[0], df["sigma_c"].iloc[0], df["sigma_f"].iloc[0]
        ),
        "bad_sale_rate": (df["outcome"] == "bad sale").mean(),
        "bad_buy_rate": (df["outcome"] == "bad buy").mean(),
        "mean_CS": df["CS"].mean(),
        "mean_FS": df["FS"].mean(),
        "mean_total": (df["CS"] + df["FS"]).mean(),
        "mean_lost_value": df["lost_value"].mean(),
        "std_CS": df["CS"].std(),
    }
