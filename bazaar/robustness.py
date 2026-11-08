"""Robustness checks for the learning result (Step B): does it survive other
noise shapes and other deal-zone sizes?

The baseline model is unchanged. This file only swaps in a different way of
drawing the agents' reports, then prices and scores each deal with the same
Bazaar and score() as the baseline.

Settings for every case (all fixed, as agreed with the human):
- 100,000 draws, seed 0, beta = 1/2 (b_c = b_f = 1).
- r = 4 always. v = r + width of the deal zone.
- Firm agent always strong: sigma_f = 0.5.
- The same noise shape is applied to BOTH agents' reports.
"""

import numpy as np
import pandas as pd

from bazaar.model import Bazaar, score
from bazaar.simulate import summarize

SEED = 0
N_DRAWS = 100_000
R = 4.0
SIGMA_F = 0.5          # strong firm agent
SIGMA_C_BASELINE = 3.0  # weak consumer agent

# Part a: noise shapes, weak consumer / strong firm, baseline v = 10.
NOISE_SHAPES = ["normal", "clipped", "lognormal"]
# Part b: normal noise, deal-zone widths v - r and consumer agent sigmas.
WIDTHS = [2.0, 6.0, 10.0]
SIGMAS_C = [1.0, 2.0, 3.0, 5.0]


def shaped_report(true_value: float, sigma: float, shape: str, rng: np.random.Generator) -> float:
    """Return one agent report of true_value with noise of the given shape.

    normal:    true + Normal(0, sigma^2). Same as the baseline Agent.report,
               and it uses the random numbers in the same order, so the same
               seed gives the same draws.
    clipped:   the normal report, but never below $0 (no negative prices).
    lognormal: true * exp(z), z ~ Normal(-s^2/2, s^2), with
               s = sqrt(ln(1 + (sigma/true)^2)). This keeps the report's mean
               equal to the true value and its dollar standard deviation equal
               to sigma, but the report is always positive and skewed right
               (big over-reports are more likely than big under-reports).
    """
    if shape == "normal":
        return true_value + rng.normal(0.0, sigma)
    if shape == "clipped":
        return max(0.0, true_value + rng.normal(0.0, sigma))
    if shape == "lognormal":
        s = np.sqrt(np.log(1.0 + (sigma / true_value) ** 2))
        z = rng.normal(-s**2 / 2.0, s)
        return true_value * np.exp(z)
    raise ValueError(f"unknown noise shape: {shape}")


def run_case(shape: str, r: float, v: float, sigma_c: float, sigma_f: float,
             n_draws: int = N_DRAWS, seed: int = SEED) -> pd.DataFrame:
    """Run n_draws deals with the given noise shape and return a results log.

    Each draw: the consumer's agent reports r_hat, the firm's agent reports
    v_hat (both with the chosen noise shape), the bazaar sets the price with
    beta = 1/2, and the deal is scored with the TRUE r and v.
    """
    rng = np.random.default_rng(seed)
    bazaar = Bazaar()
    b_c, b_f = 1.0, 1.0  # equal bargaining strength, so beta = 1/2

    rows = []
    for draw in range(n_draws):
        r_hat = shaped_report(r, sigma_c, shape, rng)
        v_hat = shaped_report(v, sigma_f, shape, rng)
        p = bazaar.set_price(r_hat, v_hat, b_c, b_f)
        rows.append({
            "scenario": f"{shape}, v-r={v - r:g}, sigma_c={sigma_c:g}",
            "draw": draw, "r": r, "v": v, "sigma_c": sigma_c, "sigma_f": sigma_f,
            "r_hat": r_hat, "v_hat": v_hat, "deal": p is not None,
            "p": np.nan if p is None else p,
            **score(r, v, p),
        })
    return pd.DataFrame(rows)


def case_row(part: str, shape: str, v: float, sigma_c: float) -> dict:
    """Run one robustness case and return its summary row for the table.

    Benchmark CS is what the consumer would get with full information and
    equal power: half the deal zone, (v - r) / 2.
    """
    s = summarize(run_case(shape, R, v, sigma_c, SIGMA_F))
    return {
        "part": part,
        "noise": shape,
        "width": v - R,
        "v": v,
        "sigma_c": sigma_c,
        "sigma_f": SIGMA_F,
        "lost_rate": s["lost_rate"],
        "bad_sale_rate": s["bad_sale_rate"],
        "bad_buy_rate": s["bad_buy_rate"],
        "mean_CS": s["mean_CS"],
        "mean_FS": s["mean_FS"],
        "benchmark_CS": (v - R) / 2.0,
    }


def run_robustness() -> pd.DataFrame:
    """Run part a (noise shapes) and part b (width x sigma grid); return one table."""
    rows = []
    for shape in NOISE_SHAPES:
        rows.append(case_row("a: noise shape", shape, R + 6.0, SIGMA_C_BASELINE))
    for width in WIDTHS:
        for sigma_c in SIGMAS_C:
            rows.append(case_row("b: grid", "normal", R + width, sigma_c))
    return pd.DataFrame(rows)
