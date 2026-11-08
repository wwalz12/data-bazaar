"""Run every v0 experiment: Step A, Step B and the noise sweep.

Saves results/*.csv and figures/*.png, and prints the tables.
See CLAUDE.md, sections "Experiments" and "Figures".

Run from the project folder with:  python3 run_v0.py
"""

import os

import numpy as np
import pandas as pd

from bazaar.model import Agent, Consumer, Firm, make_agent
from bazaar.plots import (plot_consumer_gain_hist, plot_noise_sweep,
                          plot_outcome_map, plot_surplus_split)
from bazaar.robustness import run_robustness
from bazaar.simulate import lost_deal_formula, run_scenario, summarize

SEED = 0
N_DRAWS = 100_000
SCENARIOS = [("strong", "strong"), ("weak", "strong"), ("strong", "weak"), ("weak", "weak")]
SWEEP_SIGMAS = np.arange(0, 6.25, 0.25)  # 0, 0.25, ..., 6

R = Consumer().r
V = Firm().v


def run_step_a() -> pd.DataFrame:
    """Step A, bargaining only: sigma = 0, beta from b. One draw is enough (no noise)."""
    rows = []
    for kind_c, kind_f in SCENARIOS:
        agent_c = make_agent(kind_c, sigma=0.0)
        agent_f = make_agent(kind_f, sigma=0.0)
        df = run_scenario(f"{kind_c}/{kind_f}", agent_c, agent_f, n_draws=1, seed=SEED)
        rows.append(df.iloc[0])
    return pd.DataFrame(rows)[["scenario", "beta", "p", "CS", "FS"]].reset_index(drop=True)


def run_step_b() -> dict[str, pd.DataFrame]:
    """Step B, learning only: b = 1 for both (so beta = 1/2), sigma from type.

    Returns a dict mapping scenario name to its full results log.
    """
    logs = {}
    for kind_c, kind_f in SCENARIOS:
        agent_c = make_agent(kind_c, b=1.0)
        agent_f = make_agent(kind_f, b=1.0)
        name = f"{kind_c}/{kind_f}"
        logs[name] = run_scenario(name, agent_c, agent_f, n_draws=N_DRAWS, seed=SEED)
    return logs


def run_sweep() -> pd.DataFrame:
    """Noise sweep: both agents use the same sigma, beta = 1/2. Lost deal rate vs formula."""
    rows = []
    for sigma in SWEEP_SIGMAS:
        agent = Agent(kind="sweep", sigma=float(sigma), b=1.0)
        df = run_scenario(f"sigma={sigma:g}", agent, agent, n_draws=N_DRAWS, seed=SEED)
        rows.append({
            "sigma": sigma,
            "lost_rate": (df["outcome"] == "lost deal").mean(),
            "lost_rate_formula": lost_deal_formula(R, V, sigma, sigma),
        })
    return pd.DataFrame(rows)


def pct(x: float) -> str:
    """Format a share (0.024) as a percentage string ('2.40%')."""
    return f"{x:.2%}"


def money(x: float) -> str:
    """Format a dollar amount to two decimals."""
    return f"{x:.2f}"


def main() -> None:
    """Run all experiments, save CSVs and figures, print the tables."""
    os.makedirs("results", exist_ok=True)
    os.makedirs("figures", exist_ok=True)

    # Step A
    step_a = run_step_a()
    step_a.to_csv("results/step_a.csv", index=False)
    print("\nStep A: bargaining only (sigma = 0)")
    print(step_a.to_string(index=False, formatters={"beta": "{:.3f}".format,
                                                    "p": money, "CS": money, "FS": money}))

    # Step B
    logs = run_step_b()
    all_draws = pd.concat(logs.values(), ignore_index=True)
    all_draws.to_csv("results/step_b_draws.csv", index=False)
    summary = pd.DataFrame([summarize(df) for df in logs.values()])
    summary.to_csv("results/step_b_summary.csv", index=False)
    print(f"\nStep B: learning only (beta = 1/2, {N_DRAWS:,} draws per scenario, seed {SEED})")
    print(summary.to_string(index=False, formatters={
        "lost_rate": pct, "lost_rate_formula": pct, "bad_sale_rate": pct, "bad_buy_rate": pct,
        "mean_CS": money, "mean_FS": money, "mean_total": money,
        "mean_lost_value": money, "std_CS": money}))

    # Noise sweep
    sweep = run_sweep()
    sweep.to_csv("results/sweep.csv", index=False)
    print(f"\nNoise sweep: same sigma for both agents (beta = 1/2, {N_DRAWS:,} draws each)")
    print(sweep.to_string(index=False, formatters={
        "sigma": "{:.2f}".format, "lost_rate": pct, "lost_rate_formula": pct}))

    # Robustness: other noise shapes and deal-zone widths
    robust = run_robustness()
    robust.to_csv("results/robustness.csv", index=False)
    print(f"\nRobustness: weak consumer / strong firm agent (beta = 1/2, {N_DRAWS:,} draws each)")
    print(robust.to_string(index=False, formatters={
        "width": "{:g}".format, "v": "{:g}".format, "sigma_c": "{:g}".format,
        "sigma_f": "{:g}".format, "lost_rate": pct, "bad_sale_rate": pct,
        "bad_buy_rate": pct, "mean_CS": money, "mean_FS": money, "benchmark_CS": money}))

    # Figures
    plot_outcome_map(logs["weak/weak"].head(2000), R, V, "figures/outcome_map.png")
    plot_surplus_split(summary, "figures/surplus_split.png")
    plot_consumer_gain_hist(logs["strong/strong"], logs["weak/strong"],
                            "figures/consumer_gain_hist.png")
    plot_noise_sweep(sweep, R, V, "figures/noise_sweep.png")
    print("\nSaved results/*.csv and figures/*.png")


if __name__ == "__main__":
    main()
