"""The four figures for Data Bazaar v0. See CLAUDE.md, section "Figures".

Each function takes results (DataFrames) and saves one PNG.
"""

import matplotlib

matplotlib.use("Agg")  # draw to files, not to a window
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# One fixed color per outcome, so every figure uses the same colors.
OUTCOME_COLORS = {
    "good deal": "tab:gray",
    "bad sale": "tab:orange",
    "bad buy": "tab:purple",
    "lost deal": "tab:red",
}


def plot_outcome_map(draws: pd.DataFrame, r: float, v: float, path: str) -> None:
    """Scatter of reports (r_hat, v_hat), colored by outcome, with region boundaries.

    With beta = 1/2 the price is (r_hat + v_hat) / 2, so:
    no deal when r_hat > v_hat; bad sale when r_hat + v_hat < 2r (p < r);
    bad buy when r_hat + v_hat > 2v (p > v).
    """
    fig, ax = plt.subplots(figsize=(7, 7))

    for outcome, color in OUTCOME_COLORS.items():
        subset = draws[draws["outcome"] == outcome]
        ax.scatter(subset["r_hat"], subset["v_hat"], s=8, color=color,
                   label=f"{outcome} ({len(subset)})")

    # Region boundaries (x = r_hat, y = v_hat).
    lo = min(draws["r_hat"].min(), draws["v_hat"].min()) - 1
    hi = max(draws["r_hat"].max(), draws["v_hat"].max()) + 1
    x = np.array([lo, hi])
    ax.plot(x, x, "k-", lw=1, label="r_hat = v_hat (below: no deal)")
    ax.plot(x, 2 * r - x, "k--", lw=1, label=f"r_hat + v_hat = {2 * r:g} (below: bad sale)")
    ax.plot(x, 2 * v - x, "k:", lw=1.5, label=f"r_hat + v_hat = {2 * v:g} (above: bad buy)")

    ax.plot(r, v, marker="*", color="black", markersize=18, linestyle="none",
            label=f"true (r, v) = ({r:g}, {v:g})")

    bad_share = (draws["outcome"] != "good deal").mean()
    ax.set_title(f"With two weak agents, {bad_share:.0%} of draws end in a lost deal or a bad price",
                 fontsize=11)
    ax.set_xlabel("Consumer agent's report r_hat ($)")
    ax.set_ylabel("Firm agent's report v_hat ($)")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_aspect("equal")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_surplus_split(summary: pd.DataFrame, path: str) -> None:
    """Stacked bars: mean CS + mean FS + mean lost value, one bar per scenario plus benchmark.

    Every bar totals v - r = $6, because of the identity CS + FS + lost = 6.
    summary must have columns scenario, mean_CS, mean_FS, mean_lost_value.
    """
    benchmark = pd.DataFrame([{"scenario": "benchmark", "mean_CS": 3.0,
                               "mean_FS": 3.0, "mean_lost_value": 0.0}])
    data = pd.concat([benchmark, summary], ignore_index=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(data))
    ax.bar(x, data["mean_CS"], label="Consumer surplus (CS)")
    ax.bar(x, data["mean_FS"], bottom=data["mean_CS"], label="Firm surplus (FS)")
    ax.bar(x, data["mean_lost_value"], bottom=data["mean_CS"] + data["mean_FS"],
           color=OUTCOME_COLORS["lost deal"], label="Lost value (lost deals)")

    ax.set_xticks(x)
    ax.set_xticklabels(data["scenario"])
    ax.set_xlabel("Scenario (consumer agent / firm agent)")
    ax.set_ylabel("Mean surplus per deal ($)")
    ax.set_ylim(0, 7)
    ax.set_title("Noisy agents waste surplus on lost deals, and the side with a weak agent keeps less",
                 fontsize=11)
    ax.legend(loc="upper right", fontsize=8, ncol=3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_consumer_gain_hist(strong_c: pd.DataFrame, weak_c: pd.DataFrame, path: str) -> None:
    """Histogram of CS on completed deals: strong vs weak consumer agent (firm agent strong).

    A vertical line at $0 marks the reservation price: bars to its left are
    bad sales (the consumer got less than their minimum).
    """
    cs_strong = strong_c.loc[strong_c["deal"], "CS"]
    cs_weak = weak_c.loc[weak_c["deal"], "CS"]
    bins = np.linspace(min(cs_strong.min(), cs_weak.min()),
                       max(cs_strong.max(), cs_weak.max()), 80)

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(cs_strong, bins=bins, alpha=0.6, label="strong consumer agent")
    ax.hist(cs_weak, bins=bins, alpha=0.6, label="weak consumer agent")
    ax.axvline(0, color="black", lw=1, label="CS = 0 (price = r)")

    below_weak = (cs_weak < 0).mean()
    below_strong = (cs_strong < 0).mean()
    ax.set_title(f"A weak consumer agent sells below the minimum in {below_weak:.1%} of deals, "
                 f"a strong one in {below_strong:.2%}", fontsize=11)
    ax.set_xlabel("Consumer surplus CS = p - r on completed deals ($)")
    ax.set_ylabel("Number of deals")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_noise_sweep(sweep: pd.DataFrame, r: float, v: float, path: str) -> None:
    """Lost deal rate vs sigma (same for both agents): simulated dots and formula line.

    The formula is Phi(-(v - r) / sqrt(2 sigma^2)). It is drawn on a fine
    grid so the line is smooth.
    """
    from bazaar.simulate import lost_deal_formula

    fine = np.linspace(sweep["sigma"].min(), sweep["sigma"].max(), 200)
    formula = [lost_deal_formula(r, v, s, s) for s in fine]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(fine, np.array(formula) * 100, "k-", lw=2, label="formula")
    ax.plot(sweep["sigma"], sweep["lost_rate"] * 100, "o", markersize=8, label="simulated")
    ax.set_title("Lost deals rise with agent noise, and the simulation matches the formula",
                 fontsize=11)
    ax.set_xlabel("Noise in each agent's report, sigma ($)")
    ax.set_ylabel("Lost deal rate (% of draws)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
