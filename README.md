# Data Bazaar v0

A toy simulation of one data sale in which AI agents bargain for a consumer and a firm. It shows how gaps in agent strength and agent knowledge shift who gets the surplus, and how often deals fail.

> **Limitations.** This is a toy that illustrates an argument. It has one consumer, one firm, fixed numbers and a simple price rule. It proves nothing about real data markets.

## The model

A consumer has a private **reservation price** $r = 4$ (the lowest price they would accept for their data). A firm has a private **value** $v = 10$ (the most it would pay). Each side has an AI agent of type *strong* or *weak*:

| agent type | $\sigma$ (noise in its report) | $b$ (bargaining strength) |
|---|---|---|
| strong | 0.5 | 2 |
| weak | 3.0 | 1 |

Each agent reports its principal's number with noise:

$$\hat r = r + \varepsilon_c, \qquad \hat v = v + \varepsilon_f, \qquad \varepsilon \sim N(0, \sigma^2)$$

The bazaar sees only $\hat r$, $\hat v$, $b_c$, $b_f$. It sets the consumer's share $\beta$ and the price:

$$\beta = \frac{b_c}{b_c + b_f}, \qquad p = \begin{cases} \hat r + \beta(\hat v - \hat r) & \text{if } \hat r \le \hat v \\ \text{no deal} & \text{otherwise} \end{cases}$$

Outcomes are scored with the **true** $r$ and $v$:

$$CS = p - r, \qquad FS = v - p, \qquad CS + FS + \text{lost value} = v - r = 6$$

A **lost deal** is no deal even though $r \le v$ (lost value $= v - r$). A **bad sale** is a deal with $p < r$. A **bad buy** is a deal with $p > v$.

**Benchmark** (full information, equal power): $p^* = 7$, $CS = 3$, $FS = 3$, no lost deals.

## Step A: bargaining only ($\sigma = 0$)

| consumer / firm agent | $\beta$ | $p$ | CS | FS |
|---|---|---|---|---|
| strong / strong | 1/2 | 7.00 | 3.00 | 3.00 |
| weak / strong | 1/3 | 6.00 | 2.00 | 4.00 |
| strong / weak | 2/3 | 8.00 | 4.00 | 2.00 |
| weak / weak | 1/2 | 7.00 | 3.00 | 3.00 |

## Step B: learning only ($\beta = 1/2$, 100,000 draws per scenario, seed 0)

Rates are shares of all draws. The formula column is $\Phi\left(-\frac{v - r}{\sqrt{\sigma_c^2 + \sigma_f^2}}\right)$.

| consumer / firm agent | lost deals | formula | bad sales | bad buys | mean CS | mean FS | mean total | std CS |
|---|---|---|---|---|---|---|---|---|
| strong / strong | 0.00% | 0.00% | 0.00% | 0.00% | 3.00 | 3.00 | 6.00 | 0.35 |
| weak / strong | 2.49% | 2.43% | 2.41% | 0.69% | 2.84 | 3.01 | 5.85 | 1.49 |
| strong / weak | 2.42% | 2.43% | 0.76% | 2.37% | 3.01 | 2.85 | 5.85 | 1.50 |
| weak / weak | 7.99% | 7.86% | 7.19% | 7.27% | 2.76 | 2.76 | 5.52 | 2.19 |

## Figures

![Outcome map](figures/outcome_map.png)
*2,000 weak/weak draws of the two reports, colored by outcome. The lines mark where deals fail or land at a bad price.*

![Surplus split](figures/surplus_split.png)
*Mean CS, FS and lost value per scenario. Every bar totals $6.*

![Consumer gain histogram](figures/consumer_gain_hist.png)
*Consumer surplus on completed deals with a strong firm agent: a weak consumer agent gives a much wider spread, including losses.*

![Noise sweep](figures/noise_sweep.png)
*Lost deal rate as both agents get noisier: the simulation tracks the formula.*

## What I learned

- TODO
- TODO
- TODO

## Next steps

- TODO: combine the bargaining gap (Step A) and the knowledge gap (Step B) in one run.
- TODO: v1 with cases where $r > v$, so a correct "no deal" is possible.

## How to run

```bash
pip install -r requirements.txt
python3 run_v0.py
python3 -m pytest
```

## Related work

- Karten, Crow & Jin (2026), "Agent Bazaar: Enabling Economic Alignment in Multi-Agent Marketplaces", Princeton.
- Augusto Gonzalez-Bonorino, "Privacy Brokers" (Substack, Sept 2026).
