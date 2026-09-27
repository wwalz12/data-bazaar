# Data bargaining ABM (Stage 1)

## What this is

A small agent-based simulation of a market where consumers sell access to
their personal data to firms, with a software agent bargaining on each
consumer's behalf. It exists to answer one question:

> In a market for personal data with bargaining agents, does an agent
> quality gap (richer consumers get better agents) make the distributional
> problem worse?

The model is a thought experiment from a Substack post ("Privacy Brokers")
turned into numbers. It is a toy. It illustrates an argument. It does not
prove anything about the real world, and the README must say so.

## Hard constraints

- Pure Python. Only numpy, pandas, matplotlib, pytest. No Mesa, no NetLogo,
  no scikit-learn, no torch.
- No language models. No API calls. No network access of any kind.
- Every decision in the model is a deterministic formula or an if-statement.
  Randomness enters only through seeded initial draws (numpy default_rng).
  Same seed, same output, always.
- No classes except simple @dataclass containers. Logic lives in plain
  functions. model.py stays under 200 lines.
- Every function has a one-paragraph docstring saying what it computes and
  which section of this file specifies it. Type hints everywhere.
- Do not add features beyond this spec. If something seems missing, leave a
  TODO comment and stop. The human decides.

## File layout

```
model.py         consumers, firm, bargain(), run_once()
experiments.py   parameter grid x seeds -> results/runs.csv
plots.py         three figures -> figures/*.png
tests/test_model.py
README.md        what it is, how to run, what the plots show, limitations
CLAUDE.md        this file
results/         gitignored except .gitkeep
figures/         committed
```

## Model spec

All money is in dollars. "Three months of browsing and purchase history" is
the unit being sold. One consumer sells at most once per run.

### Consumers (N = 1000)

Each consumer i gets, from one seeded rng:

- `wealth_pct`: uniform(0, 1). Wealth percentile. 0 = poorest.
- `privacy_taste`: uniform(0.2, 1.0). How much they dislike sharing.
- `min_price` (reservation price): the lowest price at which they would
  willingly sell.
  `min_price = 2 + 12 * privacy_taste * wealth_pct`
  So a poor, low-privacy consumer is near $2; a rich, high-privacy one
  near $14. Poorer consumers have lower minimums because they need the
  money more.
- `spending_pct = 0.7 * wealth_pct + 0.3 * uniform(0, 1)`, clipped to
  [0, 1]. How much they buy, which drives what their data is worth.
- `service_value`: what access to the firm's service is worth to them.
  `service_value = 8` for everyone in v1. (Later variant: proportional to
  1 - wealth_pct, since a fixed service matters more to a poorer person.)
- `agent_strength`: bargaining strength of their personal agent, one of
  two tiers. `GOOD = 0.7`, `WEAK = 0.3`. Assignment rule is a treatment,
  see Treatments.

### Firm

One representative firm in v1 (multiple firms is a v2 idea, do not build).

- `firm_value_i = 10 + 20 * spending_pct_i`. What three months of consumer
  i's data is worth to the firm. Range $10 to $30.
- `broker_price`: what the firm would pay a data broker for roughly
  equivalent data instead. This is the firm's outside option. Treatment
  parameter.
- The firm will never pay more than `ceiling_i = min(firm_value_i,
  broker_price)`.

### The bargain (one consumer, one firm)

```
effective_min = min_price
if exclusion_on:
    effective_min = max(0, min_price - service_value)
    # refusing means losing the service, so the consumer will accept
    # less than they otherwise would

ceiling = min(firm_value, broker_price)

if ceiling <= effective_min:
    no deal. price = None.
else:
    price = effective_min + agent_strength * (ceiling - effective_min)
    deal.
```

`agent_strength` is the share of the available surplus the consumer's
agent captures. 1.0 would mean the consumer gets everything up to the
firm's ceiling; 0.0 means the consumer gets exactly their effective
minimum.

### Accounting per deal

- `true_surplus = firm_value - min_price` (uses the true minimum, not the
  effective one, because that is the value actually created).
- `consumer_gain = price - min_price`. Can be negative under exclusion:
  the consumer sold below their real minimum because refusing cost them
  the service. Flag these rows as `coerced = True`.
- `firm_gain = firm_value - price`.

### Baseline: today

A run with no agents at all: every consumer for whom `firm_value > 0`
"sells" at `price = 0` (they accept the terms of service). Consumer gain
is `-min_price` for everyone. This is the comparison point.

### Treatments (the parameter grid)

- `broker_price` in {6, 12, 25}. Low means the firm barely needs you.
- `exclusion_on` in {False, True}.
- `agent_assignment` in {"equal", "wealth_correlated"}:
  - equal: every consumer gets GOOD with probability 0.5.
  - wealth_correlated: consumer i gets GOOD with probability
    `0.2 + 0.6 * wealth_pct_i`. Rich people can afford better agents.
- Plus the baseline run.

That is 3 x 2 x 2 = 12 settings plus baseline. Run each for seeds
0..199. `run_once(params, seed)` returns one DataFrame with one row per
consumer. `experiments.py` concatenates everything into
`results/runs.csv` with columns for every parameter, the seed, and every
per-consumer quantity above.

### Outcome metrics (computed per run, then averaged across seeds)

- `consumer_share`: sum(consumer_gain over deals) / sum(true_surplus over
  deals). Fraction of created value the consumers keep.
- `sell_rate_by_quintile`: fraction of consumers who sold, by wealth
  quintile (0-20%, 20-40%, ...).
- `consumer_share_by_quintile`: same as consumer_share but within quintile.
- `coerced_rate_by_quintile`: fraction of sales that were coerced, by
  quintile. Only nonzero when exclusion_on.

### The three figures

1. `fig1_share_vs_broker.png`: consumer_share (y) against broker_price (x),
   one line for exclusion off, one for exclusion on. Equal agents.
2. `fig2_sell_rate_by_quintile.png`: grouped bars, sell rate by wealth
   quintile, exclusion off vs on. broker_price = 12, equal agents.
3. `fig3_share_by_quintile_agents.png`: consumer_share by wealth quintile,
   equal agents vs wealth_correlated agents. broker_price = 12, exclusion
   off. This is the headline figure for the research question.

Every figure: matplotlib, one panel, axis labels with units, a one-line
title, no seaborn, no styling beyond defaults. Error bars or shaded bands
from the across-seed standard deviation.

## Tests (tests/test_model.py)

- Same seed twice gives identical DataFrames.
- When broker_price is below every consumer's min_price and exclusion is
  off, there are zero deals.
- Every deal price lies in [effective_min, ceiling].
- With exclusion on, effective_min is never above min_price.
- With agent_strength = 1.0, price equals ceiling for every deal.
- Baseline run has consumer_share exactly 0 (or -inf guarded: report
  consumer_gain sum instead).

## Working style

- Build in this order: model.py, tests, experiments.py, plots.py, README.
  Run tests after model.py and stop if any fail.
- After building, run `python experiments.py --quick` (3 seeds) and
  `python plots.py`, then report what the three figures show in plain
  words. Do not interpret beyond what is on the axes.
- Ask before changing any number in the Model spec. Those are the human's
  decisions.
- Keep comments rare. The docstrings plus this file are the documentation.
