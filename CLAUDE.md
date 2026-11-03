# Data Bazaar v0

## What this is

A tiny simulation of one data sale. A consumer sells access to their
personal data to a firm. Each side is represented by an AI agent. The
agents send noisy reports to a "bazaar", which sets the price.

It asks two questions:

1. **Bargaining:** how does a gap in agent bargaining strength shift the
   split of surplus between consumer and firm?
2. **Learning:** how does a gap in how well an agent knows its principal's
   true number cause lost deals, bad sales and bad buys?

It is a toy that illustrates an argument. It proves nothing about the real
world, and the README must say so.

The owner is an econ student learning to code. Clarity beats cleverness.

## Coding rules

- Plain Python. Only numpy, pandas, matplotlib, scipy, pytest. No
  frameworks, no network access, no language models.
- Keep it simple and readable. Short functions, plain names, comments that
  explain the economics as well as the code.
- Model objects are small `@dataclass`es. A dataclass may have a short
  method (e.g. `Agent.report`). No inheritance, no class hierarchies.
- Every function has a short docstring saying what it computes.
- Randomness only through `numpy.random.default_rng(seed)`. Same seed,
  same output, always. Default seed: 0.
- Do not add features beyond this spec. If something is ambiguous, ask the
  human. If something seems missing, leave a `TODO` comment and ask.
- Do not change any number in the model spec without asking.

## File layout

```
bazaar/model.py      Consumer, Firm, Agent, Bazaar dataclasses
bazaar/simulate.py   run one deal; run a scenario N times -> DataFrame
bazaar/plots.py      the four figures
run_v0.py            runs Step A, Step B, sweep; saves results/*.csv,
                     figures/*.png; prints tables
tests/test_v0.py
README.md            one-page results write-up
requirements.txt
results/             CSV output
figures/             PNG output
```

## The model

One consumer, one firm, two AI agents, one bazaar.

### Consumer

- Private reservation price `r = 4`: the lowest price they would accept
  for their data.

### Firm

- Private value `v = 10`: the most it would pay for the data.

### Agent

One class, two instances: the consumer's agent and the firm's agent. Each
agent has a type, "strong" or "weak":

| type   | sigma (knowledge noise) | b (bargaining strength) |
|--------|-------------------------|-------------------------|
| strong | 0.5                     | 2                       |
| weak   | 3.0                     | 1                       |

- `sigma`: how well the agent knows its principal's true number. Smaller
  is better.
- `b`: bargaining strength.
- The agent reports its principal's number plus noise:
  - `r_hat = r + eps_c`, `eps_c ~ Normal(0, sigma_c^2)`
  - `v_hat = v + eps_f`, `eps_f ~ Normal(0, sigma_f^2)`

### Bazaar

- Receives ONLY `r_hat`, `v_hat`, `b_c`, `b_f`.
- `beta = b_c / (b_c + b_f)` (consumer's share of the reported surplus).
- If `r_hat <= v_hat`: price `p = r_hat + beta * (v_hat - r_hat)`.
  Otherwise: no deal.
- Returns only the price or "no deal". The consumer and firm never see the
  other side's true number or report.

### Scoring (always with TRUE r and v, never the reports)

- `CS = p - r`, `FS = v - p`. Both 0 if no deal.
- **lost deal:** `r <= v` but no deal. `lost value = v - r` (else 0).
- **bad sale:** deal and `p < r` (consumer sold below their minimum).
- **bad buy:** deal and `p > v` (firm paid above its maximum).
- Identity that must hold on every draw:
  `CS + FS + lost value = v - r = 6`.

### Results log

A separate record (the modeler's view) stores every number for each deal:
`scenario, draw, r, v, sigma_c, sigma_f, r_hat, v_hat, b_c, b_f, beta,
deal, p, CS, FS, lost value, outcome`. `draw` is the draw number within
the scenario (0, 1, 2, ...). This is for testing and analysis. It is
separate from what the bazaar returns.

Outcome labels: "good deal", "bad sale", "bad buy", "lost deal", plus
"no deal" for no deal when `r > v` (walking away was correct). "no deal"
cannot happen with r = 4, v = 10 but will in v1, so keep it.

Rates (lost deal, bad sale, bad buy) are shares of ALL draws, not of
completed deals.

## Benchmark

Full information, equal power: `p* = r + 0.5 (v - r) = 7`, `CS = 3`,
`FS = 3`, total 6, no lost deals.

## Experiments

Four scenarios: consumer agent {strong, weak} x firm agent {strong, weak}.
Naming: first word = consumer's agent, second = firm's agent.

### Step A: bargaining only

`sigma = 0` for all agents; `beta` from `b`. Deterministic. Must match
exactly:

| scenario      | beta | p | CS | FS |
|---------------|------|---|----|----|
| strong/strong | 1/2  | 7 | 3  | 3  |
| weak/weak     | 1/2  | 7 | 3  | 3  |
| weak/strong   | 1/3  | 6 | 2  | 4  |
| strong/weak   | 2/3  | 8 | 4  | 2  |

### Step B: learning only

`beta = 1/2` in every scenario; `sigma` from type. 100,000 draws per
scenario, fixed seed. Report per scenario: lost deal rate, bad sale rate,
bad buy rate, mean CS, mean FS, mean total, std of CS.

Reference (earlier 2M-draw run, approximate):

| scenario      | lost | bad sale | bad buy | CS   | FS   |
|---------------|------|----------|---------|------|------|
| strong/strong | 0.0% |          |         | 3.00 | 3.00 |
| weak/strong   | 2.4% | 2.4%     | 0.7%    | 2.84 | 3.01 |
| strong/weak   | 2.4% | 0.7%     | 2.4%    | 3.01 | 2.84 |
| weak/weak     | 7.8% | 7.2%     | 7.3%    | 2.77 | 2.76 |

Check: lost deal rate should match
`Phi(-(v - r) / sqrt(sigma_c^2 + sigma_f^2))`, Phi = standard normal CDF.

### Noise sweep

Both agents use the same sigma, from 0 to 6 in steps of 0.25, `beta = 1/2`.
Plot lost deal rate, simulated vs formula.

## Figures

PNG in `figures/`. Axes in dollars where they are money. Title = one
sentence stating the finding.

1. `outcome_map.png`: 2,000 draws of `(r_hat, v_hat)` in weak/weak (Step B
   settings), colored by outcome (good deal, bad sale, bad buy, lost deal).
   Draw region boundaries: `r_hat = v_hat` (above it on the `r_hat` side =
   no deal); `r_hat + v_hat = 8` (below = bad sale, since `p < r`);
   `r_hat + v_hat = 20` (above = bad buy, since `p > v`). Mark the true
   point `(r, v) = (4, 10)`.
2. `surplus_split.png`: one stacked bar per scenario (Step B) plus the
   benchmark: mean CS + mean FS + mean lost value, so every bar totals $6.
3. `consumer_gain_hist.png`: histogram of CS on completed deals, strong vs
   weak consumer agent (firm agent strong, Step B settings), vertical line
   at 0.
4. `noise_sweep.png`: lost deal rate vs sigma, simulated dots and formula
   line.

## Tests (tests/test_v0.py)

- Step A matches the table exactly.
- Per-draw identity `CS + FS + lost value = 6` holds on every draw.
- Step B lost deal rate is within 0.5 percentage points of the formula.

## README.md

One page. Title "Data Bazaar v0", one-line summary, the model in GitHub
math (`$$` blocks), Step A table, Step B table, the four figures with
one-line captions, "What I learned" (TODO placeholders for the human to
write), "Next steps", "Related work":

- Karten, Crow & Jin (2026), "Agent Bazaar: Enabling Economic Alignment
  in Multi-Agent Marketplaces", Princeton.
- Augusto Gonzalez-Bonorino, "Privacy Brokers" (Substack, Sept 2026).

Plus a short limitations note: this is a toy.

## How to work with the human

1. Build `bazaar/model.py` plus a function that runs ONE deal and prints
   every step (`r, r_hat, v, v_hat, beta, p, CS, FS, outcome`). Show 5
   example deals. Stop and wait.
2. Build the rest, run the tests, run `run_v0.py`, show the printed tables.
3. After each file, explain what it does in plain language. Define every
   term used.
4. Nothing beyond this spec. Ask when something is ambiguous.
