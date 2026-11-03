"""The building blocks of Data Bazaar v0: consumer, firm, agent, bazaar.

Each piece is a small dataclass (a class that mostly just holds numbers).
The rules for how they interact are in CLAUDE.md, section "The model".
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Consumer:
    """The person selling their data.

    r is their reservation price: the lowest price they would accept.
    Only the consumer (and their agent, imperfectly) knows it.
    """

    r: float = 4.0


@dataclass
class Firm:
    """The company buying the data.

    v is the firm's value: the most it would pay. Only the firm (and its
    agent, imperfectly) knows it.
    """

    v: float = 10.0


# The two agent types. sigma = how noisy the agent's knowledge of its
# principal's number is (smaller = knows it better). b = bargaining strength.
AGENT_TYPES = {
    "strong": {"sigma": 0.5, "b": 2.0},
    "weak": {"sigma": 3.0, "b": 1.0},
}


@dataclass
class Agent:
    """An AI agent that represents one side (consumer or firm).

    kind is "strong" or "weak". sigma is the standard deviation of the
    noise in its report. b is its bargaining strength.
    """

    kind: str
    sigma: float
    b: float

    def report(self, true_value: float, rng: np.random.Generator) -> float:
        """Return the agent's report: the true number plus Normal(0, sigma^2) noise.

        For the consumer's agent, true_value is r and the report is r_hat.
        For the firm's agent, true_value is v and the report is v_hat.
        """
        noise = rng.normal(0.0, self.sigma)
        return true_value + noise


def make_agent(kind: str, sigma: float | None = None, b: float | None = None) -> Agent:
    """Build an Agent of the given kind, using the numbers in AGENT_TYPES.

    sigma or b can be overridden. Step A sets sigma = 0 (perfect knowledge);
    Step B and the sweep fix b so that beta = 1/2.
    """
    defaults = AGENT_TYPES[kind]
    return Agent(
        kind=kind,
        sigma=defaults["sigma"] if sigma is None else sigma,
        b=defaults["b"] if b is None else b,
    )


@dataclass
class Bazaar:
    """The marketplace that sets the price.

    It has no data of its own. It sees ONLY the two reports and the two
    bargaining strengths, never the true r or v.
    """

    def beta(self, b_c: float, b_f: float) -> float:
        """Consumer's share of the reported surplus: beta = b_c / (b_c + b_f)."""
        return b_c / (b_c + b_f)

    def set_price(self, r_hat: float, v_hat: float, b_c: float, b_f: float) -> float | None:
        """Return the price, or None for "no deal".

        If the consumer's reported minimum is at or below the firm's
        reported maximum (r_hat <= v_hat), the price splits the reported
        gap: p = r_hat + beta * (v_hat - r_hat). Otherwise there is no deal.
        """
        if r_hat <= v_hat:
            return r_hat + self.beta(b_c, b_f) * (v_hat - r_hat)
        return None


def score(r: float, v: float, p: float | None) -> dict:
    """Score a deal with the TRUE r and v (never the reports).

    Returns CS (consumer surplus = p - r), FS (firm surplus = v - p),
    lost value (v - r if a worthwhile deal did not happen) and an outcome
    label: "good deal", "bad sale" (p < r), "bad buy" (p > v) or
    "lost deal" (r <= v but no deal).
    """
    if p is None:
        cs, fs = 0.0, 0.0
        if r <= v:
            lost_value = v - r
            outcome = "lost deal"
        else:
            # Can't happen with r = 4, v = 10: no deal was the right call.
            lost_value = 0.0
            outcome = "no deal"
    else:
        cs = p - r
        fs = v - p
        lost_value = 0.0
        if p < r:
            outcome = "bad sale"
        elif p > v:
            outcome = "bad buy"
        else:
            outcome = "good deal"
    return {"CS": cs, "FS": fs, "lost_value": lost_value, "outcome": outcome}
