"""sembaker.optimizer: the external compile-vs-native optimizer.

Backend-agnostic decision layer. A backend rewriter (e.g. sembaker.backends.pz)
walks the user's pipeline, calls decide() per semantic operator, and swaps
ops to their compile-execute counterparts when beneficial.
"""

from sembaker.optimizer.cost_model import PROFILES, OpCostProfile, crossover
from sembaker.optimizer.decision import Decision, decide

__all__ = ["decide", "Decision", "crossover", "PROFILES", "OpCostProfile"]
