"""Combine several detector models into one signal with a 2-of-3 quorum.

Following Singh et al. (2026, arXiv:2609.31749): a learned fusion only beats the best single detector when
refitted on labelled in-domain data (none exists here), and detectors sharing an architecture fail together.
So each detector runs models of different architectures and a signal needs QUORUM agreeing votes.
A model without input (NaN probability) does not vote.
"""

import math

# A model votes "synthetic" when its AI probability reaches this value, else "real".
AI_PROBABILITY_THRESHOLD = 0.5
# Agreeing votes needed for "synthetic" or "real"; otherwise the detector reports "disagree".
QUORUM = 2


def quorum(probabilities: list[float]) -> str:
    votes = [p >= AI_PROBABILITY_THRESHOLD for p in probabilities if not math.isnan(p)]
    if sum(votes) >= QUORUM:
        return "synthetic"
    if len(votes) - sum(votes) >= QUORUM:
        return "real"
    return "disagree"
