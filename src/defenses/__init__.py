"""Aggregation rules. Common interface:

    aggregate(updates: Tensor[n, d], ctx: dict) -> (Tensor[d], info: dict)

`ctx` carries what a rule may legitimately use (client sample counts, round, root update, ...).
`info["weights"]` (one weight per client, summing to 1 when defined) is logged every round.
"""
from src.defenses.fedavg import fedavg

AGGREGATORS = {"fedavg": fedavg}
