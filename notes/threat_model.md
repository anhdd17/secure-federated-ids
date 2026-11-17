# Threat model (draft, 2026-11-17)

Setting: cross-silo federated intrusion detection. n = 10 clients (20 in the generalisation grid), each an
organisation or IoT gateway with its own labelled traffic; full participation every round; one server.

| | |
|---|---|
| **Goal** | *Untargeted*: lower the global model's macro-F1 on the held-out test set (label flip, sign flip, LIE, Min-Max). *Targeted (backdoor)*: attack traffic carrying a trigger pattern is classified as **Benign**, while accuracy on clean traffic stays high so the backdoor is not noticed (model replacement, constrain-and-scale, on-off). |
| **Knowledge** | The attacker knows the model architecture, the aggregation rule and its hyperparameters, and receives the global model every round like any client. For LIE and Min-Max we assume the worst case: the attacker also knows the benign updates of the current round (or their mean and standard deviation). |
| **Capability** | Controls f of n clients (f/n ∈ {10, 20, 30, 40}%), which may collude. Can change its own data, labels, local training and the submitted update arbitrarily, including its self-reported sample count. Cannot read or modify other clients' data or updates in transit, and cannot touch the server. |
| **Defender** | Honest server that sees every individual update (no secure aggregation). For FLTrust and our method it holds a small clean **root dataset** (296 samples, 37 per class) drawn from the global distribution. The server does not know f or which clients are malicious. |
| **Out of scope** | Malicious or curious server; eavesdropping or tampering on the channel; Sybil clients created during training; inference-time evasion of the deployed IDS; privacy attacks on the global model other than the DP experiments. |

## Assumptions to state as limitations

- The root dataset is a strong assumption (FLTrust). It is plausible for a SOC that owns labelled traffic,
  but its distribution matches the global one here, which favours FLTrust-style defences.
- Full participation simplifies reputation tracking; partial participation would leave reputations stale.
- The worst-case knowledge for LIE / Min-Max is deliberately pessimistic for the defender.
