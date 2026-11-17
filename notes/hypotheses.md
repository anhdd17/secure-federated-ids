# Research questions and hypotheses (draft, 2026-11-17)

**Gap.** Robust aggregation rules are evaluated mostly on image benchmarks with mild heterogeneity.
Under strong label skew, benign clients look anomalous: Phase 1 shows FedAvg alone drops from 0.748
(IID) to 0.520 macro-F1 at Dirichlet α = 0.1 with no attacker at all. Per-round trust scores such as
FLTrust's must then separate "different because of local data" from "different because malicious",
and a single round carries little evidence for either.

**RQ.** Does accumulating client trust over rounds make server-side defences for federated IDS more
accurate under non-IID data, without opening a new weakness to adaptive attackers?

| | Hypothesis | Measured by | Supported if |
|---|---|---|---|
| **H1** | At α = 0.1, FLTrust's per-round trust of benign clients fluctuates strongly, penalising them; EMA smoothing raises macro-F1 without raising ASR. | Std of benign trust over rounds; macro-F1 and ASR, ours vs FLTrust, paired by seed | Macro-F1 higher (95% CI of the paired difference > 0), ASR not higher |
| **H2** | Against a persistent attacker, accumulated trust drives malicious weights to ~0 faster than FLTrust, lowering ASR. | Rounds until malicious weight < 0.01; ASR | Fewer rounds and lower ASR (paired) |
| **H3** | Symmetric EMA is exploitable by an on-off attacker; an asymmetric EMA (fast decrease, slow increase) lowers ASR compared with the symmetric one. | ASR under on-off backdoor, symmetric vs asymmetric | Lower ASR (paired), no loss of clean macro-F1 > 0.01 |

Each hypothesis is tested with 5 seeds; a seed fixes the partition, the choice of malicious clients,
model initialisation and batch order, so comparisons between methods are paired.
