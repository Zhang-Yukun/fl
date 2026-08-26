# DLG / iDLG Attack Assumptions and Mapping to This Framework

## 1. Original paper assumptions

### 1.1 DLG

DLG studies the setting where an attacker observes the victim's per-step gradient at a known model state and optimizes dummy inputs and dummy labels so that the induced gradient matches that observed gradient.

### 1.2 iDLG

iDLG keeps the same basic assumption about the attacked object, but for common cross-entropy classification tasks it infers the label directly from the last-layer gradient sign structure, then only optimizes the input.

So in the original papers, the attacked object is still a gradient, not a final global checkpoint and not a multi-step local update.

## 2. What this framework does now

This framework no longer exposes a separate `gradient` attack mode. The current code only supports `attack.target_type=update_payload`.

That means the attacker in this repository observes and attacks the server-visible client upload payload produced after local training:

- dense methods upload dense model updates
- sparse methods upload sparse updates plus any required dense buffer state
- quantized methods upload quantized updates that are dequantized into an attack view on the server side
- EGA uploads encoded updates and attacks the decoded server-visible view

Therefore the current framework studies payload reconstruction under the actual FL communication protocol, not exact gradient inversion in the narrow original-paper sense.

## 3. Relationship to original DLG and iDLG

The mapping is:

- original DLG / iDLG: attack a per-step gradient at a known model state
- current framework: attack the transmitted local update payload at a known round-start model state

When a client trains only one sample for one step with SGD, the uploaded update can be close to a scaled gradient. But once the client trains for one epoch or over multiple mini-batches, the payload is no longer identical to a single gradient. It becomes a broader FL payload attack problem.

## 4. Current DLG and iDLG behavior in this repository

For every attack task, the optimizer compares a dummy batch induced one-step local update against the intercepted payload.

- `DLG`: optimize `dummy_x` and `dummy_y`
- `iDLG`:
  - classification tasks infer one pseudo-label using the original iDLG last-layer sign rule, then optimize only `dummy_x`
  - time-series / regression tasks fall back to the same target optimization form as DLG

The attack optimizer itself currently only supports `adam`.

## 5. Evaluation semantics in the current framework

Because the attacked object is a local update payload rather than an exact single gradient, the framework evaluates attacks against a reference training set instead of relying only on one exact target sample.

The default primary metric is `budget_recovered_fraction`, derived by:

1. reconstructing up to `max_samples` samples for one attack record
2. matching them one-to-one against the attacked client's reference training set using `recovery_match_metric`
3. judging each match with `recovery_success_metric` and its threshold
4. reporting the recovered fraction within the reconstruction budget and within the full reference set

Important derived metrics include:

- `budget_recovered_fraction`
- `coverage_recovered_fraction`
- `nearest_client_train_mse`
- `exact_target_mse`
- `objective_mse`

## 6. Practical takeaway

If the question is whether this repository still implements the exact original gradient-sharing assumption, the answer is no: it now evaluates DLG / iDLG style reconstruction against server-visible FL update payloads only.

If the question is whether the code still preserves the algorithmic difference between DLG and iDLG, the answer is yes: classification iDLG uses label inference, while time-series / regression iDLG degrades to the DLG-style optimization path because the original label inference assumption does not hold.
