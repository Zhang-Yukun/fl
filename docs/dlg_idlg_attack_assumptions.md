# DLG / iDLG Attack Assumptions and Mapping to This Framework

## 1. What DLG and iDLG originally assume

### 1.1 DLG

DLG (`Deep Leakage from Gradients`, NeurIPS 2019) studies the setting where gradients are shared during distributed training, and an honest-but-curious server or peer tries to reconstruct the private training sample from the shared gradient. In the paper formulation, the attacker knows:

- the model architecture;
- the current model parameters at the attacked step;
- the exact gradient tensor observed from the victim;
- the loss function used to produce that gradient.

The attack itself optimizes dummy inputs (and, in DLG, dummy labels as well) so that the gradient induced by the dummy pair matches the intercepted gradient. In other words, the original attacked object is **the gradient at one attacked optimization state**, not an already-aggregated global model and not a multi-round checkpoint.

### 1.2 iDLG

iDLG (`Improved Deep Leakage from Gradients`, 2020) keeps the same basic attack setting as DLG, but shows that for the common cross-entropy classification setting the true label can be inferred directly from the last-layer gradient sign structure. After that, the optimization only needs to reconstruct the input, which makes the attack more stable than vanilla DLG.

So the main change from DLG to iDLG is not the transport object. The attacked object is still the **shared gradient at a known model state**. The difference is that iDLG avoids jointly optimizing the label in the common classification case.

## 2. In the original papers, what is uploaded and aggregated?

From the perspective of communication semantics, the original DLG / iDLG assumption is:

- the victim shares **gradients**;
- the attacker observes those gradients before any privacy defense hides them;
- reconstruction is performed against those observed gradients.

The papers discuss distributed / federated style training motivation, but the attack target itself is the gradient tensor. They do **not** require the victim to upload the full model parameters. They also do not attack a final global model checkpoint.

This matters when mapping to FL:

- if a client uploads a **single-step SGD update**, then the uploaded model update is proportional to the gradient and is very close to the original DLG/iDLG assumption;
- if a client uploads a **multi-step or multi-epoch local model update**, then the uploaded object is no longer the same thing as a single-batch gradient, and the attack becomes a payload-reconstruction attack rather than the exact original DLG setting.

Therefore, “upload model”, “upload gradient”, and “upload model update” are not interchangeable:

- **full model parameters**: usually not the original DLG/iDLG assumption;
- **single-step model update**: often equivalent to gradient up to optimizer scaling;
- **multi-step local update**: only an approximation / extension of the original gradient inversion setting.

## 3. How this framework currently implements upload, aggregation, and download

The core single-node federated loop is in `fedlab/federated/algorithms.py`.

### 3.1 Round start and download

At the start of each round:

1. the server clones the current global model as `round_base_state`;
2. the server optionally prepares method-specific round context;
3. each client receives that round base state and then runs `method.prepare_client_state(...)`.

Relevant code:

- `fedlab/federated/algorithms.py`: `run_federated`
- `fedlab/federated/server.py`: `FederatedServer.build_round_context`
- `fedlab/federated/client.py`: `FederatedClient.train`
- `fedlab/federated/methods/base.py`: `prepare_client_state`

Semantically, the framework distinguishes:

- `global_state`: the server round-start model;
- `download_state`: what is counted as the transmitted download payload;
- `received_global_state`: what the client really loads for local training.

For dense FedAvg they are usually the same. For compressed download methods they can differ.

### 3.2 Local training

Each client builds a fresh local model, loads `received_global_state`, and trains locally using:

- `federated.local_steps` if configured; otherwise
- `federated.local_epochs`.

So the framework is already more general than the original DLG/iDLG assumption. If `local_steps=1` and the local optimizer is SGD, the upload can closely match one-step gradient sharing. If the client trains one epoch or more, the upload becomes a local model update accumulated over many mini-batches.

Relevant code:

- `fedlab/federated/client.py`: `FederatedClient.train`

### 3.3 Upload semantics

The framework does **not** use one universal upload object. It is method-dependent.

#### Dense methods

For dense FedAvg-style methods, the client uploads:

- `aggregation_state = local_state - global_state`

This is a **model update**, not a full model snapshot.

Relevant code:

- `fedlab/federated/methods/dense.py`

#### Sparse methods

For sparse methods, the client typically uploads:

- sparse trainable-layer update;
- plus dense buffer update if needed.

So the logical upload object is still a local update, but represented partly as sparse payload and partly as dense residual/buffer state.

Relevant code:

- `fedlab/federated/methods/sparse.py`

#### Quantized methods

For quantized methods, the client uploads:

- quantized version of the model update, usually derived from `local_state - global_state` or `local_state - received_global_state`.

The server dequantizes before aggregation.

Relevant code:

- `fedlab/federated/methods/quantized.py`

#### EGA

For EGA, the client uploads:

- encoded trainable update relative to `received_global_state`;
- optional dense buffer update separately.

So EGA is not uploading a raw gradient and not uploading a full model. It uploads an **encoded local model update**.

Relevant code:

- `fedlab/federated/methods/encoded.py`

### 3.4 Aggregation semantics

On the server side:

- dense methods call `aggregate_dense(...)`;
- sparse methods call `aggregate_sparse(...)`.

Both take the per-client **pre-aggregation upload payloads** and construct the next round global model.

Important point: the server aggregates **client uploads**, not client full local models.

Relevant code:

- `fedlab/federated/server.py`: `aggregate_dense`, `aggregate_sparse`
- `fedlab/federated/methods/dense.py`
- `fedlab/federated/methods/sparse.py`
- `fedlab/federated/methods/quantized.py`
- `fedlab/federated/methods/encoded.py`

### 3.5 Downlink after aggregation

After aggregation:

- the server updates `server.global_state`;
- the next round again starts from this `server.global_state`;
- each client then receives either the dense state or a method-specific compressed download derived from it.

So the framework is a standard iterative FL loop:

1. download current global state;
2. local train;
3. upload local update payload;
4. server aggregate into new global state;
5. repeat.

## 4. How attack targets are implemented in this framework

The framework supports two attack target types:

- `gradient`
- `update_payload`

Configured in:

- `attack.target_type`

Relevant code:

- `fedlab/federated/algorithms.py`: `_attack_target_type`, `_build_attack_round_task`
- `fedlab/security/attacks.py`

### 4.1 `gradient` mode

In `gradient` mode, the framework does **not** attack the uploaded payload. Instead, it explicitly asks the client to compute a selected batch gradient on `round_base_state`:

- `client.gradient_sample(round_base_state, ...)`

This is the closest path to the original DLG / iDLG paper assumption.

However, it is still only exactly equivalent to the original setting when the experimental setup also matches the paper-side assumptions, especially:

- attacked object is a per-step gradient;
- optimizer behavior matches the inversion objective;
- the intercepted target is not already transformed by extra local training steps.

### 4.2 `update_payload` mode

In `update_payload` mode, the framework attacks the **actual transmitted client payload**:

- dense method: dense model update;
- sparse method: sparse update merged with dense buffer update;
- quantized method: dequantized view of the transmitted quantized update;
- EGA: decoded attack view of the encoded transmitted update, plus any dense buffer update.

This is intentionally closer to the real FL communication path than pure gradient mode.

One implementation detail matters a lot for interpretation:

- the attacked `target` is the client payload produced after the configured local training process;
- but the stored `real_x / real_y` used for reconstruction reference come from `client.sample_batch(...)`, i.e. one selected local batch.

So when a client trains for a full epoch or for multiple local steps, the uploaded payload usually reflects **more than one batch**, while the reference sample shown in attack artifacts is only **one selected batch**. In that case, `update_payload` mode should be interpreted as a practical payload attack probe, not as an exact single-batch DLG/iDLG reproduction.

Relevant code:

- `fedlab/federated/algorithms.py`: `_extract_attack_payload`
- `fedlab/federated/methods/dense.py`: `extract_attack_payload`
- `fedlab/federated/methods/sparse.py`: `extract_attack_payload`
- `fedlab/federated/methods/quantized.py`: `extract_attack_payload`
- `fedlab/federated/methods/encoded.py`: `extract_attack_payload`

### 4.3 Is the server attacking pre-aggregation or post-aggregation objects?

The current implementation attacks **pre-aggregation client payloads**.

More precisely:

- the round first collects each client `result`;
- the server aggregates these `result`s to update the global model;
- then the attack task is built from the stored per-client `result` objects;
- in `update_payload` mode, the attacked target is extracted from each client’s own transmitted payload, not from the already-averaged global update.

So from the server’s viewpoint, the attack is conceptually “honest-but-curious server sees each client upload and attacks that upload”.

## 5. Oracle evaluation vs protocol evaluation in this framework

The framework also keeps a separate oracle evaluation path:

- `evaluation_state` on the client is always `local_state - global_state`;
- server-side oracle evaluation builds
  `oracle_global_state = round_base_state + average(evaluation_state_i)`.

This is only for evaluation / diagnosis. It is **not** the same object as the communication payload when compression, sparsification, quantization, or encoding is enabled.

That distinction is important:

- **protocol path** = what the FL system actually transmits and aggregates;
- **oracle path** = dense reference path used to judge what accuracy would look like without payload distortion.

Relevant code:

- `fedlab/federated/client.py`: `_evaluation_result_kwargs`
- `fedlab/federated/server.py`: `_update_oracle_evaluation_state`

## 6. Mapping the framework to DLG / iDLG assumptions

### 6.1 When the framework matches the original papers most closely

The closest configuration to original DLG / iDLG is:

- `attack.target_type=gradient`;
- `federated.local_steps=1`;
- local optimizer set to SGD;
- no extra privacy transform before the attacker sees the target;
- attacker knows the current model state (`round_base_state`), architecture, and loss.

That setup is the cleanest “gradient inversion” interpretation.

### 6.2 When the framework departs from the original papers

The framework departs from the original DLG / iDLG formulation when:

- clients train for multiple steps or full epochs before upload;
- uploads are sparse / quantized / encoded instead of raw gradients;
- the attack target is `update_payload` instead of `gradient`;
- the payload corresponds to model delta rather than exact instantaneous gradient.

In these cases the framework is still studying a meaningful reconstruction attack, but it is no longer the exact original DLG/iDLG communication assumption. It becomes a broader “attack on FL communication payloads” setting.

## 7. Practical takeaway

For this repository, the cleanest interpretation is:

- **DLG / iDLG literature baseline**: attacks exact shared gradients at a known model state;
- **this framework, `gradient` mode**: approximates / reproduces that setting;
- **this framework, `update_payload` mode**: attacks the real transmitted FL payload, which is usually a local model update or its compressed form;
- **server aggregation**: always works on per-client upload payloads, then constructs the next global model;
- **server oracle evaluation**: separately reconstructs a dense reference model for evaluation only.

So if the goal is “strictly align with original DLG/iDLG”, use `gradient` mode and one-step local training. If the goal is “evaluate privacy risk under the actual communication protocol of this framework”, use `update_payload` mode.

## 8. References

- DLG: `Deep Leakage from Gradients`, NeurIPS 2019
- iDLG: `iDLG: Improved Deep Leakage from Gradients`, 2020
- DLG arXiv: `https://arxiv.org/abs/1906.08935`
- iDLG arXiv: `https://arxiv.org/abs/2001.02610`
