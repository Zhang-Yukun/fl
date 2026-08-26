# Communication Compression and Privacy Notes

## Literature Scope

The retained rare-earth communication-compression comparison set now focuses on the active suite algorithms. Among sparse baselines, `topk` remains the primary retained compression baseline.


## Recorded Metrics

Each round records client names, payload type, compressor, upload/download bytes, upload/download parameter counts, dense FedAvg reference upload size, upload compression ratio, total communication ratio, runtime, validation metrics, and attack results. `attack_results.json` stores DLG/iDLG reconstruction metrics and success flags.

## Extra Top-k Compression Baseline

The retained Top-k baseline is `sparse_fedavg`. It implements the core Top-k update upload idea: each client sends the largest-magnitude `topk_fraction` coordinates plus their indices, and the server reconstructs sparse updates before FedAvg-weighted aggregation.

Run the rawdata2 full PatchTST Top-k experiment from `src`:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/topk.yaml
```

The default attack configuration now evaluates DLG and iDLG every federated round with 300 optimization steps on `cuda:0`.
