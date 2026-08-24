# rawdata2 PatchTST Experiment Results

Run date: 2026-07-31

## Scope

This note now keeps only the experiment entry points that still have retained configs in the repository:

- `centralized`
- `fedavg`
- `topk`
- `qsgd`
- `randomk`
- `sign`
- `adaptive_clipped_rdp_fedavg`
- `secure_quantized_fedavg`
- `ega`

Historical notes for removed config presets were intentionally dropped from this file to avoid pointing to deleted configuration files.

## Example Commands

```bash
bash scripts/run_rawdata2_centralized.sh
bash scripts/run_rawdata2_fedavg.sh
bash scripts/run_rawdata2_fedlab_topk.sh
bash scripts/run_rawdata2_qsgd.sh
bash scripts/run_rawdata2_randomk.sh
bash scripts/run_rawdata2_sign.sh
bash scripts/run_rawdata2_adaptive_clipped_rdp_fedavg_deterministic.sh
bash scripts/run_rawdata2_secure_quantized_fedavg.sh
bash scripts/run_rawdata2_ega.sh
```

## Historical Result Snapshots

### Centralized

Output directory: `outputs/centralized/`

- Model parameters: 6,269,448
- Model bytes: 25,077,796
- Epochs completed: 168
- Early stop: epoch 167
- Runtime: 1174.0644 seconds
- Best validation MSE: 0.0016235023504123092
- Test MSE: 0.0026038920041173697
- Test MAE: 0.02045556716620922
- Test MAPE: 2.4392325431108475

### FedAvg

Output directory: `outputs/fedavg/`

- Model parameters: 6,269,448
- Model bytes: 25,077,796
- Rounds completed: 162
- Early stop: round 161
- Runtime: 1065.5501 seconds
- Per-round total upload bytes: 75,233,388
- Per-round total download bytes: 75,233,376 after round 0
- Upload compression ratio: 1.0
- Total communication ratio: 1.0
- Attack evaluations: 8
- DLG/iDLG attack success rate: 0.0
- Test MSE: 0.002383003942668438
- Test MAE: 0.019473237916827202
- Test MAPE: 2.327566035091877

### Top-k Probe

Command:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/topk.yaml --override experiment.output_dir=outputs/topk_attack5 --override federated.rounds=5 --override training.patience=5
```

Output directory: `outputs/topk_attack5/`

- Algorithm: `sparse_fedavg` with Top-k update upload
- Top-k fraction: 0.05
- Rounds completed: 5
- Runtime: 111.3734 seconds
- Upload compression ratio: 6.666676236899415
- Total communication ratio: 1.7391308629399944
- Attack evaluations: 10
- DLG/iDLG attack success rate at `1e-4`: 0.0
- Average finite reconstruction MSE: 1.1044986069202423
- Minimum finite reconstruction MSE: 0.4481804370880127
- Test MSE: 0.002961103105917573
- Test MAE: 0.022901087999343872
- Test MAPE: 2.718646638095379
