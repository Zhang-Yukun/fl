# rawdata2 PatchTST Experiment Results

Run date: 2026-07-31

## Commands

```bash
conda run -n torch_env python -m scripts.run_rawdata2_patchtst
conda run -n torch_env python -m scripts.run_rawdata2_soteriafl --override experiment.output_dir=outputs/rawdata2_soteriafl_smoke --override federated.rounds=2 --override training.patience=2 --override attack.frequency_rounds=1 --override attack.steps=1 --override attack.max_samples=1
```

## Full PatchTST Centralized

Output directory: `outputs/rawdata2_patchtst_centralized/`

- Model parameters: 6,269,448
- Model bytes: 25,077,796
- Epochs completed: 168
- Early stop: epoch 167
- Runtime: 1174.0644 seconds
- Best validation MSE: 0.0016235023504123092
- Test MSE: 0.0026038920041173697
- Test MAE: 0.02045556716620922
- Test MAPE: 2.4392325431108475

## Full PatchTST FedAvg

Output directory: `outputs/rawdata2_patchtst_fedavg/`

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

## SoteriaFL Smoke

Output directory: `outputs/rawdata2_soteriafl_smoke/`

- Algorithm: SoteriaFL-style local-DP random-k sparse uploads
- Rounds completed: 2
- Runtime: 14.7108 seconds
- Upload compression ratio: 6.666676236899415
- Total communication ratio: 1.7391308629399944
- Attack evaluations: 4
- DLG/iDLG attack success rate: 0.0
- Test MSE: 0.13125701248645782
- Test MAE: 0.14800910651683807
- Test MAPE: 17.12576001882553

Detailed metrics, logs, configs, attacks, and model files remain in the output directories above. Those directories are intentionally git-ignored to avoid committing large generated artifacts.
