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

## Attack Configuration Note

The recorded 2026-07-31 FedAvg attack results above were produced with the earlier lightweight attack setting: `frequency_rounds=50`, `steps=5`, `model_mode=train`, and CPU execution. That setup was only a smoke-level attack sanity check. The current experiment configs have been strengthened for compression/privacy exploration:

- `runtime.device: cuda:0`
- FedAvg attack frequency: every 10 rounds
- SoteriaFL attack frequency: every 5 rounds
- DLG/iDLG optimization steps: 300
- Attack model mode: `eval`, to avoid dropout randomness during reconstruction evaluation
- Attack seed: 2026

New full runs should be compared against this stronger attack setting, not against the old smoke setting.

### GPU Strong-Attack Smoke

A one-round FedAvg smoke run verified the strengthened attack path on `cuda:0` with full PatchTST and `attack.steps=300`:

```bash
conda run -n torch_env python -m scripts.train --config configs/rawdata2_patchtst.yaml --override experiment.output_dir=outputs/rawdata2_patchtst_gpu_attack_smoke --override experiment.mode=federated --override federated.rounds=1 --override training.patience=1 --override attack.frequency_rounds=1
```

- Attack runtime: 18.7609 seconds
- DLG reconstruction MSE: 0.1506103128194809
- iDLG reconstruction MSE: 0.22638416290283203
- Attack success rate under `success_mse_threshold=1e-4`: 0.0

This confirms the prior zero success rate was not caused by a broken attack pipeline; the earlier attack was too lightweight and used a very strict success threshold.

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
