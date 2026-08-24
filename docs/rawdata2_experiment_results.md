# rawdata2 PatchTST Experiment Results

Run date: 2026-07-31

## Commands

```bash
bash scripts/run_rawdata2_centralized.sh
bash scripts/run_rawdata2_fedavg.sh
bash scripts/run_rawdata2_soteriafl.sh
conda run -n torch_env bash scripts/run_rawdata2_soteriafl.sh --override experiment.output_dir=outputs/soteriafl_smoke --override federated.rounds=2 --override training.patience=2 --override attack.frequency_rounds=1 --override attack.steps=1 --override attack.max_samples=1
```

The repository's default comparison set is now:

- centralized PatchTST
- standard `fedavg`
- compression-plus-security `soteriafl`

`fedaware` remains available as a supplementary Xu-related adaptive aggregation baseline, and `fedlab_topk` remains an extra compression baseline.

## Full PatchTST Centralized

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

## Full PatchTST FedAvg

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

## Attack Configuration Note

The recorded 2026-07-31 FedAvg attack results above were produced with the earlier lightweight attack setting: `frequency_rounds=50`, `steps=5`, `model_mode=train`, and CPU execution. That setup was only a smoke-level attack sanity check. The current experiment configs have been strengthened for compression/privacy exploration:

- `runtime.device: cuda:0`
- FedAvg attack frequency: every 1 round
- SoteriaFL attack frequency: every 1 round
- DLG/iDLG optimization steps: 300
- Attack model mode: `eval`, to avoid dropout randomness during reconstruction evaluation
- Attack seed: 2026

New full runs should be compared against this stronger attack setting, not against the old smoke setting.

### GPU Strong-Attack Smoke

A one-round FedAvg smoke run verified the strengthened attack path on `cuda:0` with full PatchTST and `attack.steps=300`:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/fedavg.yaml --override experiment.output_dir=outputs/fedavg_gpu_attack_smoke --override experiment.mode=federated --override federated.rounds=1 --override training.patience=1 --override attack.frequency_rounds=1
```

- Attack runtime: 18.7609 seconds
- DLG reconstruction MSE: 0.1506103128194809
- iDLG reconstruction MSE: 0.22638416290283203
- Attack success rate under `success_mse_threshold=1e-4`: 0.0

This confirms the prior zero success rate was not caused by a broken attack pipeline; the earlier attack was too lightweight and used a very strict success threshold.

## SoteriaFL Smoke

Output directory: `outputs/soteriafl_smoke/`

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


## Every-Round Attack Compression Runs

Run date: 2026-07-31

These runs use full PatchTST on `cuda:0`, evaluate DLG and iDLG every federated round, and use `attack.steps=300`. They are short 5-round probes intended to measure compression/attack behavior before launching long runs.

### FedLab-Style Top-k Sparse FedAvg

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

### FedLab-Style Top-k Sparse FedAvg 30-Round Run

Command:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/topk.yaml --override experiment.output_dir=outputs/topk_attack30
```

Output directory: `outputs/topk_attack30/`

- Algorithm: `sparse_fedavg` with Top-k update upload
- Top-k fraction: 0.05
- Rounds completed: 30
- Runtime: 632.1551 seconds
- Upload compression ratio: 6.666676236899415
- Total communication ratio: 1.7391308629399944
- Attack evaluations: 60 (DLG 30 + iDLG 30, every round)
- DLG attack success rate at `1e-4`: 0.0
- DLG average reconstruction MSE: 1.1209555685520172
- DLG minimum reconstruction MSE: 0.849487841129303
- iDLG attack success rate at `1e-4`: 0.0
- iDLG average reconstruction MSE: 1.4839166820049285
- iDLG minimum reconstruction MSE: 0.9376352429389954
- Test MSE: 0.003715549362823367
- Test MAE: 0.02532796934247017
- Test MAPE: 2.9567522928118706

### SoteriaFL-Style Random-k Local-DP Probe

Command:

```bash
conda run -n torch_env python -m fedlab.entrypoints.train --config configs/soteriafl.yaml --override experiment.output_dir=outputs/soteriafl_attack5 --override federated.rounds=5 --override training.patience=5
```

Output directory: `outputs/soteriafl_attack5/`

- Algorithm: `soteriafl` with clipped/noisy update and unbiased random-k upload
- Top-k/random-k fraction: 0.05
- Noise multiplier: 0.05
- Rounds completed: 5
- Runtime: 110.3381 seconds
- Upload compression ratio: 6.666676236899415
- Total communication ratio: 1.7391308629399944
- Attack evaluations: 10
- DLG/iDLG attack success rate at `1e-4`: 0.0
- Finite attack MSE count: 4 / 10
- Average finite reconstruction MSE: 1.0701223462820053
- Minimum finite reconstruction MSE: 0.5142163038253784
- Test metrics: NaN

The SoteriaFL-style random-k local-DP probe is currently numerically unstable for full PatchTST with `topk_fraction=0.05` and `noise_multiplier=0.05`; validation metrics became NaN from round 2 onward. This should be treated as a negative/stability result for the current hyperparameters, not as a privacy success.
