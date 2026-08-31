from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


PART_SPECS = (
    ("run_exp_seed42_part1.sh", "42", "58000", "single_sync"),
    ("run_exp_seed42_part2.sh", "42", "58100", "multi_sync"),
    ("run_exp_seed4096_part1.sh", "4096", "59000", "single_sync"),
    ("run_exp_seed4096_part2.sh", "4096", "59100", "multi_sync"),
    ("run_exp_seed2026_part1.sh", "2026", "60000", "single_sync"),
    ("run_exp_seed2026_part2.sh", "2026", "60100", "multi_sync"),
    ("run_exp_seed8192_part1.sh", "8192", "61000", "single_sync"),
    ("run_exp_seed8192_part2.sh", "8192", "61100", "multi_sync"),
)


def test_seed_part_wrappers_delegate_with_expected_split():
    for name, seed, port, mode in PART_SPECS:
        path = SCRIPT_DIR / name
        assert path.exists(), name
        assert path.stat().st_mode & 0o111, name
        content = path.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash")
        assert f'SUITE_SEED="${{SUITE_SEED:-{seed}}}"' in content
        assert f'BASE_PORT="${{BASE_PORT:-{port}}}"' in content
        assert f'MODE="{mode}"' in content
        assert 'LOSSES=(mse)' in content
        assert 'BASE_ALGOS=fedavg,topk,ega' in content
        assert 'TASKS_RAW="${TASKS:-rare mnist cifar10}"' in content
        assert 'read -r -a TASKS <<< "${TASKS_RAW}"' in content
        assert 'TASK_IN_BASE_OUTPUT=true' in content
        assert 'bash scripts/run_controlled_suite.sh' in content
