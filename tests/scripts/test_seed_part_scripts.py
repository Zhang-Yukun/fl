from pathlib import Path


SCRIPT_DIR = Path(__file__).parents[2] / "scripts"


PART_SPECS = (
    ("run_exp_seed42_part1.sh", "42", "58000", "noattack", "single_sync"),
    ("run_exp_seed42_part2.sh", "42", "58100", "noattack", "multi_sync"),
    ("run_exp_seed42_part3.sh", "42", "58200", "attack", "single_sync"),
    ("run_exp_seed42_part4.sh", "42", "58300", "attack", "multi_sync"),
    ("run_exp_seed55_part1.sh", "55", "59000", "noattack", "single_sync"),
    ("run_exp_seed55_part2.sh", "55", "59100", "noattack", "multi_sync"),
    ("run_exp_seed55_part3.sh", "55", "59200", "attack", "single_sync"),
    ("run_exp_seed55_part4.sh", "55", "59300", "attack", "multi_sync"),
    ("run_exp_seed2026_part1.sh", "2026", "59000", "noattack", "single_sync"),
    ("run_exp_seed2026_part2.sh", "2026", "59100", "noattack", "multi_sync"),
    ("run_exp_seed2026_part3.sh", "2026", "59200", "attack", "single_sync"),
    ("run_exp_seed2026_part4.sh", "2026", "59300", "attack", "multi_sync"),
    ("run_exp_seed8192_part1.sh", "8192", "58000", "noattack", "single_sync"),
    ("run_exp_seed8192_part2.sh", "8192", "58100", "noattack", "multi_sync"),
    ("run_exp_seed8192_part3.sh", "8192", "58200", "attack", "single_sync"),
    ("run_exp_seed8192_part4.sh", "8192", "58300", "attack", "multi_sync"),
)


def test_seed_part_wrappers_delegate_with_expected_split():
    for name, seed, port, profile, mode in PART_SPECS:
        path = SCRIPT_DIR / name
        assert path.exists(), name
        assert path.stat().st_mode & 0o111, name
        content = path.read_text(encoding="utf-8")
        assert content.startswith("#!/usr/bin/env bash")
        assert f'SUITE_SEED="${{SUITE_SEED:-{seed}}}"' in content
        assert f'BASE_PORT="${{BASE_PORT:-{port}}}"' in content
        assert f'PROFILE="{profile}"' in content
        assert f'MODE="{mode}"' in content
        assert 'BASE_ALGOS=fedavg,topk,ega' in content
        assert 'TASKS=(rare mnist cifar10)' in content
        assert 'TASK_IN_BASE_OUTPUT=true' in content
        assert 'bash scripts/run_controlled_suite.sh' in content
