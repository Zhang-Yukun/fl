#!/usr/bin/env python3
"""Materialize role-specific minimal datasets from one prepared split directory."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any


_TASK_SOURCE_DIRS = {
    'rare': 'rare_earth_rawdata2',
    'mnist': 'mnist',
    'cifar10': 'cifar10',
}


def _reset_output_dir(output_dir: Path) -> None:
    if output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _copy_file(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    return str(target)


def _discover_clients(split_dir: Path) -> list[str]:
    clients_dir = split_dir / 'clients'
    if not clients_dir.exists():
        raise FileNotFoundError(f'Missing clients directory: {clients_dir}')
    client_ids = sorted(path.name for path in clients_dir.iterdir() if path.is_dir())
    if not client_ids:
        raise ValueError(f'No client directories found under {clients_dir}')
    return client_ids


def _copy_optional(source: Path, target: Path) -> str | None:
    if not source.exists():
        return None
    return _copy_file(source, target)


def _write_summary(output_dir: Path, payload: dict[str, Any]) -> dict[str, Any]:
    summary_path = output_dir / 'summary.json'
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    return payload


def _copy_common_summary(split_dir: Path, target_dir: Path) -> str | None:
    return _copy_optional(split_dir / 'summary.json', target_dir / 'summary.json')


def prepare_rare_role_dataset(
    split_dir: Path,
    output_dir: Path,
    *,
    clients: list[str] | None = None,
    attack_clients: list[str] | None = None,
    evaluation_context_path: Path | None = None,
) -> dict[str, Any]:
    client_ids = list(clients or _discover_clients(split_dir))
    selected_attack_clients = list(attack_clients or client_ids)
    _reset_output_dir(output_dir)

    copied: dict[str, Any] = {
        'client': {},
        'server': {'files': []},
        'attack': {},
        'test': {'files': []},
    }

    for client_id in client_ids:
        client_train = split_dir / 'clients' / client_id / 'train.csv'
        if not client_train.exists():
            raise FileNotFoundError(f'Missing client train split: {client_train}')
        copied['client'][client_id] = {
            'split_dir': str(output_dir / 'client' / client_id),
            'files': [
                _copy_file(
                    client_train,
                    output_dir / 'client' / client_id / 'clients' / client_id / 'train.csv',
                ),
            ],
        }

        client_val = split_dir / 'clients' / client_id / 'val.csv'
        if not client_val.exists():
            raise FileNotFoundError(f'Missing client validation split: {client_val}')
        copied['server']['files'].append(
            _copy_file(
                client_val,
                output_dir / 'server' / 'clients' / client_id / 'val.csv',
            )
        )
        copied_test = _copy_optional(
            split_dir / 'clients' / client_id / 'test.csv',
            output_dir / 'test' / 'clients' / client_id / 'test.csv',
        )
        if copied_test is not None:
            copied['test']['files'].append(copied_test)

    for client_id in selected_attack_clients:
        client_train = split_dir / 'clients' / client_id / 'train.csv'
        if not client_train.exists():
            raise FileNotFoundError(f'Missing attack reference train split: {client_train}')
        copied['attack'][client_id] = {
            'split_dir': str(output_dir / 'attack' / client_id),
            'files': [
                _copy_file(
                    client_train,
                    output_dir / 'attack' / client_id / 'clients' / client_id / 'train.csv',
                ),
            ],
        }

    server_summary = _copy_common_summary(split_dir, output_dir / 'server')
    if server_summary is not None:
        copied['server']['files'].append(server_summary)
    test_summary = _copy_common_summary(split_dir, output_dir / 'test')
    if test_summary is not None:
        copied['test']['files'].append(test_summary)
    for client_id in client_ids:
        _copy_common_summary(split_dir, output_dir / 'client' / client_id)
    for client_id in selected_attack_clients:
        _copy_common_summary(split_dir, output_dir / 'attack' / client_id)

    evaluation_context_target = None
    if evaluation_context_path is not None:
        evaluation_context_target = _copy_file(
            evaluation_context_path,
            output_dir / 'test' / 'evaluation_context.json',
        )
        copied['test']['files'].append(evaluation_context_target)

    return _write_summary(
        output_dir,
        {
            'task': 'rare',
            'source_split_dir': str(split_dir),
            'roles': copied,
            'notes': {
                'client': 'Each client receives only its own train.csv.',
                'server': 'Server receives all client val.csv only.',
                'attack': 'Attack replay receives only selected clients train.csv.',
                'test': 'Offline test receives all client test.csv; rare-earth offline test also needs evaluation_context.json.',
                'evaluation_context_path': evaluation_context_target,
            },
        },
    )


def prepare_classification_role_dataset(
    split_dir: Path,
    output_dir: Path,
    *,
    task: str,
    clients: list[str] | None = None,
    attack_clients: list[str] | None = None,
) -> dict[str, Any]:
    client_ids = list(clients or _discover_clients(split_dir))
    selected_attack_clients = list(attack_clients or client_ids)
    _reset_output_dir(output_dir)

    copied: dict[str, Any] = {
        'client': {},
        'server': {'files': []},
        'attack': {},
        'test': {'files': []},
    }

    for client_id in client_ids:
        train_path = split_dir / 'clients' / client_id / 'train.pt'
        if not train_path.exists():
            raise FileNotFoundError(f'Missing client train split: {train_path}')
        copied['client'][client_id] = {
            'split_dir': str(output_dir / 'client' / client_id),
            'files': [
                _copy_file(
                    train_path,
                    output_dir / 'client' / client_id / 'clients' / client_id / 'train.pt',
                ),
            ],
        }

    server_val = split_dir / 'server' / 'val.pt'
    if not server_val.exists():
        raise FileNotFoundError(f'Missing server validation split: {server_val}')
    copied['server']['files'].append(_copy_file(server_val, output_dir / 'server' / 'server' / 'val.pt'))
    for client_id in selected_attack_clients:
        train_path = split_dir / 'clients' / client_id / 'train.pt'
        copied['attack'][client_id] = {
            'split_dir': str(output_dir / 'attack' / client_id),
            'files': [
                _copy_file(
                    train_path,
                    output_dir / 'attack' / client_id / 'clients' / client_id / 'train.pt',
                ),
            ],
        }

    test_source = split_dir / 'server' / 'test.pt'
    if test_source.exists():
        copied['test']['files'].append(_copy_file(test_source, output_dir / 'test' / 'server' / 'test.pt'))
    else:
        for client_id in client_ids:
            client_test = split_dir / 'clients' / client_id / 'test.pt'
            if not client_test.exists():
                raise FileNotFoundError(f'Missing classification test split: {client_test}')
            copied['test']['files'].append(
                _copy_file(
                    client_test,
                    output_dir / 'test' / 'clients' / client_id / 'test.pt',
                )
            )

    server_summary = _copy_common_summary(split_dir, output_dir / 'server')
    if server_summary is not None:
        copied['server']['files'].append(server_summary)
    test_summary = _copy_common_summary(split_dir, output_dir / 'test')
    if test_summary is not None:
        copied['test']['files'].append(test_summary)
    for client_id in client_ids:
        _copy_common_summary(split_dir, output_dir / 'client' / client_id)
    for client_id in selected_attack_clients:
        _copy_common_summary(split_dir, output_dir / 'attack' / client_id)

    return _write_summary(
        output_dir,
        {
            'task': task,
            'source_split_dir': str(split_dir),
            'roles': copied,
            'notes': {
                'client': 'Each client receives only its own train.pt.',
                'server': 'Server receives shared val.pt only.',
                'attack': 'Attack replay receives only selected clients train.pt.',
                'test': 'Offline test receives server/test.pt when available, otherwise each client test.pt.',
            },
        },
    )


def prepare_role_datasets(
    *,
    task: str,
    source_root: Path,
    output_root: Path,
    clients: list[str] | None = None,
    attack_clients: list[str] | None = None,
    evaluation_context_path: Path | None = None,
) -> dict[str, Any]:
    tasks = [task] if task != 'all' else ['rare', 'mnist', 'cifar10']
    payload: dict[str, Any] = {}
    for task_name in tasks:
        split_dir = source_root / _TASK_SOURCE_DIRS[task_name]
        target_dir = output_root / task_name
        if task_name == 'rare':
            payload[task_name] = prepare_rare_role_dataset(
                split_dir,
                target_dir,
                clients=clients,
                attack_clients=attack_clients,
                evaluation_context_path=evaluation_context_path,
            )
        else:
            payload[task_name] = prepare_classification_role_dataset(
                split_dir,
                target_dir,
                task=task_name,
                clients=clients,
                attack_clients=attack_clients,
            )
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task', choices=('rare', 'mnist', 'cifar10', 'all'), default='all')
    parser.add_argument('--source-root', type=Path, default=Path('../data'))
    parser.add_argument('--output-root', type=Path, default=Path('../data/role_datasets'))
    parser.add_argument('--clients', nargs='*', default=None, help='Optional subset of client ids to materialize')
    parser.add_argument('--attack-clients', nargs='*', default=None, help='Optional subset of client ids to materialize for attack replay')
    parser.add_argument('--evaluation-context', type=Path, default=None, help='Optional rare-earth evaluation_context.json copied into the offline test role')
    args = parser.parse_args()

    payload = prepare_role_datasets(
        task=args.task,
        source_root=args.source_root.expanduser(),
        output_root=args.output_root.expanduser(),
        clients=None if not args.clients else list(args.clients),
        attack_clients=None if not args.attack_clients else list(args.attack_clients),
        evaluation_context_path=None if args.evaluation_context is None else args.evaluation_context.expanduser(),
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
