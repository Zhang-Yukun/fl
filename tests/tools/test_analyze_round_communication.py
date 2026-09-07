import importlib.util
import json
import sys
from pathlib import Path

import yaml


def _load_module(path: Path, module_name: str):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


MODULE_PATH = Path(__file__).parents[2] / 'fedlab' / 'tools' / 'analyze_round_communication.py'
module = _load_module(MODULE_PATH, 'analyze_round_communication_script')


PacketRecord = module.PacketRecord
build_report = module.build_report
summarize_packets_by_windows = module.summarize_packets_by_windows
build_round_windows = module.build_round_windows
build_round_analysis = module.build_round_analysis
_build_run_experiment_command = module._build_run_experiment_command
render_report = module.render_report


def _write_run_artifacts(run_dir: Path, *, algorithm: str = 'ega_fedavg') -> None:
    metrics = [
        {
            'round': 0,
            'algorithm': algorithm,
            'model_parameters': 100,
            'model_bytes': 400,
            'total_parameter_download_bytes': 1200,
            'total_parameter_upload_bytes': 900,
            'total_parameter_bytes': 2100,
            'total_transport_download_bytes': 1500,
            'total_transport_upload_bytes': 1100,
            'total_transport_bytes': 2600,
            'total_transport_download_overhead_bytes': 300,
            'total_transport_upload_overhead_bytes': 200,
            'clients': [
                {
                    'client_id': 'Nd2O3',
                    'num_samples': 32,
                    'aggregation_payload_kind': 'ega_encoded_update' if algorithm == 'ega_fedavg' else 'dense_update',
                    'compressor': 'ega_b256_h144_s159_int8' if algorithm == 'ega_fedavg' else 'none',
                    'parameter_download_bytes': 400,
                    'parameter_upload_bytes': 300,
                    'transport_download_bytes': 500,
                    'transport_upload_bytes': 360,
                },
                {
                    'client_id': 'CeO2',
                    'num_samples': 32,
                    'aggregation_payload_kind': 'ega_encoded_update' if algorithm == 'ega_fedavg' else 'dense_update',
                    'compressor': 'ega_b256_h144_s159_int8' if algorithm == 'ega_fedavg' else 'none',
                    'parameter_download_bytes': 400,
                    'parameter_upload_bytes': 300,
                    'transport_download_bytes': 500,
                    'transport_upload_bytes': 370,
                },
                {
                    'client_id': 'La2O3',
                    'num_samples': 32,
                    'aggregation_payload_kind': 'ega_encoded_update' if algorithm == 'ega_fedavg' else 'dense_update',
                    'compressor': 'ega_b256_h144_s159_int8' if algorithm == 'ega_fedavg' else 'none',
                    'parameter_download_bytes': 400,
                    'parameter_upload_bytes': 300,
                    'transport_download_bytes': 500,
                    'transport_upload_bytes': 370,
                },
            ],
        }
    ]
    config = {
        'task': {'type': 'forecasting'},
        'model': {'name': 'patchtst'},
        'data': {'clients': ['Nd2O3', 'CeO2', 'La2O3']},
        'federated': {'algorithm': algorithm},
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'metrics.json').write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding='utf-8')
    (run_dir / 'config.yaml').write_text(yaml.safe_dump(config, allow_unicode=True, sort_keys=False), encoding='utf-8')


def test_build_round_analysis_explains_ega_round_zero_payloads(tmp_path):
    run_dir = tmp_path / 'run'
    _write_run_artifacts(run_dir, algorithm='ega_fedavg')
    report = build_report(run_dir=run_dir, first_n=1)

    round_payload = report['rounds'][0]
    assert round_payload['task_type'] == 'forecasting'
    assert round_payload['model_name'] == 'patchtst'
    assert 'download_state' in round_payload['parameter']['download_includes'][0]
    assert any('ega_codec_payload' in item for item in round_payload['parameter']['download_context_hints'])
    assert any('ega_payload' in item for item in round_payload['parameter']['upload_includes'])
    assert any('ClientResult metadata' in item for item in round_payload['transport']['upload_includes'])
    assert 'ega_payload_keys' in round_payload['protocol_shapes']['client_to_server_message']
    assert 'ega_normalization' in round_payload['protocol_shapes']['server_to_client_message']['round_context_keys']


def test_build_round_analysis_explains_dense_uploads(tmp_path):
    run_dir = tmp_path / 'run_dense'
    _write_run_artifacts(run_dir, algorithm='fedavg')
    report = build_report(run_dir=run_dir, first_n=1)

    round_payload = report['rounds'][0]
    assert round_payload['payload_kind'] == 'dense_update'
    assert any('dense client update payload' in item for item in round_payload['parameter']['upload_includes'])
    assert 'ega_payload_keys' not in round_payload['protocol_shapes']['client_to_server_message']


def test_summarize_packets_by_windows_slices_external_capture_by_round():
    packets = [
        PacketRecord(timestamp=1.0, sent_frame_bytes=100, sent_tcp_payload_bytes=70, sent_packets=1),
        PacketRecord(timestamp=1.5, received_frame_bytes=120, received_tcp_payload_bytes=90, received_packets=1),
        PacketRecord(timestamp=3.0, sent_frame_bytes=130, sent_tcp_payload_bytes=95, sent_packets=1),
        PacketRecord(timestamp=3.5, received_frame_bytes=140, received_tcp_payload_bytes=100, received_packets=1),
    ]
    windows = {0: (0.0, 2.0), 1: (2.0, 4.0)}

    summary = summarize_packets_by_windows(packets, windows)

    assert summary[0]['total_bytes'] == 220
    assert summary[0]['total_tcp_payload_bytes'] == 160
    assert summary[1]['total_bytes'] == 270
    assert summary[1]['total_tcp_payload_bytes'] == 195


def test_build_round_windows_uses_previous_round_end_as_start():
    packets = [PacketRecord(timestamp=10.0), PacketRecord(timestamp=20.0)]
    windows = build_round_windows([0, 1], {0: 12.0, 1: 18.0}, packets)

    assert windows[0] == (10.0, 12.0)
    assert windows[1] == (12.0, 18.0)


def test_build_run_experiment_command_targets_grpc_sync_output(tmp_path):
    command, env, run_dir = _build_run_experiment_command(
        task='rare',
        algorithm='ega',
        rounds_to_run=10,
        seed=2026,
        base_port=43000,
        runtime_device='cuda:0',
        startup_wait_seconds=7,
        poll_seconds=1.5,
        output_root=tmp_path / 'outputs',
    )

    assert command == ['bash', 'scripts/run_suite.sh', '--modes', 'grpc_sync', '--tasks', 'rare', '--algorithms', 'ega']
    assert env['MONITOR_GRPC_PORT_TRAFFIC'] == 'true'
    assert env['ROUNDS'] == '10'
    assert env['BASE_PORT'] == '43000'
    assert env['RUNTIME_DEVICE'] == 'cuda:0'
    assert env['TASK_SET'] == 'rare'
    assert env['FEDERATED_ALGORITHMS'] == 'ega'
    assert run_dir == (tmp_path / 'outputs' / 'rare' / 'ega')


def test_render_report_uses_chinese_labels(tmp_path):
    run_dir = tmp_path / 'run_cn'
    _write_run_artifacts(run_dir, algorithm='ega_fedavg')
    report = build_report(run_dir=run_dir, first_n=1)
    rendered = render_report(report)

    assert '实验目录:' in rendered
    assert 'Parameter 口径:' in rendered
    assert 'Transport 口径:' in rendered
    assert 'grpc_port_traffic.summary.json key:' in rendered
