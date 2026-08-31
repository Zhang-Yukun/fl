"""Saved-update capture persistence for offline attack replay."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch


def save_captured_update_records(output_dir: Path, records: list[dict[str, Any]]) -> list[Path]:
    """Persist captured server-visible updates under per-client subdirectories."""

    if not records:
        return []
    capture_root = output_dir / "saved_updates"
    capture_root.mkdir(parents=True, exist_ok=True)
    saved_paths: list[Path] = []
    for record in sorted(records, key=lambda item: (str(item["client_id"]), int(item["round_index"]))):
        client_id = str(record["client_id"])
        round_index = int(record["round_index"])
        relative_path = Path(client_id) / f"round_{round_index:04d}.pt"
        path = capture_root / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(record, path)
        saved_paths.append(path)
    index: list[dict[str, Any]] = []
    for path in sorted(capture_root.rglob("round_*.pt")):
        payload = torch.load(path, map_location="cpu", weights_only=False)
        index.append(
            {
                "client_id": str(payload["client_id"]),
                "round_index": int(payload["round_index"]),
                "target_type": payload.get("target_type", "update_payload"),
                "path": str(path.relative_to(output_dir)),
            }
        )
    with (capture_root / "index.json").open("w", encoding="utf-8") as handle:
        json.dump(index, handle, ensure_ascii=False, indent=2)
    return saved_paths


def load_captured_update_records(run_dir: Path) -> list[dict[str, Any]]:
    """Load persisted per-client update captures sorted by round and client."""

    capture_root = Path(run_dir) / "saved_updates"
    if not capture_root.exists():
        return []
    index_path = capture_root / "index.json"
    if index_path.exists():
        index = json.loads(index_path.read_text(encoding="utf-8"))
        records = [torch.load(Path(run_dir) / entry["path"], map_location="cpu", weights_only=False) for entry in index]
    else:
        records = [torch.load(path, map_location="cpu", weights_only=False) for path in sorted(capture_root.rglob("round_*.pt"))]
    return sorted(records, key=lambda item: (int(item["round_index"]), str(item["client_id"])))
