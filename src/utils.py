
import os
import json
from pathlib import Path

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def to_frame_name(idx: int) -> str:
    return f"frame_{idx:06d}"

def to_frame_index(stem: str) -> int:
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return -1

def discover_frames(frames_dir: Path):
    if not frames_dir.exists():
        return []
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    return frame_paths

def write_manifest(atlas_dir: Path, manifest_path: Path):
    entries = sorted(
        [
            f"{d.name}/atlas.png"
            for d in atlas_dir.iterdir()
            if d.is_dir() and (d / "atlas.png").exists()
        ]
    )
    manifest = {
        "basePath": "",
        "frames": entries,
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(f"Wrote manifest with {len(entries)} entries to {manifest_path}")
