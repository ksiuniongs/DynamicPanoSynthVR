import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from single_view_mpi.libs import mpi
from generate_full_atlas_video import load_model, process_frame


def to_frame_index(stem: str) -> int:
    try:
        return int(stem.split("_")[-1])
    except ValueError:
        return -1


def discover_frames(frames_dir: Path):
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate MPI atlas sequence from background frames."
    )
    parser.add_argument("--frames_dir", required=True, help="Directory containing frame_*.png backgrounds")
    parser.add_argument("--output", required=True, help="Output directory for MPI results")
    parser.add_argument("--width", type=int, required=True, help="Processing width (must match extractor)")
    parser.add_argument("--height", type=int, required=True, help="Processing height (must match extractor)")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit processed frames")
    parser.add_argument("--manifest", help="Optional manifest destination path")
    parser.add_argument("--no_atlas", action="store_true", help="Skip atlas creation and only dump layers")
    args = parser.parse_args()

    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_files = discover_frames(frames_dir)
    if not frame_files:
        raise RuntimeError(f"No frame_*.png files found in {frames_dir}")

    if args.max_frames != -1:
        frame_files = frame_files[: args.max_frames]

    model = load_model()
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()

    for default_idx, frame_path in enumerate(frame_files):
        frame_bgr = cv2.imread(str(frame_path), cv2.IMREAD_COLOR)
        if frame_bgr is None:
            print(f"Skipping unreadable frame {frame_path}")
            continue
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        parsed_idx = to_frame_index(frame_path.stem)
        frame_index = parsed_idx if parsed_idx >= 0 else default_idx

        process_frame(
            model=model,
            frame_rgb=frame_rgb,
            depths=depths,
            output_dir=str(output_dir),
            frame_index=frame_index,
            output_width=args.width,
            output_height=args.height,
            build_atlas_output=not args.no_atlas,
        )
        print(f"Processed frame {frame_index:06d}")

    if args.manifest:
        import json

        manifest_path = Path(args.manifest)
        write_manifest(output_dir, manifest_path)


if __name__ == "__main__":
    main()
