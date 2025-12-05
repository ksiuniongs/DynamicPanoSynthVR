import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import json
import os
from pathlib import Path

import cv2
import gc
import numpy as np
import tensorflow as tf

from single_view_mpi.libs import mpi
from generate_full_atlas_video import load_model


def list_frames(frames_dir: Path):
    return sorted([p for p in frames_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')])


def build_atlas_optimized(layers_tensor, mask, output_width, output_height):
    rows, cols = 8, 4
    H, W = output_height, output_width
    atlas = np.zeros((H * rows, W * cols, 4), dtype=np.uint8)
    n = 0
    
    # mask is (H, W, 1) float32
    
    for r in range(rows):
        myr = (rows - 1) - r
        for c in range(cols):
            # Extract single layer from tensor: shape (H, W, 4)
            # Doing .numpy() here only brings 1 layer into CPU memory (approx 32MB)
            layer = layers_tensor[n].numpy()
            
            # Apply mask to this layer
            layer *= mask
            
            # Pre-multiply alpha
            layer[:, :, :3] *= layer[:, :, 3:]
            
            # Convert to uint8
            layer = (layer * 255).astype(np.uint8)
            
            # RGBA to BGRA
            layer = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA)
            
            # Place in atlas (flip horizontally as per original logic)
            atlas[H * myr:H * (myr + 1), W * c:W * (c + 1)] = layer[:, ::-1]
            
            n += 1
            
    return atlas


def write_manifest(atlas_dir: Path, manifest_path: Path):
    entries = sorted(
        [
            f"{folder.name}/atlas.png"
            for folder in atlas_dir.iterdir()
            if folder.is_dir() and (folder / "atlas.png").exists()
        ]
    )
    manifest = {"basePath": "", "frames": entries}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote manifest with {len(entries)} entries to {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate MPI atlases for foreground RGBA frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing frame_XXXXXX RGBA images")
    parser.add_argument("--output", required=True, help="Output directory for atlas folders")
    parser.add_argument("--width", type=int, required=True, help="Processing width")
    parser.add_argument("--height", type=int, required=True, help="Processing height")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames")
    parser.add_argument("--manifest", help="Optional manifest destination path")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = list_frames(input_dir)
    if not frame_paths:
        raise RuntimeError(f"No foreground images found in {input_dir}")
    if args.max_frames != -1:
        frame_paths = frame_paths[: args.max_frames]

    model = load_model()
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()

    for idx, frame_path in enumerate(frame_paths):
        rgba = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if rgba is None:
            print(f"Skipping unreadable frame: {frame_path}")
            continue
        if rgba.shape[2] == 3:
            alpha = np.ones((rgba.shape[0], rgba.shape[1]), dtype=np.uint8) * 255
        else:
            alpha = rgba[:, :, 3]
        resized_rgba = cv2.resize(rgba[:, :, :3], (args.width, args.height))
        resized_alpha = cv2.resize(alpha, (args.width, args.height), interpolation=cv2.INTER_NEAREST)

        rgb = cv2.cvtColor(resized_rgba, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask = (resized_alpha.astype(np.float32) / 255.0)[..., None]
        rgb_masked = rgb * mask

        frame_tensor = tf.convert_to_tensor(rgb_masked, dtype=tf.float32)
        frame_tensor = tf.image.resize(frame_tensor, (args.height, args.width), method='area')
        input_rgb = frame_tensor.numpy()

        height, width = input_rgb.shape[:2]
        padding = width // 4
        left = input_rgb[:, 0:padding]
        right = input_rgb[:, width - padding:width]
        input_rgb_padded = np.concatenate((right, input_rgb, left), axis=1)

        layers_padded = model(input_rgb_padded[tf.newaxis])[0]
        layers = layers_padded[:, :, padding:-padding, :]
        
        # Optimized atlas building: process one layer at a time
        atlas = build_atlas_optimized(layers, mask, args.width, args.height)

        frame_folder = output_dir / f"frame_{idx:06d}"
        frame_folder.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(frame_folder / "atlas.png"), atlas)
        
        # Explicit cleanup
        del layers
        del layers_padded
        del atlas
        del input_rgb
        del input_rgb_padded
        gc.collect()

        if (idx + 1) % 10 == 0:
            print(f"Processed {idx + 1} frames")

    if args.manifest:
        write_manifest(output_dir, Path(args.manifest))


if __name__ == "__main__":
    main()
