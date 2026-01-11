import argparse
import re
from pathlib import Path

import cv2
import numpy as np


def discover_masks(mask_dir: Path, mask_glob: str):
    paths = sorted(mask_dir.glob(mask_glob))
    mapping = {}
    for idx, p in enumerate(paths):
        stem = p.stem
        m = re.match(r"^(\d+)", stem)
        if m:
            mapping.setdefault(int(m.group(1)), []).append(p)
        else:
            mapping.setdefault(idx, []).append(p)
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description="Preview precomputed masks by overlaying them onto the source video.")
    parser.add_argument("--video", required=True)
    parser.add_argument("--mask_dir", required=True)
    parser.add_argument("--mask_glob", default="*.png")
    parser.add_argument("--mask_offset", type=int, default=0)
    parser.add_argument("--width", type=int, default=0, help="Resize width (0 keeps original)")
    parser.add_argument("--height", type=int, default=0, help="Resize height (0 keeps original)")
    parser.add_argument("--max_frames", type=int, default=-1)
    parser.add_argument("--invert_mask", action="store_true")
    parser.add_argument("--mask_bin_threshold", type=int, default=128)
    parser.add_argument("--alpha", type=float, default=0.5, help="Overlay alpha (0-1)")
    parser.add_argument("--output", required=True, help="Output mp4 path")
    args = parser.parse_args()

    mask_dir = Path(args.mask_dir)
    if not mask_dir.exists():
        raise FileNotFoundError(mask_dir)

    masks = discover_masks(mask_dir, args.mask_glob)
    if not masks:
        raise RuntimeError(f"No masks found in {mask_dir} matching {args.mask_glob}")

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    ret, first = cap.read()
    if not ret:
        raise RuntimeError("Video has no frames")

    src_h, src_w = first.shape[:2]
    out_w = args.width or src_w
    out_h = args.height or src_h

    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(out_path), fourcc, fps, (out_w, out_h))
    if not writer.isOpened():
        raise RuntimeError(f"Unable to open VideoWriter: {out_path}")

    def process(frame_idx: int, frame_bgr: np.ndarray):
        frame_bgr = cv2.resize(frame_bgr, (out_w, out_h), interpolation=cv2.INTER_AREA)
        mask_paths = masks.get(frame_idx + args.mask_offset)
        if not mask_paths:
            mask = np.zeros((out_h, out_w), dtype=np.uint8)
        else:
            combined = None
            for mask_path in mask_paths:
                mask_img = cv2.imread(str(mask_path), cv2.IMREAD_UNCHANGED)
                if mask_img is None:
                    continue
                if mask_img.ndim == 3:
                    mask_img = cv2.cvtColor(mask_img, cv2.COLOR_BGR2GRAY)
                mask_img = cv2.resize(mask_img, (out_w, out_h), interpolation=cv2.INTER_NEAREST)
                bin_mask = ((mask_img >= int(args.mask_bin_threshold)).astype(np.uint8) * 255)
                if args.invert_mask:
                    bin_mask = 255 - bin_mask
                combined = bin_mask if combined is None else np.maximum(combined, bin_mask)
            mask = combined if combined is not None else np.zeros((out_h, out_w), dtype=np.uint8)

        overlay = frame_bgr.copy()
        red = np.zeros_like(frame_bgr)
        red[:, :, 2] = 255
        m = (mask > 0)
        overlay[m] = (args.alpha * red[m] + (1.0 - args.alpha) * overlay[m]).astype(np.uint8)
        return overlay

    frame_idx = 0
    writer.write(process(frame_idx, first))
    frame_idx += 1
    written = 1

    while True:
        if args.max_frames != -1 and written >= args.max_frames:
            break
        ret, frame = cap.read()
        if not ret:
            break
        writer.write(process(frame_idx, frame))
        frame_idx += 1
        written += 1
        if written % 25 == 0:
            print(f"Wrote {written} frames...")

    cap.release()
    writer.release()
    print(f"Done. Wrote {written} frames to: {out_path}")


if __name__ == "__main__":
    main()
