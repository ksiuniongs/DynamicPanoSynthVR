import argparse
import os
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and resize video frames to a folder.")
    parser.add_argument("--video", required=True, help="Input video path")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--width", type=int, required=True, help="Output width")
    parser.add_argument("--height", type=int, required=True, help="Output height")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to extract (-1 for all)")
    parser.add_argument(
        "--ext",
        default="png",
        choices=["png", "jpg", "jpeg"],
        help="Image extension/format (default: png)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start frame index (default: 0)",
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    if args.start > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start)

    frame_idx = args.start
    written = 0
    while True:
        if args.max_frames != -1 and written >= args.max_frames:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        resized = cv2.resize(frame_bgr, (args.width, args.height), interpolation=cv2.INTER_AREA)
        out_path = output_dir / f"frame_{written:06d}.{args.ext}"
        ok = cv2.imwrite(str(out_path), resized)
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")

        written += 1
        frame_idx += 1
        if written % 25 == 0:
            print(f"Wrote {written} frames...")

    cap.release()
    print(f"Done. Wrote {written} frames to: {output_dir}")


if __name__ == "__main__":
    main()

