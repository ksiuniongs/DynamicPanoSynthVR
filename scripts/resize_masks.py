import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Resize mask PNGs (nearest-neighbor) into a new folder.")
    parser.add_argument("--input_dir", required=True, help="Directory containing mask images")
    parser.add_argument("--output_dir", required=True, help="Directory to write resized masks")
    parser.add_argument("--width", type=int, required=True, help="Target width")
    parser.add_argument("--height", type=int, required=True, help="Target height")
    parser.add_argument("--glob", default="*.png", help="Glob for input masks (default: *.png)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    if not input_dir.exists():
        raise FileNotFoundError(input_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(input_dir.glob(args.glob))
    if not paths:
        raise RuntimeError(f"No files found in {input_dir} matching {args.glob}")

    written = 0
    for p in paths:
        img = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
        if img is None:
            raise RuntimeError(f"Failed to read: {p}")

        resized = cv2.resize(img, (args.width, args.height), interpolation=cv2.INTER_NEAREST)
        out_path = output_dir / p.name
        ok = cv2.imwrite(str(out_path), resized)
        if not ok:
            raise RuntimeError(f"Failed to write: {out_path}")

        written += 1
        if written % 50 == 0:
            print(f"Wrote {written}/{len(paths)}...")

    print(f"Done. Wrote {written} masks to: {output_dir}")


if __name__ == "__main__":
    main()

