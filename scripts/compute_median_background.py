import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

def main():
    parser = argparse.ArgumentParser(description="Compute the median frame from a sequence of images to create a clean, static background.")
    parser.add_argument("--input_dir", required=True, help="Directory containing background frames")
    parser.add_argument("--output", required=True, help="Output path for the median image (e.g., median_bg.png)")
    parser.add_argument("--limit", type=int, default=-1, help="Limit number of frames to use")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory not found: {input_dir}")

    # Discover frames
    frame_paths = sorted([
        p for p in input_dir.iterdir() 
        if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')
    ])

    if not frame_paths:
        raise RuntimeError(f"No images found in {input_dir}")

    if args.limit > 0:
        frame_paths = frame_paths[:args.limit]

    print(f"Computing median from {len(frame_paths)} frames...")

    # Memory check: 2048x1024x3 = 6MB. 100 frames = 600MB. 
    # If we have < 200 frames, we can load all into memory.
    # For safety, we can process in batches if needed, but for now let's try full load.
    
    # Read first frame to get shape
    first = cv2.imread(str(frame_paths[0]))
    h, w, c = first.shape
    
    # Pre-allocate stack
    stack = np.zeros((len(frame_paths), h, w, c), dtype=np.uint8)
    
    for i, p in enumerate(tqdm(frame_paths, desc="Loading frames")):
        img = cv2.imread(str(p))
        if img.shape != (h, w, c):
            img = cv2.resize(img, (w, h))
        stack[i] = img

    print("Calculating median (this may take a moment)...")
    median_frame = np.median(stack, axis=0).astype(np.uint8)

    cv2.imwrite(args.output, median_frame)
    print(f"Saved median background to {args.output}")

if __name__ == "__main__":
    main()
