import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import cv2
import os
from pathlib import Path
from tqdm import tqdm

def frames_to_video(input_dir, output_video, fps=30, ref_video=None, max_frames=-1):
    input_path = Path(input_dir)
    
    # List images
    extensions = ['*.png', '*.jpg', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(list(input_path.glob(ext)))
    files = sorted(files)
    
    if not files:
        print(f"No images found in {input_dir}")
        return

    if max_frames > 0:
        files = files[:max_frames]

    # Read first image to get dimensions
    first_frame = cv2.imread(str(files[0]))
    if first_frame is None:
        raise RuntimeError(f"Failed to read first frame {files[0]}")
    
    height, width, layers = first_frame.shape
    size = (width, height)
    
    # Determine FPS
    if ref_video:
        cap = cv2.VideoCapture(ref_video)
        if cap.isOpened():
            fps = cap.get(cv2.CAP_PROP_FPS)
            print(f"Read FPS from {ref_video}: {fps}")
            cap.release()
        else:
            print(f"Warning: Could not open {ref_video}, using default FPS: {fps}")
    
    print(f"Found {len(files)} frames.")
    print(f"Video resolution: {width}x{height}, FPS: {fps}")
    
    # Initialize VideoWriter
    # Try avc1 (H.264) first for better compatibility
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_video, fourcc, fps, size)
    
    if not out.isOpened():
        print("Failed to open VideoWriter with avc1, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, fps, size)
        
    if not out.isOpened():
        raise RuntimeError("Failed to open VideoWriter with both avc1 and mp4v")
    
    for file_path in tqdm(files):
        img = cv2.imread(str(file_path))
        if img is None:
            print(f"Skipping unreadable frame: {file_path}")
            continue
            
        # Resize if dimensions don't match
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, size)
            
        out.write(img)
        
    out.release()
    print(f"Video saved to {output_video}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert sequence of images to video")
    parser.add_argument("--input", required=True, help="Input directory of images")
    parser.add_argument("--output", required=True, help="Output video path")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30)")
    parser.add_argument("--ref_video", help="Optional: Path to original video to copy FPS from")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to process")
    
    args = parser.parse_args()
    
    frames_to_video(args.input, args.output, args.fps, args.ref_video, args.max_frames)
