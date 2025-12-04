import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
from tqdm import tqdm

from single_view_mpi.libs import mpi
from generate_full_atlas_video import load_model, build_atlas

def process_frames(input_dir, output_dir, width, height):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # List images
    extensions = ['*.jpg', '*.png', '*.jpeg']
    files = []
    for ext in extensions:
        files.extend(list(input_path.glob(ext)))
    files = sorted(files)
    
    if not files:
        print(f"No images found in {input_dir}")
        return

    print("Loading model...")
    model = load_model()
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()
    
    print(f"Processing {len(files)} frames...")
    
    for file_path in tqdm(files):
        # Output filename
        out_name = file_path.stem + ".png" # Atlas is usually png
        out_file = output_path / out_name
        
        if out_file.exists():
            continue
            
        # Read image
        frame_bgr = cv2.imread(str(file_path))
        if frame_bgr is None:
            print(f"Failed to read {file_path}")
            continue
            
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0
        
        # Resize
        frame_tensor = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
        frame_tensor = tf.image.resize(frame_tensor, (height, width), method='area')
        input_rgb = frame_tensor.numpy()
        
        # Pad
        h, w = input_rgb.shape[:2]
        padding = w // 4
        left = input_rgb[:, 0:padding]
        right = input_rgb[:, w - padding:w]
        input_rgb_padded = np.concatenate((right, input_rgb, left), axis=1)
        
        # Inference
        layers_padded = model(input_rgb_padded[tf.newaxis])[0]
        layers = layers_padded[:, :, padding:-padding, :]
        
        # Build atlas
        atlas = build_atlas(layers, width, height)
        
        # Save
        cv2.imwrite(str(out_file), atlas)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="step1_frames", help="Input directory of frames")
    parser.add_argument("--output", default="step2_atlases", help="Output directory for atlases")
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=512)
    args = parser.parse_args()
    
    process_frames(args.input, args.output, args.width, args.height)
