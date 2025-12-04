import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import shutil
import sys
import time
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

# Import generation logic
from generate_full_atlas_video import load_model, process_frame
from single_view_mpi.libs import mpi

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def generate_atlases(video_path, output_dir, width, height, max_frames=-1):
    print(f"Loading model...")
    model = load_model()
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if max_frames > 0:
        total_frames = min(total_frames, max_frames)

    print(f"Processing {total_frames} frames from {video_path}...")
    
    atlas_dir = os.path.join(output_dir, "atlas")
    ensure_dir(atlas_dir)

    frame_idx = 0
    while True:
        if max_frames > -1 and frame_idx >= max_frames:
            break

        ret, frame_bgr = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frame_rgb = frame_rgb.astype(np.float32) / 255.0

        # We need to adapt process_frame to save directly to our desired path
        # or we can use it as is and move files, but process_frame creates folders.
        # Let's just copy the logic from process_frame but simplified for this use case.
        
        # Resize
        frame_rgb_tf = tf.convert_to_tensor(frame_rgb, dtype=tf.float32)
        frame_rgb_tf = tf.image.resize(frame_rgb_tf, (height, width), method='area')
        input_rgb = frame_rgb_tf.numpy()

        # Padding
        h, w = input_rgb.shape[:2]
        padding = w // 4
        left = input_rgb[:, 0:padding]
        right = input_rgb[:, w - padding:w]
        input_rgb_padded = np.concatenate((right, input_rgb, left), axis=1)

        # Inference
        layers_padded = model(input_rgb_padded[tf.newaxis])[0]
        layers = layers_padded[:, :, padding:-padding, :]

        # Build Atlas
        # Reuse build_atlas from generate_full_atlas_video logic
        # We can import it or reimplement it. It's short.
        
        # build_atlas logic:
        rows = 8
        cols = 4
        atlas = np.zeros((height * rows, width * cols, 4), dtype='uint8')
        n = 0
        for r in range(rows):
            myr = (rows - 1) - r
            for c in range(cols):
                layer = layers[n].numpy()
                layer[:, :, :3] *= layer[:, :, 3:]  # pre-multiply alpha
                layer = (layer * 255).astype('uint8')
                layer_bgra = cv2.cvtColor(layer, cv2.COLOR_RGBA2BGRA)
                
                atlas[height * myr:height * (myr + 1), width * c:width * (c + 1)] = layer_bgra[:, ::-1]
                n += 1
        
        output_filename = f"frame_{frame_idx:06d}.png"
        cv2.imwrite(os.path.join(atlas_dir, output_filename), atlas)

        if (frame_idx + 1) % 10 == 0:
            print(f"Processed {frame_idx + 1}/{total_frames} frames")

        frame_idx += 1

    cap.release()
    print(f"Finished processing {frame_idx} frames.")
    return frame_idx, fps

def serve_and_view(output_name, frame_count, fps):
    ip = "127.0.0.1"
    port = 3600
    
    # The viewer expects files in docs/assets/[name]
    # We assume output_dir was docs/assets/[name]
    
    url = f"http://{ip}:{port}/docs/renderer.html?mode=sequence&name={output_name}&frames={frame_count}&fps={fps}"
    print(f"Opening viewer at: {url}")

    def start_server():
        # Serve from current directory (project root)
        server_address = (ip, port)
        httpd = HTTPServer(server_address, SimpleHTTPRequestHandler)
        print(f"Serving on {ip}:{port}...")
        httpd.serve_forever()

    threading.Thread(target=start_server, daemon=True).start()
    webbrowser.open_new(url)

    print("Press Ctrl+C to stop server")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping server...")
        sys.exit(0)

def main():
    parser = argparse.ArgumentParser(description="Generate MCI video from input video and view in WebXR")
    parser.add_argument("--input", required=True, help="Input video file")
    parser.add_argument("--width", type=int, default=512, help="Output width per frame")
    parser.add_argument("--height", type=int, default=256, help="Output height per frame")
    parser.add_argument("--max_frames", type=int, default=-1, help="Max frames to process")
    parser.add_argument("--skip_generation", action="store_true", help="Skip generation if files exist")
    
    args = parser.parse_args()

    video_name = Path(args.input).stem
    output_base = Path("docs/assets")
    output_dir = output_base / video_name
    
    if not args.skip_generation:
        if output_dir.exists():
            print(f"Warning: Output directory {output_dir} exists. Overwriting...")
            # shutil.rmtree(output_dir) # Optional: clear it
        
        ensure_dir(output_dir)
        
        frame_count, fps = generate_atlases(
            args.input, 
            str(output_dir), 
            args.width, 
            args.height, 
            args.max_frames
        )
    else:
        # Try to guess frame count if skipping generation
        atlas_dir = output_dir / "atlas"
        frames = list(atlas_dir.glob("frame_*.png"))
        frame_count = len(frames)
        fps = 30 # Default if skipping
        print(f"Skipping generation. Found {frame_count} frames.")

    serve_and_view(video_name, frame_count, fps)

if __name__ == "__main__":
    main()
