
import argparse
import sys
import os
import cv2
import argparse
import sys
import os
import cv2
import numpy as np
import gc
from pathlib import Path
from tqdm import tqdm

# Ensure src is in path if running from root
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.utils import ensure_dir, to_frame_name, discover_frames, write_manifest, to_frame_index

def cmd_extract(args):
    from src.segmentation import RaftSegmenter, FarnebackSegmenter, clean_mask, inpaint_background, save_foreground
    output_dir = Path(args.output)
    bg_dir = output_dir / "background_frames"
    mask_dir = output_dir / "masks"
    fg_dir = output_dir / "foreground_rgba"

    for directory in (bg_dir, mask_dir, fg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    if args.mask_method == "raft":
        if not args.raft_model:
            raise ValueError("--raft_model is required when mask_method=raft")
        
        # Auto-detect small model
        model_path = Path(args.raft_model).expanduser().resolve()
        is_small = args.raft_small
        if "small" in model_path.name and not is_small:
            print(f"Info: Detected 'small' in model path {model_path.name}, enabling small RAFT architecture.")
            is_small = True

        segmenter = RaftSegmenter(
            model_path=model_path,
            threshold=args.mask_threshold,
            iters=args.iters,
            small=is_small,
            mixed_precision=args.mixed_precision,
            alternate_corr=args.alternate_corr,
        )
    else:
        segmenter = FarnebackSegmenter(threshold=args.mask_threshold)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    ret, initial_frame = cap.read()
    if not ret:
        raise RuntimeError("Video has no frames.")
    current = cv2.resize(initial_frame, (args.width, args.height))
    prev_for_flow = current.copy()
    frame_idx = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if args.max_frames != -1:
        total_frames = min(total_frames, args.max_frames)

    pbar = tqdm(total=total_frames, unit="frame")

    while True:
        if args.max_frames != -1 and frame_idx >= args.max_frames:
            break

        if frame_idx % args.frame_interval == 0:
            flow_max = 0.0
            if frame_idx == 0:
                mask = np.zeros(current.shape[:2], dtype=np.uint8)
                raw_px = 0
                final_px = 0
            else:
                mask, flow_max = segmenter.compute_mask(prev_for_flow, current)
                raw_px = np.count_nonzero(mask)
                mask = clean_mask(mask, args.mask_kernel)
                final_px = np.count_nonzero(mask)

            background = inpaint_background(current, mask, args.inpaint_radius)

            frame_name = to_frame_name(frame_idx)
            cv2.imwrite(str(bg_dir / f"{frame_name}.png"), background)
            cv2.imwrite(str(mask_dir / f"{frame_name}.png"), mask)
            if args.save_foreground:
                save_foreground(current, mask, fg_dir / f"{frame_name}.png")

            pbar.set_postfix(max_flow=f"{flow_max:.2f}", raw_px=raw_px, final_px=final_px)
            prev_for_flow = current.copy()
        
        pbar.update(1)

        frame_idx += 1
        ret, next_frame = cap.read()
        if not ret:
            break
        current = cv2.resize(next_frame, (args.width, args.height))

    pbar.close()
    cap.release()
    print(f"Finished extracting {frame_idx} frames into {output_dir}")

def cmd_background(args):
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

    # Lazy import for MPI generation
    import tensorflow as tf
    from src.mpi import load_model, process_frame
    from single_view_mpi.libs import mpi

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
        manifest_path = Path(args.manifest)
        write_manifest(output_dir, manifest_path)

def cmd_foreground(args):
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    frame_paths = sorted([p for p in input_dir.iterdir() if p.is_file() and p.suffix.lower() in ('.png', '.jpg', '.jpeg')])
    if not frame_paths:
        raise RuntimeError(f"No foreground images found in {input_dir}")
    if args.max_frames != -1:
        frame_paths = frame_paths[: args.max_frames]

    # Lazy import for MPI generation
    import tensorflow as tf
    from src.mpi import load_model, build_atlas_optimized

    model = load_model()
    # depths = mpi.make_depths(1.0, 100.0, 32).numpy() # Not used in optimized build? Wait, process_frame uses it.
    # Ah, generate_foreground_atlas.py uses model() directly, not process_frame.
    
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

from src.generate_index import generate_index

def cmd_index(args):
    generate_index()

def cmd_serve(args):
    import http.server
    import socketserver
    
    PORT = args.port
    Handler = http.server.SimpleHTTPRequestHandler
    
    print(f"Serving at http://127.0.0.1:{PORT}/docs/index.html")
    print("Press Ctrl+C to stop.")
    
    with socketserver.TCPServer(("", PORT), Handler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nServer stopped.")

def main():
    parser = argparse.ArgumentParser(description="PanoSynthVR Pipeline Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Serve
    p_serve = subparsers.add_parser("serve", help="Start local HTTP server")
    p_serve.add_argument("--port", type=int, default=3600, help="Port number (default: 3600)")

    # Extract
    p_extract = subparsers.add_parser("extract", help="Extract foreground masks and background frames")
    p_extract.add_argument("--video", required=True, help="Input video file")
    p_extract.add_argument("--output", required=True, help="Output directory")
    p_extract.add_argument("--width", type=int, required=True, help="Processing width")
    p_extract.add_argument("--height", type=int, required=True, help="Processing height")
    p_extract.add_argument("--mask_method", choices=["raft", "farneback"], default="raft")
    p_extract.add_argument("--raft_model", help="Path to RAFT checkpoint (.pth)")
    p_extract.add_argument("--mask_threshold", type=float, default=1.5, help="Magnitude threshold for flow mask")
    p_extract.add_argument("--mask_kernel", type=int, default=5, help="Morphology kernel size")
    p_extract.add_argument("--inpaint_radius", type=int, default=3, help="Inpaint radius for background fill")
    p_extract.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames to process")
    p_extract.add_argument("--iters", type=int, default=12, help="RAFT inference iterations")
    p_extract.add_argument("--raft_small", action="store_true", help="Use RAFT small architecture")
    p_extract.add_argument("--mixed_precision", action="store_true", help="Use RAFT mixed precision")
    p_extract.add_argument("--alternate_corr", action="store_true", help="Use RAFT alternate correlation")
    p_extract.add_argument("--save_foreground", action="store_true", help="Store RGBA foreground frames")
    p_extract.add_argument("--frame_interval", type=int, default=1, help="Process every Nth frame")

    # Background
    p_bg = subparsers.add_parser("background", help="Generate MPI atlas for background frames")
    p_bg.add_argument("--frames_dir", required=True, help="Directory containing frame_*.png backgrounds")
    p_bg.add_argument("--output", required=True, help="Output directory for MPI results")
    p_bg.add_argument("--width", type=int, required=True, help="Processing width (must match extractor)")
    p_bg.add_argument("--height", type=int, required=True, help="Processing height (must match extractor)")
    p_bg.add_argument("--max_frames", type=int, default=-1, help="Limit processed frames")
    p_bg.add_argument("--manifest", help="Optional manifest destination path")
    p_bg.add_argument("--no_atlas", action="store_true", help="Skip atlas creation and only dump layers")

    # Foreground
    p_fg = subparsers.add_parser("foreground", help="Generate MPI atlas for foreground RGBA frames")
    p_fg.add_argument("--input_dir", required=True, help="Directory containing frame_XXXXXX RGBA images")
    p_fg.add_argument("--output", required=True, help="Output directory for atlas folders")
    p_fg.add_argument("--width", type=int, required=True, help="Processing width")
    p_fg.add_argument("--height", type=int, required=True, help="Processing height")
    p_fg.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames")
    p_fg.add_argument("--manifest", help="Optional manifest destination path")

    # Index
    p_index = subparsers.add_parser("index", help="Generate index.html with links to all scenes")

    args = parser.parse_args()
    
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "background":
        cmd_background(args)
    elif args.command == "foreground":
        cmd_foreground(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "serve":
        cmd_serve(args)

if __name__ == "__main__":
    main()
