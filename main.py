
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
    from src.segmentation import (
        RaftSegmenter,
        FarnebackSegmenter,
        BackgroundSubtractionSegmenter,
        LaMaInpainter,
        clean_mask,
        inpaint_background,
        save_foreground,
    )
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
    elif args.mask_method == "background_subtraction":
        if not args.bg_image:
            raise ValueError("--bg_image is required when mask_method=background_subtraction")
        bg_img = cv2.imread(args.bg_image)
        if bg_img is None:
            raise FileNotFoundError(f"Background image not found: {args.bg_image}")
        segmenter = BackgroundSubtractionSegmenter(bg_img, args.mask_threshold)
    else:
        segmenter = FarnebackSegmenter(threshold=args.mask_threshold)

    lama_inpainter = None
    if args.inpaint_method == "lama":
        if not args.lama_ckpt:
            raise ValueError("--lama_ckpt is required when inpaint_method=lama")
        lama_inpainter = LaMaInpainter(
            repo_path=args.lama_repo,
            config_path=args.lama_config,
            checkpoint_path=args.lama_ckpt,
            device=args.lama_device,
        )

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

    from collections import deque
    
    # Buffer to store past frames for look-ahead flow computation
    # We want to compute flow between frame[i] and frame[i + interval]
    # So we need a buffer of size 'interval'.
    # When we have 'interval' frames in buffer and read one more (current),
    # buffer[0] is frame[i] and current is frame[i + interval].
    frame_buffer = deque()
    
    # Pre-fill buffer with first 'interval' frames
    while len(frame_buffer) < args.frame_interval:
        if args.max_frames != -1 and frame_idx >= args.max_frames:
            break
        
        # We already read the first frame before the loop
        if frame_idx == 0:
            frame_buffer.append((frame_idx, current))
            frame_idx += 1
            continue
            
        ret, frame = cap.read()
        if not ret:
            break
        
        if args.width and args.height:
            frame = cv2.resize(frame, (args.width, args.height))
        frame_buffer.append((frame_idx, frame))
        frame_idx += 1

    # Adjust total frames for progress bar
    # We only process frames that have a "future" frame 'interval' steps ahead
    # So we lose the last 'interval' frames.
    process_total = total_frames - args.frame_interval
    if process_total < 0: process_total = 0
    
    pbar = tqdm(total=process_total, unit="frame")

    while True:
        if args.max_frames != -1 and frame_idx >= args.max_frames:
            break
            
        # Read the "future" frame (frame i + interval)
        ret, future_frame = cap.read()
        if not ret:
            break
            
        if args.width and args.height:
            future_frame = cv2.resize(future_frame, (args.width, args.height))
            
        # The frame to process is at the front of the buffer
        if not frame_buffer:
            break
            
        target_idx, target_img = frame_buffer.popleft()
        
        # Add the future frame to buffer (it will be a target later)
        frame_buffer.append((frame_idx, future_frame))
        
        # Compute mask: Flow(target -> future)
        # target_img is Frame i
        # future_frame is Frame i + interval
        mask, flow_max = segmenter.compute_mask(target_img, future_frame)
        
        mask = clean_mask(mask, args.mask_kernel)
        
        # Inpaint and Save
        background = inpaint_background(
            target_img,
            mask,
            args.inpaint_radius,
            method=args.inpaint_method,
            lama_inpainter=lama_inpainter,
        )
        frame_name = to_frame_name(target_idx)
        
        cv2.imwrite(str(bg_dir / f"{frame_name}.png"), background)
        cv2.imwrite(str(mask_dir / f"{frame_name}.png"), mask)
        if args.save_foreground:
            save_foreground(target_img, mask, fg_dir / f"{frame_name}.png")
        
        raw_px = np.count_nonzero(mask)
        final_px = np.count_nonzero(mask)
        pbar.set_postfix(max_flow=f"{flow_max:.2f}", raw_px=raw_px, final_px=final_px)
        pbar.update(1)
        
        frame_idx += 1

    pbar.close()
    
    # Fill remaining frames with empty foregrounds
    # The buffer logic stops when it can't find a future frame, leaving some frames unprocessed.
    # We need to output them as empty foregrounds to match the expected sequence length.
    
    target_end = total_frames
    if args.max_frames != -1:
        target_end = min(total_frames, args.max_frames)
        
    if frame_idx < target_end:
        print(f"Filling remaining frames {frame_idx} to {target_end-1} with empty foregrounds...")
        
        # We need a reference image size. Use the last processed frame or read from cap if possible.
        # Since cap might be exhausted or closed, we can use 'current' if available, or create blank.
        # 'current' holds the last read frame (resized).
        
        blank_mask = np.zeros((args.height, args.width), dtype=np.uint8)
        # For background, we can just use the last known background or the current frame itself.
        # Let's use the current frame (which is effectively the background since we assume no foreground).
        
        while frame_idx < target_end:
            frame_name = to_frame_name(frame_idx)
            
            # For these frames, we assume no foreground movement could be detected
            # So background = original frame (or we could use the clean background if we had access to it here)
            # But cmd_extract doesn't know about the clean_bg file unless passed.
            # However, for the purpose of the pipeline, what matters is the FOREGROUND atlas.
            # The background atlas is generated separately from clean_bg.
            # So here we just need to output an empty foreground.
            
            # We need an image for save_foreground to work (even if mask is empty).
            # We can use a black image or the last 'current' frame.
            dummy_img = np.zeros((args.height, args.width, 3), dtype=np.uint8)
            
            # Save empty mask
            cv2.imwrite(str(mask_dir / f"{frame_name}.png"), blank_mask)
            
            # Save empty foreground
            if args.save_foreground:
                save_foreground(dummy_img, blank_mask, fg_dir / f"{frame_name}.png")
                
            # We also need to save a background frame because the pipeline might expect it
            # (though the hybrid pipeline uses clean_bg for the static background).
            # Let's just save the dummy image as background to be safe.
            cv2.imwrite(str(bg_dir / f"{frame_name}.png"), dummy_img)
            
            frame_idx += 1

    cap.release()
    print(f"Finished extracting frames into {output_dir}")

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

def cmd_video_atlas(args):
    frames_dir = Path(args.frames_dir)
    if not frames_dir.exists():
        raise FileNotFoundError(frames_dir)

    frame_dirs = sorted([p for p in frames_dir.iterdir() if p.is_dir() and p.name.startswith("frame_")])
    if not frame_dirs:
        raise RuntimeError(f"No frame_* directories found in {frames_dir}")

    if args.max_frames != -1:
        frame_dirs = frame_dirs[:args.max_frames]

    atlas_paths = [d / "atlas.png" for d in frame_dirs]
    atlas_paths = [p for p in atlas_paths if p.exists()]
    if not atlas_paths:
        raise RuntimeError(f"No atlas.png files found under {frames_dir}")

    first = cv2.imread(str(atlas_paths[0]), cv2.IMREAD_UNCHANGED)
    if first is None:
        raise RuntimeError(f"Failed to read {atlas_paths[0]}")

    height, width = first.shape[:2]
    target_width = width * 2

    fps = args.fps
    if args.ref_video:
        cap = cv2.VideoCapture(args.ref_video)
        if cap.isOpened():
            ref_fps = cap.get(cv2.CAP_PROP_FPS)
            if ref_fps and not np.isnan(ref_fps):
                fps = ref_fps
                print(f"Using FPS {fps:.4f} from {args.ref_video}")
            cap.release()

    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(args.output, fourcc, fps, (int(target_width), int(height)))
    if not out.isOpened():
        print("Failed to open VideoWriter with avc1, falling back to mp4v")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(args.output, fourcc, fps, (int(target_width), int(height)))
    if not out.isOpened():
        raise RuntimeError("Unable to open VideoWriter with avc1 or mp4v")

    processed = 0
    for atlas_path in atlas_paths:
        img = cv2.imread(str(atlas_path), cv2.IMREAD_UNCHANGED)
        if img is None:
            print(f"Skipping unreadable atlas: {atlas_path}")
            continue

        if img.shape[0] != height or img.shape[1] != width:
            img = cv2.resize(img, (width, height))

        if img.shape[2] == 4:
            color = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            alpha = img[:, :, 3]
        else:
            color = img if img.shape[2] == 3 else cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            alpha = np.full((height, width), 255, dtype=np.uint8)
        alpha_rgb = cv2.merge([alpha, alpha, alpha])
        frame = np.concatenate([color, alpha_rgb], axis=1)
        out.write(frame)
        processed += 1

    out.release()
    print(f"Wrote {processed} frames to {args.output} at {fps:.4f} FPS")

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

def cmd_clean_bg(args):
    print(f"Generating clean background from {args.video}...")
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Strategy: Sample N frames evenly distributed across the video
    # More frames = better result but more memory/time
    sample_count = args.sample_count
    indices = np.linspace(0, total_frames - 1, sample_count, dtype=int)
    
    frames = []
    print(f"Sampling {sample_count} frames for median filtering...")
    
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize to specified resolution to save memory and match pipeline resolution
            if args.width and args.height:
                frame = cv2.resize(frame, (args.width, args.height))
            frames.append(frame)
    
    cap.release()

    if not frames:
        raise RuntimeError("No frames could be read from the video.")

    print("Computing temporal median (this may take a moment)...")
    # Stack frames: (N, H, W, 3)
    stack = np.stack(frames, axis=0)
    
    # Compute median along the time axis (axis 0)
    median_frame = np.median(stack, axis=0).astype(np.uint8)
    
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), median_frame)
    print(f"Success! Clean background saved to: {output_path}")

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
    p_extract.add_argument("--mask_method", choices=["raft", "farneback", "background_subtraction"], default="raft")
    p_extract.add_argument("--bg_image", help="Path to clean background image for background_subtraction method")
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
    p_extract.add_argument(
        "--inpaint_method",
        choices=["opencv", "lama"],
        default="opencv",
        help="Background inpainting backend (OpenCV Telea or LaMa).",
    )
    p_extract.add_argument(
        "--lama_repo",
        default=os.path.join("submodules", "lama"),
        help="Path to the local clone of https://github.com/advimman/lama",
    )
    p_extract.add_argument(
        "--lama_config",
        default=os.path.join("submodules", "lama", "configs", "prediction", "default.yaml"),
        help="Path to the LaMa config YAML used for inference.",
    )
    p_extract.add_argument(
        "--lama_ckpt",
        help="Path to the LaMa checkpoint (.ckpt or .safetensors). Required when --inpaint_method=lama.",
    )
    p_extract.add_argument(
        "--lama_device",
        default="auto",
        help="Device for LaMa inference: auto, cpu, cuda, or cuda:N",
    )

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

    # Video Atlas
    p_vid = subparsers.add_parser("video_atlas", help="Pack atlas frames into a side-by-side video for renderer mode=video_atlas")
    p_vid.add_argument("--frames_dir", required=True, help="Directory containing frame_XXXXXX folders with atlas.png")
    p_vid.add_argument("--output", required=True, help="Output video path (e.g., docs/assets/scene/atlas_video.mp4)")
    p_vid.add_argument("--fps", type=float, default=24.0, help="Frames per second (default: 24)")
    p_vid.add_argument("--ref_video", help="Optional source video to copy FPS from")
    p_vid.add_argument("--max_frames", type=int, default=-1, help="Limit number of atlas frames to encode")

    # Clean Background
    p_clean = subparsers.add_parser("clean_bg", help="Generate a clean static background using temporal median")
    p_clean.add_argument("--video", required=True, help="Input video file")
    p_clean.add_argument("--output", required=True, help="Output path for the clean background image")
    p_clean.add_argument("--sample_count", type=int, default=50, help="Number of frames to sample for median (default: 50)")
    p_clean.add_argument("--width", type=int, help="Resize width")
    p_clean.add_argument("--height", type=int, help="Resize height")

    # Index
    p_index = subparsers.add_parser("index", help="Generate index.html with links to all scenes")

    args = parser.parse_args()
    
    if args.command == "extract":
        cmd_extract(args)
    elif args.command == "background":
        cmd_background(args)
    elif args.command == "foreground":
        cmd_foreground(args)
    elif args.command == "video_atlas":
        cmd_video_atlas(args)
    elif args.command == "clean_bg":
        cmd_clean_bg(args)
    elif args.command == "index":
        cmd_index(args)
    elif args.command == "serve":
        cmd_serve(args)

if __name__ == "__main__":
    main()
