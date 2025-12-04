import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import cv2
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="Generate RGBD side-by-side images from foreground frames.")
    parser.add_argument("--input_dir", required=True, help="Directory containing foreground_rgba images")
    parser.add_argument("--output_dir", required=True, help="Output directory for side-by-side RGBD images")
    parser.add_argument("--width", type=int, default=512, help="Resize width for processing")
    parser.add_argument("--height", type=int, default=256, help="Resize height for processing")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Depth Anything Model
    print("Loading Depth Anything V2 model...")
    try:
        # Using the small version for speed/compatibility
        checkpoint = "depth-anything/Depth-Anything-V2-Small-hf" 
        image_processor = AutoImageProcessor.from_pretrained(checkpoint)
        model = AutoModelForDepthEstimation.from_pretrained(checkpoint)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model.to(device)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Please ensure you have 'transformers' and 'torch' installed.")
        return

    frames = sorted(input_dir.glob("*.png"))
    if not frames:
        print(f"No images found in {input_dir}")
        return

    print(f"Processing {len(frames)} frames...")

    for idx, frame_path in enumerate(tqdm(frames)):
        # 2. Read Foreground Image (RGBA)
        # We need the Alpha to mask the depth later
        original_rgba = cv2.imread(str(frame_path), cv2.IMREAD_UNCHANGED)
        if original_rgba is None:
            continue
            
        # Resize if necessary
        original_rgba = cv2.resize(original_rgba, (args.width, args.height))
        
        # Extract RGB and Alpha
        rgb = original_rgba[:, :, :3]
        alpha = original_rgba[:, :, 3]

        # 3. Prepare for Depth Estimation
        # Depth Anything works best on standard RGB. 
        # We paint the background black/grey to help the model.
        # (Actually, for foreground objects, context matters. 
        # If the background is black, the model might think it's far away, which is good.)
        rgb_pil = Image.fromarray(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB))

        # 4. Infer Depth
        inputs = image_processor(images=rgb_pil, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
            predicted_depth = outputs.predicted_depth

        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=(args.height, args.width),
            mode="bicubic",
            align_corners=False,
        )

        # Normalize depth to 0-255
        depth = prediction.squeeze().cpu().numpy()
        depth_min = depth.min()
        depth_max = depth.max()
        depth_norm = (depth - depth_min) / (depth_max - depth_min + 1e-8)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)

        # 5. Mask Depth with Alpha
        # We only care about the depth of the person. Background depth should be 0 (far).
        # But in our shader, 0 means far, 1 means near.
        # Let's keep the person's depth, and set alpha=0 regions to 0.
        mask_bool = alpha > 10
        depth_masked = np.zeros_like(depth_uint8)
        depth_masked[mask_bool] = depth_uint8[mask_bool]

        # 6. Create Side-by-Side Image
        # Left: RGB, Right: Depth (Grayscale)
        # We need to save as RGBA to keep the alpha channel for the RGB part
        
        # Construct the right side (Depth)
        # It's grayscale, so R=G=B=depth. Alpha can be 255 (opaque) or same as mask.
        depth_rgb = cv2.cvtColor(depth_masked, cv2.COLOR_GRAY2BGR)
        depth_rgba = cv2.cvtColor(depth_rgb, cv2.COLOR_BGR2BGRA)
        # Optional: Apply alpha to depth side too? 
        # Usually depth maps are fully opaque blocks, but let's keep it clean.
        depth_rgba[:, :, 3] = alpha 

        # Combine
        combined = np.hstack([original_rgba, depth_rgba])

        # 7. Save
        # Ensure filename is frame_000000.png
        out_name = f"frame_{idx:06d}.png"
        cv2.imwrite(str(output_dir / out_name), combined)

    print("Done! RGBD sequence generated.")
