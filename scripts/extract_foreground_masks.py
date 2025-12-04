import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
from typing import Tuple
from tqdm import tqdm


def to_frame_name(idx: int) -> str:
    return f"frame_{idx:06d}"


def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed


def save_foreground(frame_bgr: np.ndarray, mask: np.ndarray, path: Path):
    fg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    rgba = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(str(path), rgba)


def inpaint_background(frame_bgr: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if mask.max() == 0:
        return frame_bgr.copy()
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(frame_bgr, dilated, float(radius), cv2.INPAINT_TELEA)


class MotionSegmenter:
    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        raise NotImplementedError


class FarnebackSegmenter(MotionSegmenter):
    def __init__(self, threshold: float):
        self.threshold = threshold

    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        prev_gray = cv2.cvtColor(prev_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_bgr, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray,
            curr_gray,
            None,
            pyr_scale=0.5,
            levels=3,
            winsize=15,
            iterations=3,
            poly_n=5,
            poly_sigma=1.2,
            flags=0,
        )
        mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
        # print(f"Farneback Stats - Max Flow: {mag.max():.4f}, Mean: {mag.mean():.4f}, Threshold: {self.threshold}")
        mask = (mag > self.threshold).astype(np.uint8) * 255
        return mask, mag.max()


class RaftSegmenter(MotionSegmenter):
    def __init__(
        self,
        model_path: Path,
        threshold: float,
        iters: int,
        small: bool,
        mixed_precision: bool,
        alternate_corr: bool,
    ):
        import torch

        raft_core = Path(__file__).parent.parent / "submodules" / "RAFT" / "core"
        sys.path.append(str(raft_core))

        from raft import RAFT
        from utils.utils import InputPadder

        self.torch = torch
        self.InputPadder = InputPadder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        class AttrDict(dict):
            def __getattr__(self, key):
                return self[key]
            def __setattr__(self, key, value):
                self[key] = value
            def __contains__(self, item):
                return dict.__contains__(self, item)

        raft_args = AttrDict(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
        )
        self.model = RAFT(raft_args)
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Handle 'module.' prefix if present (from DataParallel)
        new_state = {}
        for k, v in state.items():
            if k.startswith("module."):
                new_state[k[7:]] = v
            else:
                new_state[k] = v
                
        self.model.load_state_dict(new_state)
        self.model.to(self.device)
        self.model.eval()

        self.threshold = threshold
        self.iters = iters

    def _to_tensor(self, frame_bgr: np.ndarray):
        tensor = self.torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return tensor[None].to(self.device)

    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        with self.torch.no_grad():
            image1 = self._to_tensor(prev_bgr)
            image2 = self._to_tensor(curr_bgr)
            padder = self.InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = self.model(image1, image2, iters=self.iters, test_mode=True)
            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            # print(f"RAFT Stats - Max Flow: {mag.max():.4f}, Mean: {mag.mean():.4f}, Threshold: {self.threshold}")
            mask = (mag > self.threshold).astype(np.uint8) * 255
        return mask, mag.max()


def build_segmenter(args):
    if args.mask_method == "raft":
        if not args.raft_model:
            raise ValueError("--raft_model is required when mask_method=raft")
        model_path = Path(args.raft_model).expanduser().resolve()
        if not model_path.exists():
            raise FileNotFoundError(model_path)

        # Auto-detect small model
        is_small = args.raft_small
        if "small" in model_path.name and not is_small:
            print(f"Info: Detected 'small' in model path {model_path.name}, enabling small RAFT architecture.")
            is_small = True

        return RaftSegmenter(
            model_path=model_path,
            threshold=args.mask_threshold,
            iters=args.iters,
            small=is_small,
            mixed_precision=args.mixed_precision,
            alternate_corr=args.alternate_corr,
        )
    return FarnebackSegmenter(threshold=args.mask_threshold)


def main():
    parser = argparse.ArgumentParser(
        description="Extract background frames and foreground RGBA masks using optical flow."
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--width", type=int, required=True, help="Processing width")
    parser.add_argument("--height", type=int, required=True, help="Processing height")
    parser.add_argument("--mask_method", choices=["raft", "farneback"], default="raft")
    parser.add_argument("--raft_model", help="Path to RAFT checkpoint (.pth)")
    parser.add_argument("--mask_threshold", type=float, default=1.5, help="Magnitude threshold for flow mask")
    parser.add_argument("--mask_kernel", type=int, default=5, help="Morphology kernel size")
    parser.add_argument("--inpaint_radius", type=int, default=3, help="Inpaint radius for background fill")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames to process")
    parser.add_argument("--iters", type=int, default=12, help="RAFT inference iterations")
    parser.add_argument("--raft_small", action="store_true", help="Use RAFT small architecture")
    parser.add_argument("--mixed_precision", action="store_true", help="Use RAFT mixed precision")
    parser.add_argument("--alternate_corr", action="store_true", help="Use RAFT alternate correlation")
    parser.add_argument("--save_foreground", action="store_true", help="Store RGBA foreground frames")
    parser.add_argument("--frame_interval", type=int, default=1, help="Process every Nth frame")
    args = parser.parse_args()

    output_dir = Path(args.output)
    bg_dir = output_dir / "background_frames"
    mask_dir = output_dir / "masks"
    fg_dir = output_dir / "foreground_rgba"

    for directory in (bg_dir, mask_dir, fg_dir):
        directory.mkdir(parents=True, exist_ok=True)

    segmenter = build_segmenter(args)

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


if __name__ == "__main__":
    main()
