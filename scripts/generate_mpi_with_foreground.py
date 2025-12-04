import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import argparse
import os
import sys
from types import SimpleNamespace
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from single_view_mpi.libs import mpi
from generate_full_atlas_video import load_model, process_frame


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def to_frame_name(idx):
    return f"frame_{idx:06d}"


class MotionSegmenter:
    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        raise NotImplementedError


class FarnebackSegmenter(MotionSegmenter):
    def __init__(self, threshold: float):
        self.threshold = threshold
        self.prev_gray = None

    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
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
        mask = (mag > self.threshold).astype(np.uint8) * 255
        return mask


class RaftSegmenter(MotionSegmenter):
    def __init__(
        self,
        model_path: str,
        threshold: float,
        iters: int = 12,
        small: bool = False,
        mixed_precision: bool = False,
        alternate_corr: bool = False,
    ):
        import torch

        raft_root = Path(__file__).parent.parent / "submodules" / "RAFT" / "core"
        sys.path.append(str(raft_root))

        from raft import RAFT
        from utils.utils import InputPadder

        self.torch = torch
        self.InputPadder = InputPadder
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        args = SimpleNamespace(
            small=small,
            mixed_precision=mixed_precision,
            alternate_corr=alternate_corr,
        )
        self.model = RAFT(args)
        state = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()
        self.threshold = threshold
        self.iters = iters

    def _to_tensor(self, frame_bgr: np.ndarray):
        tensor = self.torch.from_numpy(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float()
        return tensor[None].to(self.device)

    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> np.ndarray:
        with self.torch.no_grad():
            image1 = self._to_tensor(prev_bgr)
            image2 = self._to_tensor(curr_bgr)
            padder = self.InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)
            _, flow_up = self.model(image1, image2, iters=self.iters, test_mode=True)
            flow = padder.unpad(flow_up[0]).permute(1, 2, 0).cpu().numpy()
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            mask = (mag > self.threshold).astype(np.uint8) * 255
        return mask


def clean_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 0:
        return mask
    k = np.ones((kernel, kernel), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
    return closed


def inpaint_background(frame_bgr: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if mask.max() == 0:
        return frame_bgr.copy()
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(frame_bgr, dilated, float(radius), cv2.INPAINT_TELEA)


def save_foreground(frame_bgr: np.ndarray, mask: np.ndarray, path: str):
    fg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    rgba = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(path, rgba)


def main():
    parser = argparse.ArgumentParser(
        description="Split video into MPI background atlases and foreground RGBA frames."
    )
    parser.add_argument("--video", required=True, help="Input video file")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--width", type=int, required=True, help="Processing width")
    parser.add_argument("--height", type=int, required=True, help="Processing height")
    parser.add_argument("--mask_method", choices=["raft", "farneback"], default="raft")
    parser.add_argument("--raft_model", help="Path to RAFT checkpoint (.pth)")
    parser.add_argument("--mask_threshold", type=float, default=1.5, help="Flow magnitude threshold for masks")
    parser.add_argument("--mask_kernel", type=int, default=5, help="Morphology kernel size")
    parser.add_argument("--inpaint_radius", type=int, default=3, help="Radius parameter for cv2.inpaint")
    parser.add_argument("--max_frames", type=int, default=-1, help="Limit number of frames to process")
    parser.add_argument("--iters", type=int, default=12, help="RAFT inference iterations")
    parser.add_argument("--save_background_frames", action="store_true", help="Export inpainted background frames")
    parser.add_argument("--no_atlas", action="store_true", help="Skip atlas building and only export layers")
    args = parser.parse_args()

    bg_atlas_dir = os.path.join(args.output, "background_atlas")
    fg_dir = os.path.join(args.output, "foreground_rgba")
    mask_dir = os.path.join(args.output, "masks")
    bg_frame_dir = os.path.join(args.output, "background_frames")

    for d in (bg_atlas_dir, fg_dir, mask_dir):
        ensure_dir(d)
    if args.save_background_frames:
        ensure_dir(bg_frame_dir)

    if args.mask_method == "raft":
        if not args.raft_model:
            raise ValueError("--raft_model is required when mask_method=raft")
        segmenter = RaftSegmenter(
            model_path=args.raft_model,
            threshold=args.mask_threshold,
            iters=args.iters,
        )
    else:
        segmenter = FarnebackSegmenter(threshold=args.mask_threshold)

    model = load_model()
    depths = mpi.make_depths(1.0, 100.0, 32).numpy()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open video: {args.video}")

    ret, first_frame = cap.read()
    if not ret:
        raise RuntimeError("Video has no frames.")
    current_frame = cv2.resize(first_frame, (args.width, args.height))
    prev_for_mask = current_frame.copy()
    frame_idx = 0

    while True:
        if args.max_frames != -1 and frame_idx >= args.max_frames:
            break

        if frame_idx == 0:
            mask = np.zeros(current_frame.shape[:2], dtype=np.uint8)
        else:
            mask = segmenter.compute_mask(prev_for_mask, current_frame)
            mask = clean_mask(mask, args.mask_kernel)

        background = inpaint_background(current_frame, mask, args.inpaint_radius)
        if args.save_background_frames:
            cv2.imwrite(os.path.join(bg_frame_dir, f"{to_frame_name(frame_idx)}.png"), background)

        save_foreground(current_frame, mask, os.path.join(fg_dir, f"{to_frame_name(frame_idx)}.png"))
        cv2.imwrite(os.path.join(mask_dir, f"{to_frame_name(frame_idx)}.png"), mask)

        frame_rgb = cv2.cvtColor(background, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        process_frame(
            model=model,
            frame_rgb=frame_rgb,
            depths=depths,
            output_dir=bg_atlas_dir,
            frame_index=frame_idx,
            output_width=args.width,
            output_height=args.height,
            build_atlas_output=not args.no_atlas,
        )

        frame_idx += 1
        prev_for_mask = current_frame.copy()
        ret, next_frame = cap.read()
        if not ret:
            break
        current_frame = cv2.resize(next_frame, (args.width, args.height))

    cap.release()
    print(f"Finished processing {frame_idx} frames.")


if __name__ == "__main__":
    main()
