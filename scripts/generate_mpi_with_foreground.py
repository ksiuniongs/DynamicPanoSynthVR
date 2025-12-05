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


class LaMaInpainter:
    """
    Wrapper around https://github.com/advimman/lama.git for background inpainting.
    """

    def __init__(self, repo_path: str, config_path: str, checkpoint_path: str, device: str = "auto"):
        repo_dir = Path(repo_path).expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"LaMa repo not found at {repo_dir}")
        if str(repo_dir) not in sys.path:
            sys.path.append(str(repo_dir))

        try:
            import torch
            import torch.nn.functional as F
            from omegaconf import OmegaConf
            from saicinpainting.training.trainers import load_checkpoint
        except ImportError as exc:
            raise ImportError(
                "Missing LaMa dependencies. Please install torch, omegaconf and "
                "the saicinpainting package from https://github.com/advimman/lama."
            ) from exc

        self.torch = torch
        self.F = F
        self.OmegaConf = OmegaConf
        self.load_checkpoint = load_checkpoint

        cfg_path = Path(config_path).expanduser().resolve()
        ckpt_path = Path(checkpoint_path).expanduser().resolve()
        if not cfg_path.exists():
            raise FileNotFoundError(f"LaMa config not found: {cfg_path}")
        if not ckpt_path.exists():
            raise FileNotFoundError(f"LaMa checkpoint not found: {ckpt_path}")

        torch_device = device
        if torch_device == "auto":
            torch_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(torch_device)

        config = OmegaConf.load(str(cfg_path))
        if hasattr(config, "training_model"):
            config.training_model.predict_only = True
        if hasattr(config, "visualizer"):
            config.visualizer.kind = "noop"

        self.model = load_checkpoint(config, str(ckpt_path), strict=False, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        pad_mod = 8
        data_cfg = getattr(config, "data", None)
        if data_cfg is not None and hasattr(data_cfg, "pad_out_to_modulo"):
            pad_mod = int(data_cfg.pad_out_to_modulo)
        self.pad_mod = pad_mod

    def _pad_tensor(self, tensor, mode="reflect"):
        _, _, h, w = tensor.shape
        pad_h = (self.pad_mod - h % self.pad_mod) % self.pad_mod
        pad_w = (self.pad_mod - w % self.pad_mod) % self.pad_mod
        if pad_h == 0 and pad_w == 0:
            return tensor
        pad = (0, pad_w, 0, pad_h)
        if mode == "constant":
            return self.F.pad(tensor, pad, mode="constant", value=0.0)
        return self.F.pad(tensor, pad, mode=mode)

    def __call__(self, frame_bgr: np.ndarray, mask: np.ndarray) -> np.ndarray:
        if mask is None or mask.max() == 0:
            return frame_bgr.copy()

        image_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        mask_float = (mask > 0).astype(np.float32)

        image_rgb = image_rgb.copy()
        image_rgb[mask_float > 0] = 0.0

        tensor = self.torch.from_numpy(image_rgb).permute(2, 0, 1).unsqueeze(0)
        tensor = tensor * 2.0 - 1.0
        mask_tensor = self.torch.from_numpy(mask_float).unsqueeze(0).unsqueeze(0)

        tensor = self._pad_tensor(tensor, mode="reflect")
        mask_tensor = self._pad_tensor(mask_tensor, mode="constant")

        batch = {
            "image": tensor.to(self.device),
            "mask": mask_tensor.to(self.device),
        }

        with self.torch.no_grad():
            batch = self.model(batch)

        prediction = batch.get("inpainted") or batch.get("predicted_image") or batch.get("image")
        if prediction is None:
            raise RuntimeError("LaMa model did not return an inpainted tensor.")

        prediction = prediction.detach().cpu()
        orig_h, orig_w = frame_bgr.shape[:2]
        prediction = prediction[:, :, :orig_h, :orig_w]
        prediction = prediction[0].permute(1, 2, 0).numpy()
        prediction = ((prediction + 1.0) * 0.5).clip(0.0, 1.0)
        prediction = (prediction * 255.0).astype(np.uint8)
        return cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)


def clean_mask(mask: np.ndarray, kernel: int) -> np.ndarray:
    if kernel <= 0:
        return mask
    k = np.ones((kernel, kernel), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, k)
    return closed


def inpaint_background(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    radius: int,
    method: str = "opencv",
    lama_inpainter=None,
) -> np.ndarray:
    if mask.max() == 0:
        return frame_bgr.copy()

    if method == "lama":
        if lama_inpainter is None:
            raise ValueError("LaMa inpainting requested but no LaMaInpainter instance was provided.")
        return lama_inpainter(frame_bgr, mask)

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
    parser.add_argument(
        "--inpaint_method",
        choices=["opencv", "lama"],
        default="opencv",
        help="Backend for background inpainting (OpenCV Telea or LaMa).",
    )
    parser.add_argument(
        "--lama_repo",
        default=os.path.join("submodules", "lama"),
        help="Path to the local clone of https://github.com/advimman/lama",
    )
    parser.add_argument(
        "--lama_config",
        default=os.path.join("submodules", "lama", "configs", "prediction", "default.yaml"),
        help="Path to the LaMa prediction config (.yaml).",
    )
    parser.add_argument(
        "--lama_ckpt",
        help="Path to the LaMa checkpoint (.ckpt or .safetensors) downloaded from the official repo.",
    )
    parser.add_argument(
        "--lama_device",
        default="auto",
        help="Device for LaMa inference: 'auto', 'cpu', 'cuda', or explicit cuda:N.",
    )
    args = parser.parse_args()

    if args.inpaint_method == "lama" and not args.lama_ckpt:
        parser.error("--lama_ckpt is required when inpaint_method=lama")

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

    lama_inpainter = None
    if args.inpaint_method == "lama":
        lama_inpainter = LaMaInpainter(
            repo_path=args.lama_repo,
            config_path=args.lama_config,
            checkpoint_path=args.lama_ckpt,
            device=args.lama_device,
        )

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

        background = inpaint_background(
            current_frame,
            mask,
            args.inpaint_radius,
            method=args.inpaint_method,
            lama_inpainter=lama_inpainter,
        )
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
