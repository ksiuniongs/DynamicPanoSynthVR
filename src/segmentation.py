
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import torch

# Add RAFT to path
PROJECT_ROOT = Path(__file__).parent.parent
RAFT_CORE = PROJECT_ROOT / "submodules" / "RAFT" / "core"
sys.path.append(str(RAFT_CORE))

try:
    from raft import RAFT
    from utils.utils import InputPadder
except ImportError:
    print("Warning: Could not import RAFT. Ensure submodules are initialized.")

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
        
        # Load weights
        if isinstance(model_path, str):
            model_path = Path(model_path)
            
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        
        # Handle 'module.' prefix if present
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
            mask = (mag > self.threshold).astype(np.uint8) * 255
        return mask, mag.max()


class BackgroundSubtractionSegmenter(MotionSegmenter):
    def __init__(self, background_bgr: np.ndarray, threshold: float):
        self.background = background_bgr
        self.threshold = threshold

    def compute_mask(self, prev_bgr: np.ndarray, curr_bgr: np.ndarray) -> Tuple[np.ndarray, float]:
        # Ensure background matches frame size
        if self.background.shape != prev_bgr.shape:
            self.background = cv2.resize(self.background, (prev_bgr.shape[1], prev_bgr.shape[0]))
            
        diff = cv2.absdiff(prev_bgr, self.background)
        gray_diff = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        gray_diff = cv2.GaussianBlur(gray_diff, (5, 5), 0)
        
        # Thresholding
        _, mask = cv2.threshold(gray_diff, self.threshold, 255, cv2.THRESH_BINARY)
        
        # Optional: Morphological operations to fill holes
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        return mask, float(gray_diff.max())


class LaMaInpainter:
    """
    Wrapper around https://github.com/advimman/lama.git for higher quality background inpainting.
    """

    def __init__(
        self,
        repo_path: str,
        config_path: str,
        checkpoint_path: str,
        device: str = "auto",
    ):
        repo_dir = Path(repo_path).expanduser().resolve()
        if not repo_dir.exists():
            raise FileNotFoundError(f"LaMa repo not found at {repo_dir}")
        if str(repo_dir) not in sys.path:
            sys.path.append(str(repo_dir))

        try:
            import torch as _torch
            import torch.nn.functional as F
            from omegaconf import OmegaConf
            from saicinpainting.training.trainers import load_checkpoint
        except ImportError as exc:
            raise ImportError(
                "Unable to import LaMa dependencies. Please install the requirements "
                "from https://github.com/advimman/lama."
            ) from exc

        self.torch = _torch
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
            torch_device = "cuda" if _torch.cuda.is_available() else "cpu"
        self.device = _torch.device(torch_device)

        cfg = OmegaConf.load(str(cfg_path))
        if hasattr(cfg, "training_model"):
            cfg.training_model.predict_only = True
        if hasattr(cfg, "visualizer"):
            cfg.visualizer.kind = "noop"

        self.model = load_checkpoint(cfg, str(ckpt_path), strict=False, map_location=self.device)
        self.model.to(self.device)
        self.model.eval()

        pad_mod = 8
        data_cfg = getattr(cfg, "data", None)
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
        orig_h, orig_w = mask.shape[:2]
        prediction = prediction[:, :, :orig_h, :orig_w]
        prediction = prediction[0].permute(1, 2, 0).numpy()
        prediction = ((prediction + 1.0) * 0.5).clip(0.0, 1.0)
        prediction = (prediction * 255.0).astype(np.uint8)
        return cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    # Erode slightly to remove edge artifacts (black borders)
    eroded = cv2.erode(closed, kernel, iterations=1)
    return eroded

def inpaint_background(
    frame_bgr: np.ndarray,
    mask: np.ndarray,
    radius: int,
    method: str = "opencv",
    lama_inpainter: Optional[LaMaInpainter] = None,
) -> np.ndarray:
    if mask.max() == 0:
        return frame_bgr.copy()

    if method == "lama":
        if lama_inpainter is None:
            raise ValueError("LaMa inpainting selected but no LaMaInpainter instance was provided.")
        return lama_inpainter(frame_bgr, mask)

    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(frame_bgr, dilated, float(radius), cv2.INPAINT_TELEA)

def save_foreground(frame_bgr: np.ndarray, mask: np.ndarray, path: Path):
    # Ensure mask is single channel
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
        
    # Normalize alpha to 0-1
    alpha = mask.astype(np.float32) / 255.0
    
    # Convert frame to float
    frame_float = frame_bgr.astype(np.float32)
    
    # Premultiply RGB by Alpha
    # This darkens the RGB values based on transparency
    premultiplied = frame_float * alpha[:, :, np.newaxis]
    
    # Combine into BGRA
    bgra = np.dstack((premultiplied, mask)).astype(np.uint8)
    
    cv2.imwrite(str(path), bgra)
