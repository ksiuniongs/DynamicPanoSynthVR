
import sys
import cv2
import numpy as np
from pathlib import Path
from typing import Tuple
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

def clean_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    if kernel_size <= 0:
        return mask
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    closed = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
    return closed

def inpaint_background(frame_bgr: np.ndarray, mask: np.ndarray, radius: int) -> np.ndarray:
    if mask.max() == 0:
        return frame_bgr.copy()
    dilated = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=1)
    return cv2.inpaint(frame_bgr, dilated, float(radius), cv2.INPAINT_TELEA)

def save_foreground(frame_bgr: np.ndarray, mask: np.ndarray, path: Path):
    fg = cv2.bitwise_and(frame_bgr, frame_bgr, mask=mask)
    rgba = cv2.cvtColor(fg, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = mask
    cv2.imwrite(str(path), rgba)
