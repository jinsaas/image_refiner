# -------------------------------
# IR Lite — Composite Nodes
# -------------------------------

import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
import cv2

from comfy_api.latest import IO, UI

# ---------------------------------------
# Header Utils
#----------------------------------------

def to_tensor_output(tensor: torch.Tensor):
    """
    Torch tensor return(add batch)
    """
    if tensor.ndim == 3:  # (H, W, C)
        tensor = tensor.unsqueeze(0)
    return tensor.float().clamp(0.0, 1.0)

def to_torch_image(image):
    """
    input-> Torch float32 [0,1] tensor
    """
    if isinstance(image, torch.Tensor):
        return image.float().clamp(0.0, 1.0)
    elif isinstance(image, np.ndarray):
        arr = torch.from_numpy(image).float()
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr.unsqueeze(0) if arr.ndim == 3 else arr
    elif isinstance(image, Image.Image):
        arr = torch.from_numpy(np.array(image.convert("RGB"))).float() / 255.0
        return arr.unsqueeze(0)
    else:
        raise TypeError("Unsupported image type")

def torch_to_numpy(image_t: torch.Tensor) -> np.ndarray:
    """
    Torch tensor -> numpy (H,W,3) uint8
    Supports (B,C,H,W) or (B,H,W,C)
    """
    if image_t.ndim == 4:
        image_t = image_t[0]

    if image_t.shape[0] in [1,3]:  
        # (C,H,W)
        image_np = image_t.permute(1, 2, 0).cpu().numpy()
    else:
        # (H,W,C)
        image_np = image_t.cpu().numpy()

    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    return image_np


def numpy_to_torch(image_np: np.ndarray) -> torch.Tensor:
    """
    numpy (H,W,3) or (H,W,1) or (B,H,W,3/1) uint8 -> Torch tensor (B,H,W,C) [0,1]
    """
    arr = torch.from_numpy(image_np).float() / 255.0

    if arr.ndim == 3:
        # (H,W,C) → (1,H,W,C)
        arr = arr.unsqueeze(0)
    elif arr.ndim == 4:
        # (B,H,W,C)
        pass
    else:
        raise ValueError(f"Unexpected shape for image_np: {arr.shape}")

    return arr

    
    
def gaussian_blur(tensor, kernel_size=5, sigma=2):
    # tensor: (B,1,H,W)
    k = kernel_size // 2
    x = torch.arange(-k, k+1, dtype=torch.float32)
    gauss = torch.exp(-(x**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
    kernel2d = kernel2d / kernel2d.sum()
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

    blurred = F.conv2d(tensor, kernel2d, padding=k)
    return blurred  # (B,1,H,W)


def ensure_mask_tensor(t: torch.Tensor) -> torch.Tensor:
    if not isinstance(t, torch.Tensor):
        t = torch.from_numpy(np.array(t)).float()
    if t.dim() == 2:
        t = t.unsqueeze(0).unsqueeze(0)
    elif t.dim() == 3:
        t = t.unsqueeze(1)
    elif t.dim() == 4:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {t.shape}")
    return t.float()

def process_mask(mask, a, Mask_mode="normal"):
    m = ensure_mask_tensor(mask)  # (B,1,H,W)

    if Mask_mode == "Small_spread":
        m = F.max_pool2d(m, 3, stride=1, padding=1)
    elif Mask_mode == "big_spread":
        m = F.max_pool2d(m, 5, stride=1, padding=2)
    elif Mask_mode == "blur":
        m = gaussian_blur(m, kernel_size=5, sigma=2)
        m = m.expand_as(a)

    # (B,1,H,W) → (B,H,W,1)
    m = m.permute(0, 2, 3, 1)

    # (B,H,W,1) → (B,H,W,3)
    m = m.expand(-1, -1, -1, a.shape[-1])

    return m

def apply_blend_mode(a, b, mode="Blend", factor=0.5):
    if mode == "Blend":
        return (a * (1.0 - factor) + b * factor).clamp(0.0, 1.0)
    elif mode == "Overlay":
        return torch.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    elif mode == "Add":
        return (a + b).clamp(0.0, 1.0)
    elif mode == "Multiply":
        return (a * b).clamp(0.0, 1.0)
    elif mode == "Difference":
        return torch.abs(a - b)
    else:
        return a

def apply_weighted(a, blended, priority_a, priority_b, strength):
    weighted = (a * priority_a + blended * priority_b) / (priority_a + priority_b)
    return (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)

# -------------------------------

class IRL_Imagecomposite(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Imagecomposite",
            display_name="이미지 합성(통합)",
            description="두 이미지를 선택한 방식에 맞게 처리합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("factor", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend 모드 처리 비율. 수치를 손대지 않을경우 무시됩니다."),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="블렌드 효과 강도"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("blend_mode", options=["Blend", "Overlay", "Add", "Multiply", "Difference"], default="Blend", tooltip="합성 방식"),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
                IO.Float.Input("saturation", default=0.00, min=-1.00, max=1.00, step=0.01,
                               tooltip="마스크 영역 채도 조정"),
                IO.Float.Input("out_satur", default=0.00, min=-1.00, max=1.00, step=0.01,
                               tooltip="마스크 반전 영역 채도 조정")
            ],
            outputs=[IO.Image.Output("image", tooltip="두 이미지의 픽셀값 차이 결과")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, factor=0.5, strength=1.0, priority_a=1.0, priority_b=1.0, mask=None, blend_mode="Blend", Mask_mode="normal", saturation=0.00, out_satur=0.00) -> IO.NodeOutput:
        factor = max(0.00, min(1.00, factor))
        strength = max(0.00, min(1.00, strength))
        priority_a = max(0.00, min(3.00, priority_a))
        priority_b = max(0.00, min(3.00, priority_b))
        saturation = max(-1.00, min(1.00, saturation))
        a = to_torch_image(image_a)
        b = to_torch_image(image_b)

        if mask is not None:
            m = process_mask(mask, a, Mask_mode)
        else:
            m = torch.ones_like(a)

        blended = apply_blend_mode(a, b, blend_mode, factor)
        blended = blended * m + a * (1 - m)
        
        result = apply_weighted(a, blended, priority_a, priority_b, strength)

        if saturation != 0.0:
            blended_np = torch_to_numpy(blended)
            hsv = cv2.cvtColor(blended_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            s = s * (1.0 + saturation)
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
            blended_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            blended_sat_t = numpy_to_torch(blended_sat)

            result = result * (1 - m) + blended_sat_t * m

        if out_satur != 0.0:
            blended_np = torch_to_numpy(blended)
            hsv = cv2.cvtColor(blended_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            s = s * (1.0 + out_satur)
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
            blended_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            blended_sat_t = numpy_to_torch(blended_sat)

            result = result * m + blended_sat_t * (1 - m)

        return IO.NodeOutput(result)
    
# -------------------------------
COMPOSITE_NODE_CLASS_MAPPINGS = {
    "IRL_Imagecomposite": IRL_Imagecomposite,
}

COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_Imagecomposite": "이미지 합성(통합)",
}


# -------------------------------
