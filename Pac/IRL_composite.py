# -------------------------------
# IR Lite — Composite Nodes
# (LOCALE-based multilingual description support included)

# -------------------------------

import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image

# ComfyUI 최신 API
from comfy_api.latest import IO, UI

# -------------------------------
# 공통 유틸 함수

def to_tensor_output(tensor: torch.Tensor):
    """
    Torch 텐서를 그대로 반환 (batch 포함)
    """
    if tensor.ndim == 3:  # (H, W, C)
        tensor = tensor.unsqueeze(0)  # batch 차원 추가
    return tensor.float().clamp(0.0, 1.0)

def to_torch_image(image):
    """
    입력을 Torch float32 [0,1] 텐서로 변환
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
        

def gaussian_blur(tensor, kernel_size=5, sigma=2):
    # 간단한 Gaussian blur 구현
    import math
    k = kernel_size // 2
    x = torch.arange(-k, k+1, dtype=torch.float32)
    gauss = torch.exp(-(x**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    kernel1d = gauss.unsqueeze(0)
    kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
    kernel2d = kernel2d / kernel2d.sum()
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)

    tensor = tensor.unsqueeze(0).unsqueeze(0)
    blurred = F.conv2d(tensor, kernel2d, padding=k)
    return blurred.squeeze()


# -------------------------------

class IRL_ImageBlend(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageBlend",
            display_name="이미지 블렌드",
            description="두 이미지를 지정된 비율로 블렌딩합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("factor", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="이미지 B가 섞이는 비율 (0.0 = only A, 1.0 = only B)"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="블렌드 효과 강도 (0.0 = 원본 유지, 1.0 = 완전 적용)"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="블렌딩된 이미지"),
            ],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, factor, strength, priority_a, priority_b, mask=None, Mask_mode="normal") -> IO.NodeOutput:
        # Torch 기반 float 연산
        a = to_torch_image(image_a)  # [0,1] float tensor
        b = to_torch_image(image_b)
        
        
        if mask is not None:
            mask = to_torch_image(mask)
            
            # 배치 차원 맞추기
            if mask.ndim == 2:  # (H, W)
                mask = mask.unsqueeze(-1).expand_as(a[0])  # 채널 확장
                mask = mask.unsqueeze(0)                   # 배치 추가
            elif mask.ndim == 3:
                if mask.shape[-1] == 1:  # (H, W, 1)
                    mask = mask.expand_as(a)
                mask = mask.unsqueeze(0)     # 배치 추가
            elif mask.ndim == 4:
                if mask.shape[-1] == 1:   # (B, H, W, 1)
                    mask = mask.expand_as(a)
            else:
                raise ValueError("Unsupported mask dimensions")
                
            m = mask
            
            if m.shape[-1] == 1:
                if Mask_mode == "Small_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
                elif Mask_mode == "big_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
                elif Mask_mode == "blur":
                    m = gaussian_blur(m, kernel_size=5, sigma=2)
                    m = m.expand_as(a)
        else:
            m = torch.ones_like(a)

        # 기본 블렌드
        blended = (a * (1.0 - factor) + b * factor).clamp(0.0, 1.0)
        blended = blended * m + a * (1 - m)

        # 우선도 반영
        weighted = (a * priority_a + blended * priority_b) / (priority_a + priority_b)

        # 강도 조절 (원본 a와 blended 사이 보간)
        result = (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)

        return IO.NodeOutput(result)



# -------------------------------

class IRL_ImageOverlay(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageOverlay",
            display_name="이미지 오버레이",
            description="두 이미지를 오버레이 블렌드 모드로 합성합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="오버레이 효과 강도"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
            ],
            outputs=[IO.Image.Output("image", tooltip="오버레이 합성이 적용된 이미지")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, strength, priority_a, priority_b, mask=None, Mask_mode="normal") -> IO.NodeOutput:
        a = to_torch_image(image_a)
        b = to_torch_image(image_b)
        
        if mask is not None:
            mask = to_torch_image(mask)
            
            # 배치 차원 맞추기
            if mask.ndim == 2:  # (H, W)
                mask = mask.unsqueeze(-1).expand_as(a[0])  # 채널 확장
                mask = mask.unsqueeze(0)                   # 배치 추가
            elif mask.ndim == 3:
                if mask.shape[-1] == 1:  # (H, W, 1)
                    mask = mask.expand_as(a)
                mask = mask.unsqueeze(0)     # 배치 추가
            elif mask.ndim == 4:
                if mask.shape[-1] == 1:   # (B, H, W, 1)
                    mask = mask.expand_as(a)
            else:
                raise ValueError("Unsupported mask dimensions")
                
            m = mask
            
            if m.shape[-1] == 1:
                if Mask_mode == "Small_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
                elif Mask_mode == "big_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
                elif Mask_mode == "blur":
                    m = gaussian_blur(m, kernel_size=5, sigma=2)
                    m = m.expand_as(a)
        else:
            m = torch.ones_like(a)

        overlay = torch.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
        overlay = overlay * m + a * (1 - m)

        weighted = (a * priority_a + overlay * priority_b) / (priority_a + priority_b)
        result = (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)

        return IO.NodeOutput(result)

# -------------------------------

class IRL_ImageAdd(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageAdd",
            display_name="이미지 더하기",
            description="두 이미지를 더하고 클리핑합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Add 효과 강도"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
            ],
            outputs=[IO.Image.Output("image", tooltip="두 이미지가 더해진 결과")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, strength, priority_a, priority_b, mask=None, Mask_mode="normal") -> IO.NodeOutput:
        a = to_torch_image(image_a)
        b = to_torch_image(image_b)
        
        if mask is not None:
            mask = to_torch_image(mask)
            
            # 배치 차원 맞추기
            if mask.ndim == 2:  # (H, W)
                mask = mask.unsqueeze(-1).expand_as(a[0])  # 채널 확장
                mask = mask.unsqueeze(0)                   # 배치 추가
            elif mask.ndim == 3:
                if mask.shape[-1] == 1:  # (H, W, 1)
                    mask = mask.expand_as(a)
                mask = mask.unsqueeze(0)     # 배치 추가
            elif mask.ndim == 4:
                if mask.shape[-1] == 1:   # (B, H, W, 1)
                    mask = mask.expand_as(a)
            else:
                raise ValueError("Unsupported mask dimensions")
                
            m = mask
            
            if m.shape[-1] == 1:
                if Mask_mode == "Small_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
                elif Mask_mode == "big_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
                elif Mask_mode == "blur":
                    m = gaussian_blur(m, kernel_size=5, sigma=2)
                    m = m.expand_as(a)
        else:
            m = torch.ones_like(a)

        added = (a + b).clamp(0.0, 1.0)
        added = added * m + a * (1 - m)

        weighted = (a * priority_a + added * priority_b) / (priority_a + priority_b)
        result = (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)

        return IO.NodeOutput(result)

# -------------------------------

class IRL_ImageMultiply(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageMultiply",
            display_name="이미지 곱하기",
            description="두 이미지를 곱하여 합성합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Multiply 효과 강도"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
            ],
            outputs=[IO.Image.Output("image", tooltip="곱셈 합성이 적용된 이미지")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, strength, priority_a, priority_b, mask=None, Mask_mode="normal") -> IO.NodeOutput:
        a = to_torch_image(image_a)
        b = to_torch_image(image_b)
        
        if mask is not None:
            mask = to_torch_image(mask)
            
            # 배치 차원 맞추기
            if mask.ndim == 2:  # (H, W)
                mask = mask.unsqueeze(-1).expand_as(a[0])  # 채널 확장
                mask = mask.unsqueeze(0)                   # 배치 추가
            elif mask.ndim == 3:
                if mask.shape[-1] == 1:  # (H, W, 1)
                    mask = mask.expand_as(a)
                mask = mask.unsqueeze(0)     # 배치 추가
            elif mask.ndim == 4:
                if mask.shape[-1] == 1:   # (B, H, W, 1)
                    mask = mask.expand_as(a)
            else:
                raise ValueError("Unsupported mask dimensions")
                
            m = mask
            
            if m.shape[-1] == 1:
                if Mask_mode == "Small_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
                elif Mask_mode == "big_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
                elif Mask_mode == "blur":
                    m = gaussian_blur(m, kernel_size=5, sigma=2)
                    m = m.expand_as(a)
        else:
            m = torch.ones_like(a)

        multiplied = (a * b).clamp(0.0, 1.0)
        multiplied = multiplied * m + a * (1 - m)

        weighted = (a * priority_a + multiplied * priority_b) / (priority_a + priority_b)
        result = (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)

        return IO.NodeOutput(result)
        
# -------------------------------

class IRL_ImageDifference(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageDifference",
            display_name="이미지 차이",
            description="두 이미지의 픽셀 단위 절대 차이를 계산합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="Difference 효과 강도"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
            ],
            outputs=[IO.Image.Output("image", tooltip="두 이미지의 픽셀값 차이 결과")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, strength, priority_a, priority_b, mask=None, Mask_mode="normal") -> IO.NodeOutput:
        a = to_torch_image(image_a)
        b = to_torch_image(image_b)
        
        if mask is not None:
            mask = to_torch_image(mask)
            
            # 배치 차원 맞추기
            if mask.ndim == 2:  # (H, W)
                mask = mask.unsqueeze(-1).expand_as(a[0])  # 채널 확장
                mask = mask.unsqueeze(0)                   # 배치 추가
            elif mask.ndim == 3:
                if mask.shape[-1] == 1:  # (H, W, 1)
                    mask = mask.expand_as(a)
                mask = mask.unsqueeze(0)     # 배치 추가
            elif mask.ndim == 4:
                if mask.shape[-1] == 1:   # (B, H, W, 1)
                    mask = mask.expand_as(a)
            else:
                raise ValueError("Unsupported mask dimensions")
                
            m = mask
            
            if m.shape[-1] == 1:
                if Mask_mode == "Small_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 3, stride=1, padding=1).squeeze()
                elif Mask_mode == "big_spread":
                    m = F.max_pool2d(m.unsqueeze(0).unsqueeze(0), 5, stride=1, padding=2).squeeze()
                elif Mask_mode == "blur":
                    m = gaussian_blur(m, kernel_size=5, sigma=2)
                    m = m.expand_as(a)
        else:
            m = torch.ones_like(a)

        diff = torch.abs(a - b).clamp(0.0, 1.0)
        diff = diff * m + a * (1 - m)
        weighted = (a * priority_a + diff * priority_b) / (priority_a + priority_b)
        result = (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)

        return IO.NodeOutput(result)
        

# -------------------------------
COMPOSITE_NODE_CLASS_MAPPINGS = {
    "IRL_ImageBlend": IRL_ImageBlend,
    "IRL_ImageOverlay": IRL_ImageOverlay,
    "IRL_ImageAdd": IRL_ImageAdd,
    "IRL_ImageMultiply": IRL_ImageMultiply,
    "IRL_ImageDifference": IRL_ImageDifference,
}

COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_ImageBlend": "이미지 블렌드",
    "IRL_ImageOverlay": "이미지 오버레이",
    "IRL_ImageAdd": "이미지 더하기",
    "IRL_ImageMultiply": "이미지 곱하기",
    "IRL_ImageDifference": "이미지 차이",
}


# -------------------------------