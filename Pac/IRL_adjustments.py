# -------------------------------
# IR Lite — Adjustments Nodes
# (LOCALE-based multilingual description support included)
# -------------------------------

import numpy as np
import torch
import cv2
from PIL import Image
from skimage import exposure

# ComfyUI 최신 API
from comfy_api.latest import IO, UI


# -------------------------------
# 공통 유틸 함수
def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]  # batch 차원 추가
    return torch.from_numpy(arr)

def to_numpy_image(image):
    if isinstance(image, torch.Tensor):
        arr = image[0].cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return arr
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        return image.astype(np.uint8)
    else:
        raise TypeError("Unsupported image type")


# -------------------------------
# Node definitions
# -------------------------------


class IRL_RGBLevels(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_RGBLevels",
            display_name="RGB 레벨",
            category="이미지 리파이너/이미지조정",
            description="각 RGB 채널의 강도 레벨을 조정합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Float.Input("r_level", default=1.00, min=0.00, max=3.00, step=0.01, tooltip="R 채널 강도"),
                IO.Float.Input("g_level", default=1.00, min=0.00, max=3.00, step=0.01, tooltip="G 채널 강도"),
                IO.Float.Input("b_level", default=1.00, min=0.00, max=3.00, step=0.01, tooltip="B 채널 강도")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="레벨 조정된 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, r_level=1.00, g_level=1.00, b_level=1.00) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)

        # 채널별 강도 조정
        arr[..., 0] = arr[..., 0] * r_level
        arr[..., 1] = arr[..., 1] * g_level
        arr[..., 2] = arr[..., 2] * b_level

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))

# -------------------------------


class IRL_BlackWhiteLevels(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_BlackWhiteLevels",
            display_name="블랙 & 화이트 레벨",
            category="이미지 리파이너/이미지조정",
            description="이미지의 전체 블랙/화이트 포인트를 조정합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Int.Input("black_point", default=0, min=0, max=255,
                             tooltip="블랙 포인트 (어두운 영역 기준)"),
                IO.Int.Input("white_point", default=255, min=0, max=255,
                             tooltip="화이트 포인트 (밝은 영역 기준)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="블랙/화이트 포인트 조정된 결과 이미지")
            ],
        )

    @classmethod
    def execute(cls, image, black_point=0, white_point=0) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        arr = exposure.rescale_intensity(arr, in_range=(black_point, white_point))
        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))

# -------------------------------


class IRL_LevelsAdjustment(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_LevelsAdjustment",
            display_name="레벨 조정",
            category="이미지 리파이너/이미지조정",
            description="입력/출력 포인트와 감마 보정을 통한 레벨 조정을 수행합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Float.Input("in_brightness", default=1.00, min=0.00, max=3.00, step=0.01,
                               tooltip="입력 밝기 배율"),
                IO.Float.Input("gamma", default=1.00, min=0.01, max=5.00, step=0.01,
                               tooltip="감마 보정 값"),
                IO.Float.Input("out_brightness", default=1.00, min=0.00, max=3.00, step=0.01,
                               tooltip="출력 밝기 배율")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="밝기 및 감마 조정된 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, in_brightness=1.00, gamma=1.00, out_brightness=1.00) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)

        # 입력 밝기 조정
        arr = arr * in_brightness

        # 감마 보정
        arr = exposure.adjust_gamma(arr, gamma)

        # 출력 밝기 조정
        arr = arr * out_brightness

        arr = np.clip(arr, 0, 255).astype(np.uint8)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))

# -------------------------------


class IRL_GradientMap(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_GradientMap",
            display_name="그라디언트 맵",
            category="이미지 리파이너/이미지조정",
            description="그레이스케일 값을 두 색상 사이의 그라디언트로 매핑합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="그레이스케일 값을 매핑할 대상 이미지"),
                IO.Float.Input("dark_strength", default=0.0, min=0.0, max=1.0, step=0.05,
                               tooltip="어두운 영역 강조 강도"),
                IO.Float.Input("light_strength", default=1.0, min=0.0, max=1.0, step=0.05,
                               tooltip="밝은 영역 강조 강도"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="그레이스케일을 지정된 색상 그라디언트로 매핑한 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, color_dark="#000000", color_light="#FFFFFF") -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)
        gray = np.mean(arr, axis=2) / 255.0

        c1 = np.array([0, 0, 0], dtype=np.float32)
        c2 = np.array([255, 255, 255], dtype=np.float32)

        mapped = (c1 * (1 - gray[..., None]) + c2 * gray[..., None]).astype(np.uint8)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(mapped)))

# -------------------------------


class IRL_ShadowsHighlights(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ShadowsHighlights",
            display_name="그림자 & 하이라이트",
            category="이미지 리파이너/이미지조정",
            description="LAB 색 공간에서 그림자와 하이라이트를 조정합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Float.Input("shadow_amount", default=0.50, min=0.00, max=5.00, step=0.01,
                               tooltip="그림자 영역 밝기 조정 강도"),
                IO.Float.Input("highlight_amount", default=0.50, min=0.00, max=5.00, step=0.01,
                               tooltip="하이라이트 영역 밝기 조정 강도"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="그림자/하이라이트가 조정된 결과 이미지")
            ],
        )

    @classmethod
    def execute(cls, image, shadow_amount=0.50, highlight_amount=0.50) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        if arr is None or arr.ndim != 3:
            raise ValueError("Invalid input image for ShadowsHighlights")

        lab = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)

        L = lab[..., 0] / 255.0
        shadow_mask = (L < 0.5).astype(np.float32)
        highlight_mask = (L >= 0.5).astype(np.float32)

        # 조정
        L = L + shadow_mask * shadow_amount * (0.5 - L)
        L = L - highlight_mask * highlight_amount * (L - 0.5)

        lab[..., 0] = np.clip(L * 255.0, 0, 255)

        out = cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2RGB)
        if out is None:
            raise RuntimeError("cv2.cvtColor failed in ShadowsHighlights")

        return IO.NodeOutput(to_tensor_output(Image.fromarray(out)))


       

# -------------------------------
# 클래스 매핑
ADJUSTMENTS_NODE_CLASS_MAPPINGS = {
    "IRL_RGBLevels": IRL_RGBLevels,
    "IRL_BlackWhiteLevels": IRL_BlackWhiteLevels,
    "IRL_LevelsAdjustment": IRL_LevelsAdjustment,
    "IRL_GradientMap": IRL_GradientMap,
    "IRL_ShadowsHighlights": IRL_ShadowsHighlights,
}

# 이름 매핑
ADJUSTMENTS_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_RGBLevels": "RGB 레벨",
    "IRL_BlackWhiteLevels": "블랙 & 화이트 레벨",
    "IRL_LevelsAdjustment": "레벨 조정",
    "IRL_GradientMap": "그라디언트 맵",
    "IRL_ShadowsHighlights": "그림자 & 하이라이트",
}

# -------------------------------