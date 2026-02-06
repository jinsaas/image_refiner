# -------------------------------
# IR Lite — Transform Node
# (LOCALE-based multilingual description support included)
# -------------------------------

import numpy as np
import torch
from PIL import Image
import cv2

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

class IRL_Resize(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Resize",
            display_name="리사이즈",
            description="이미지를 지정된 너비와 높이로 리사이즈합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="리사이즈할 이미지"),
                IO.Int.Input("width", default=256, min=1, tooltip="출력 이미지의 너비"),
                IO.Int.Input("height", default=256, min=1, tooltip="출력 이미지의 높이"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="리사이즈된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, width, height) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        resized = pil_img.resize((width, height), Image.LANCZOS)
        return IO.NodeOutput(to_tensor_output(resized))

# -------------------------------

class IRL_Rotate(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Rotate",
            display_name="회전",
            description="이미지를 지정된 각도로 회전합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="회전할 이미지"),
                IO.Float.Input("angle", default=90.0, min=-360.0, max=360.0, step=1.0, tooltip="회전 각도 (도 단위)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="회전된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, angle) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        rotated = pil_img.rotate(angle, expand=True)
        return IO.NodeOutput(to_tensor_output(rotated))

# -------------------------------

class IRL_Flip(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Flip",
            display_name="플립",
            description="이미지를 수평 또는 수직으로 뒤집습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="뒤집을 이미지"),
                IO.Combo.Input("mode", options=["horizontal","vertical"], default="horizontal", tooltip="출력 모드 선택"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="뒤집힌 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, mode="horizontal") -> IO.NodeOutput:

        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)

        if mode == "horizontal":  # horizontal
            flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:          # vertical
            flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        return IO.NodeOutput(to_tensor_output(flipped))

# -------------------------------

class IRL_Crop(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Crop",
            display_name="크롭",
            description="이미지를 지정된 사각형 영역으로 크롭합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="크롭 처리할 이미지"),
                IO.Int.Input("x", default=0, min=0, tooltip="크롭 기준점 X 좌표"),
                IO.Int.Input("y", default=0, min=0, tooltip="크롭 기준점 Y 좌표"),
                IO.Int.Input("width", default=256, min=1, tooltip="크롭 영역의 너비"),
                IO.Int.Input("height", default=256, min=1, tooltip="크롭 영역의 높이"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="크롭된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, x, y, width, height) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        cropped = pil_img.crop((x, y, x + width, y + height))
        return IO.NodeOutput(to_tensor_output(cropped))
        
# -------------------------------

class IRL_CropMargins(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_CropMargins",
            display_name="크롭 마진",
            description="이미지 중앙을 기점으로 각 면의 모서리로부터 픽셀단위로 이미지 자르기를 수행합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="크롭 처리할 이미지"),
                IO.Int.Input("left", default=0, min=0, tooltip="왼쪽에서 자를 픽셀 수"),
                IO.Int.Input("right", default=0, min=0, tooltip="오른쪽에서 자를 픽셀 수"),
                IO.Int.Input("top", default=0, min=0, tooltip="위쪽에서 자를 픽셀 수"),
                IO.Int.Input("bottom", default=0, min=0, tooltip="아래쪽에서 자를 픽셀 수"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="크롭된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, left, right, top, bottom) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        w, h = pil_img.size

        # 잘라낼 영역 계산
        x1 = left
        y1 = top
        x2 = w - right
        y2 = h - bottom

        cropped = pil_img.crop((x1, y1, x2, y2))
        return IO.NodeOutput(to_tensor_output(cropped))

# -------------------------------

class IRL_PerspectiveWarp(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_PerspectiveWarp",
            display_name="퍼스펙티브 왜곡",
            description="원본과 대상 좌표를 사용하여 이미지에 원근 왜곡을 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="왜곡할 이미지"),
                # Source points
                IO.Int.Input("src_p1_x", default=0, tooltip="원본 P1 X"),
                IO.Int.Input("src_p1_y", default=0, tooltip="원본 P1 Y"),
                IO.Int.Input("src_p2_x", default=100, tooltip="원본 P2 X"),
                IO.Int.Input("src_p2_y", default=0, tooltip="원본 P2 Y"),
                IO.Int.Input("src_p3_x", default=0, tooltip="원본 P3 X"),
                IO.Int.Input("src_p3_y", default=100, tooltip="원본 P3 Y"),
                IO.Int.Input("src_p4_x", default=100, tooltip="원본 P4 X"),
                IO.Int.Input("src_p4_y", default=100, tooltip="원본 P4 Y"),
                # Destination points
                IO.Int.Input("dst_p1_x", default=0, tooltip="대상 P1 X"),
                IO.Int.Input("dst_p1_y", default=0, tooltip="대상 P1 Y"),
                IO.Int.Input("dst_p2_x", default=100, tooltip="대상 P2 X"),
                IO.Int.Input("dst_p2_y", default=0, tooltip="대상 P2 Y"),
                IO.Int.Input("dst_p3_x", default=0, tooltip="대상 P3 X"),
                IO.Int.Input("dst_p3_y", default=100, tooltip="대상 P3 Y"),
                IO.Int.Input("dst_p4_x", default=100, tooltip="대상 P4 X"),
                IO.Int.Input("dst_p4_y", default=100, tooltip="대상 P4 Y"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="퍼스펙티브 왜곡이 적용된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(
        cls,
        image,
        src_p1_x, src_p1_y,
        src_p2_x, src_p2_y,
        src_p3_x, src_p3_y,
        src_p4_x, src_p4_y,
        dst_p1_x, dst_p1_y,
        dst_p2_x, dst_p2_y,
        dst_p3_x, dst_p3_y,
        dst_p4_x, dst_p4_y
    ) -> IO.NodeOutput:
        arr = to_numpy_image(image)

        # Source and destination points
        src = np.float32([
            [src_p1_x, src_p1_y],
            [src_p2_x, src_p2_y],
            [src_p3_x, src_p3_y],
            [src_p4_x, src_p4_y]
        ])
        dst = np.float32([
            [dst_p1_x, dst_p1_y],
            [dst_p2_x, dst_p2_y],
            [dst_p3_x, dst_p3_y],
            [dst_p4_x, dst_p4_y]
        ])

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(arr, matrix, (arr.shape[1], arr.shape[0]))

        return IO.NodeOutput(to_tensor_output(Image.fromarray(warped)))
        
# -------------------------------

TRANSFORM_NODE_CLASS_MAPPINGS = {
    "IRL_Resize": IRL_Resize,
    "IRL_Rotate": IRL_Rotate,
    "IRL_Flip": IRL_Flip,
    "IRL_Crop": IRL_Crop,
    "IRL_CropMargins": IRL_CropMargins,
    "IRL_PerspectiveWarp": IRL_PerspectiveWarp,
}

TRANSFORM_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_Resize": "리사이즈",
    "IRL_Rotate": "회전",
    "IRL_Flip": "플립",
    "IRL_Crop": "크롭",
    "IRL_CropMargins": "크롭 마진",
    "IRL_PerspectiveWarp": "퍼스펙티브 왜곡",
}

# -------------------------------