# -------------------------------
# IR Lite — Transform Node
# -------------------------------

import os
import sys
import numpy as np
import torch
from PIL import Image
import cv2

from comfy_api.latest import IO, UI
import comfy.utils
from comfy.utils import ProgressBar

# ---------------------------------------
# Header Utils
#----------------------------------------

def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]  # add batch
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

def image_to_vector(image_arr):
    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def resize_vector(contours, new_width, new_height, orig_width, orig_height):
    scale_x = new_width / orig_width
    scale_y = new_height / orig_height
    scaled_contours = []
    for cnt in contours:
        if cnt.shape[0] > 0:
            scaled = cnt.astype(np.float32) * [scale_x, scale_y]
            scaled = scaled.astype(np.int32)
            scaled_contours.append(scaled)
    return scaled_contours

def vector_to_image(contours, width, height, base_image=None, draw_color=(0,0,0), thickness=1, interpolation=cv2.INTER_LANCZOS4):
    if base_image is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    else:
        canvas = cv2.resize(base_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    cv2.drawContours(canvas, contours, -1, draw_color, thickness)
    return canvas

def progressbar_to_base(total_steps):
    from comfy.utils import ProgressBar
    return ProgressBar(int(total_steps))

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
                IO.Int.Input("width", default=256, min=64, max=2048, tooltip="출력 이미지의 너비"),
                IO.Int.Input("height", default=256, min=64, max=2048, tooltip="출력 이미지의 높이"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="리사이즈된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, width, height) -> IO.NodeOutput:
        total_steps = 3
        pbar = progressbar_to_base(total_steps)
        arr = to_numpy_image(image)
        pbar.update(1)
        pil_img = Image.fromarray(arr)

        width  = min(max(width, 64), 2048)
        height = min(max(height, 64), 2048)

        pbar.update(1)
        resized = pil_img.resize((width, height), Image.LANCZOS)
        pbar.update(1)
        canvas = to_tensor_output(resized)
        return IO.NodeOutput(canvas)
# -------------------------------

class IRL_VecterResize(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_VecterResize",
            display_name="벡터 리사이즈",
            description="이미지를 벡터처리해 지정된 너비와 높이로 리사이즈합니다.\n"
                        "고화질 일러스트나 사진일 경우 품질이 하락할 수 있습니다.\n"
                        "보간법마다 한계는 달라도 다운사이징의 경우 256 이하로 내리는건 추천하지 않습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="리사이즈할 이미지"),
                IO.Int.Input("width", default=256, min=64, max=2048, tooltip="출력 이미지의 너비"),
                IO.Int.Input("height", default=256, min=64, max=2048, tooltip="출력 이미지의 높이"),
                IO.Image.Input("sample_image", tooltip="컬러 보정용 참고 이미지", optional=True),
                IO.Combo.Input("method", default="Bicubic", options=["Lanczos","Bicubic","Nearest","PixelBox"], tooltip="보간용 처리법"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="리사이즈된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, width, height, sample_image=None,method="Bicubic") -> IO.NodeOutput:
        total_steps = 6
        pbar = progressbar_to_base(total_steps)
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        pbar.update(1)

        width  = min(max(width, 64), 2048)
        height = min(max(height, 64), 2048)


        # 1. pixel resizing
        interp_map = {"Bicubic": Image.BICUBIC,"Lanczos": Image.LANCZOS,"Nearest":Image.NEAREST,"PixelBox": Image.BOX,}
        resized = pil_img.resize((width, height), interp_map[method])
        canvas = np.array(resized)
        pbar.update(1)

        # 2. image to vector
        contours = image_to_vector(arr)
        pbar.update(1)

        # 3. vector resizing
        scaled_contours = resize_vector(contours, width, height, arr.shape[1], arr.shape[0])
        pbar.update(1)

        # 4. return image (pixel resize + Line&value resize)
        canvas = vector_to_image(scaled_contours, width, height, base_image=canvas, interpolation=interp_map[method])
        pbar.update(1)

        # 5. Reinhard Color Transfer
        if sample_image is not None:
            samp_arr = to_numpy_image(sample_image)
            arr_lab  = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB).astype(np.float32)
            samp_lab = cv2.cvtColor(samp_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            for i in range(3):
                arr_mean, arr_std   = arr_lab[:,:,i].mean(), arr_lab[:,:,i].std()
                samp_mean, samp_std = samp_lab[:,:,i].mean(), samp_lab[:,:,i].std()
                arr_lab[:,:,i] = (arr_lab[:,:,i] - arr_mean) * (samp_std / (arr_std+1e-5)) + samp_mean
            arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
            canvas = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2RGB)
        pbar.update(1)

        return IO.NodeOutput(to_tensor_output(canvas))
# -------------------------------

class IRL_Resize_Upsize_only(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Resize_Upsize_only",
            display_name="리사이즈(업사이즈 전용)",
            description="이미지를 지정된 너비와 높이로 리사이즈합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="리사이즈할 이미지"),
                IO.Int.Input("width", default=256, min=1, tooltip="출력 이미지의 너비"),
                IO.Int.Input("height", default=256, min=1, tooltip="출력 이미지의 높이"),
                IO.Combo.Input("method", default="Lanczos", options=["Lanczos","Bicubic","Nearest"]),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="리사이즈된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, width, height, method) -> IO.NodeOutput:
        total_steps = 3
        pbar = progressbar_to_base(total_steps)
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        pbar.update(1)
        w, h = pil_img.size

        target_w = min(max(width, w), w * 4)
        target_h = min(max(height, h), h * 4)

        pbar.update(1)
        interp_map = {"Lanczos":Image.LANCZOS,"Bicubic":Image.BICUBIC,"Nearest":Image.NEAREST}
        resized = pil_img.resize((width, height), interp_map[method])
        pbar.update(1)
        canvas=to_tensor_output(resized)
        return IO.NodeOutput(canvas)

# -------------------------------

class IRL_Resize_downsize_only(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Resize_downsize_only",
            display_name="리사이즈(다운사이즈 전용)",
            description="이미지를 지정된 너비와 높이로 리사이즈합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="리사이즈할 이미지"),
                IO.Int.Input("width", default=256, min=1, tooltip="출력 이미지의 너비"),
                IO.Int.Input("height", default=256, min=1, tooltip="출력 이미지의 높이"),
                IO.Combo.Input("method", default="Lanczos", options=["Bicubic","Lanczos","PixelBox"]),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="리사이즈된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, width, height, method) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        w, h = pil_img.size

        target_w = max(min(width, w), max(1, w // 4))
        target_h = max(min(height, h), max(1, h // 4))

        interp_map = {"Bicubic": Image.BICUBIC,"Lanczos": Image.LANCZOS,"PixelBox": Image.BOX,}
        resized = pil_img.resize((width, height), interp_map[method])
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
        w, h = pil_img.size

        x1 = min(max(x, 0), w)
        y1 = min(max(y, 0), h)
        x2 = min(max(x + width, 1), w)
        y2 = min(max(y + height, 1), h)
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


        x1 = min(max(left, 0), w)
        y1 = min(max(top, 0), h)
        x2 = min(max(w - right, 1), w)
        y2 = min(max(h - bottom, 1), h)

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
    def execute(cls, image, src_p1_x, src_p1_y, src_p2_x, src_p2_y, src_p3_x, src_p3_y, 
                src_p4_x, src_p4_y, dst_p1_x, dst_p1_y, dst_p2_x, dst_p2_y, dst_p3_x, dst_p3_y, 
                dst_p4_x, dst_p4_y) -> IO.NodeOutput:

        arr = to_numpy_image(image)

        # Source and destination points
        src = np.float32([
            [min(max(src_p1_x, 0), w), min(max(src_p1_y, 0), h)],
            [min(max(src_p2_x, 0), w), min(max(src_p2_y, 0), h)],
            [min(max(src_p3_x, 0), w), min(max(src_p3_y, 0), h)],
            [min(max(src_p4_x, 0), w), min(max(src_p4_y, 0), h)]
        ])
        dst = np.float32([
            [min(max(dst_p1_x, 0), w), min(max(dst_p1_y, 0), h)],
            [min(max(dst_p2_x, 0), w), min(max(dst_p2_y, 0), h)],
            [min(max(dst_p3_x, 0), w), min(max(dst_p3_y, 0), h)],
            [min(max(dst_p4_x, 0), w), min(max(dst_p4_y, 0), h)]
        ])

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(arr, matrix, (arr.shape[1], arr.shape[0]))

        return IO.NodeOutput(to_tensor_output(Image.fromarray(warped)))
        
# -------------------------------

TRANSFORM_NODE_CLASS_MAPPINGS = {
    "IRL_Resize": IRL_Resize,
    "IRL_VecterResize": IRL_VecterResize,
    "IRL_Resize_Upsize_only": IRL_Resize_Upsize_only,
    "IRL_Resize_downsize_only": IRL_Resize_downsize_only,
    "IRL_Rotate": IRL_Rotate,
    "IRL_Flip": IRL_Flip,
    "IRL_Crop": IRL_Crop,
    "IRL_CropMargins": IRL_CropMargins,
    "IRL_PerspectiveWarp": IRL_PerspectiveWarp,
}

TRANSFORM_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_Resize": "리사이즈",
    "IRL_VecterResize": "벡터 리사이즈",
    "IRL_Resize_Upsize_only": "리사이즈(업사이즈 전용)",
    "IRL_Resize_downsize_only": "리사이즈(다운사이즈 전용)",
    "IRL_Rotate": "회전",
    "IRL_Flip": "플립",
    "IRL_Crop": "크롭",
    "IRL_CropMargins": "크롭 마진",
    "IRL_PerspectiveWarp": "퍼스펙티브 왜곡",
}

# -------------------------------
