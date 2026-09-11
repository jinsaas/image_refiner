# -------------------------------
# IR Lite — Transform Node
# -------------------------------

import os
import sys
import math
import numpy as np
import torch
from PIL import Image
import cv2
import re
import scipy
from scipy.interpolate import griddata, Rbf

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


def ensure_image_tensor(arr):
    if not isinstance(arr, torch.Tensor):
        arr = torch.from_numpy(np.array(arr)).float()

    if arr.dim() == 2:
        arr = arr.unsqueeze(0).unsqueeze(0)

    elif arr.dim() == 3:
        if arr.shape[-1] in (1,3,4):
            arr = arr.permute(2,0,1).unsqueeze(0)
        else:
            arr = arr.unsqueeze(0)

    elif arr.dim() == 4:
        if arr.shape[-1] in (1,3,4):
            arr = arr.permute(0,3,1,2)

    else:
        raise ValueError(f"Unsupported image shape: {arr.shape}")


    return arr.float()

def image_to_vector(image_arr, threshold=127):
    gray = cv2.cvtColor(image_arr, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

def vector_to_image(contours, width, height, base_image=None, draw_color=(0,0,0), thickness=1, interpolation=cv2.INTER_LANCZOS4, lineType=cv2.LINE_AA, contour_blur=None):
    if base_image is None:
        canvas = np.zeros((height, width, 3), dtype=np.uint8)
    elif base_image.shape[:2] == (height, width):
        canvas = base_image.copy()          # If the size already matches, skip resizing
    else:
        canvas = cv2.resize(base_image, (width, height), interpolation=cv2.INTER_LANCZOS4)
    valid_contours = [cnt for cnt in contours if cnt is not None and len(cnt) > 0]
    if valid_contours:
        cv2.drawContours(canvas, valid_contours, -1, draw_color, thickness, lineType=lineType)
    else:
        print(f"Since contour traversal failed, a normal resized image is returned.")
        pass

    if contour_blur:
            canvas = cv2.GaussianBlur(canvas, (3, 3), 0.5)

    return canvas

def progressbar_to_base(total_steps):
    from comfy.utils import ProgressBar
    return ProgressBar(int(total_steps))

# -------------------------------



def parse_points(points_str: str):
    """
    String example: "P1{0.0,0.0}, P2{0.5,0.0}, P3{1.0,0.0}, ..."
    Return: dict { "P1": np.array([x,y]), ... }
    """
    pattern = r"(P\d+)\{([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\}"
    matches = re.findall(pattern, points_str)

    result = {}
    for label, x, y in matches:
        result[label] = np.array([float(x), float(y)], dtype=np.float32)
    return result
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
                IO.Int.Input("width", default=256, min=64, max=4096, tooltip="출력 이미지의 너비"),
                IO.Int.Input("height", default=256, min=64, max=4096, tooltip="출력 이미지의 높이"),
                IO.Image.Input("sample_image", tooltip="컬러 보정용 참고 이미지", optional=True),
                IO.Combo.Input("method", default="Bicubic", options=["Lanczos","Bicubic","Nearest","PixelBox"], tooltip="보간용 처리법"),
                IO.Combo.Input("lineType", default="LINE_AA", options=["LINE_AA", "LINE_4", "LINE_8"], tooltip="컨투어 라인 처리법"),
                IO.Int.Input("contour_threshold", default=127, min=0, max=255, step=1, tooltip="컨투어 추출을 위한 임계값 (높을수록 진한 선만 추출)"),
                IO.Boolean.Input("contour_blur", default=True, tooltip="컨투어 라인에 약한 블러를 겁니다."),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="리사이즈된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, width, height, sample_image=None,method="Bicubic", lineType="LINE_AA", contour_threshold=127, contour_blur=True) -> IO.NodeOutput:
        total_steps = 6
        pbar = progressbar_to_base(total_steps)
        arr = to_numpy_image(image)
        pbar.update(1)

        width  = min(max(width, 64), 4096)
        height = min(max(height, 64), 4096)


        cv_interp_map = {"Bicubic": cv2.INTER_CUBIC, "Lanczos": cv2.INTER_LANCZOS4, "Nearest": cv2.INTER_NEAREST, "PixelBox": cv2.INTER_AREA}
        # 1. pixel resizing
        canvas = cv2.resize(arr, (width, height), interpolation=cv_interp_map[method])
        pbar.update(1)

        # 2. image to vector
        contours = image_to_vector(arr, threshold=contour_threshold)
        pbar.update(1)

        # 3. vector resizing
        scaled_contours = resize_vector(contours, width, height, arr.shape[1], arr.shape[0])
        pbar.update(1)

        # 4. lineType Mappings
        cv_line_type_map = {"LINE_AA": cv2.LINE_AA, "LINE_4": cv2.LINE_4, "LINE_8": cv2.LINE_8}
        selected_line_type = cv_line_type_map.get(lineType, cv2.LINE_AA)

        # 5. return image (pixel resize + Line&value resize)

        canvas = vector_to_image(scaled_contours, width, height, base_image=canvas, interpolation=cv_interp_map[method], lineType=selected_line_type, contour_blur=contour_blur)
        pbar.update(1)

        # 6. Reinhard Color Transfer
        if sample_image is not None:
            try:
                samp_arr = to_numpy_image(sample_image)
                arr_lab  = cv2.cvtColor(canvas, cv2.COLOR_RGB2LAB).astype(np.float32)
                samp_lab = cv2.cvtColor(samp_arr, cv2.COLOR_RGB2LAB).astype(np.float32)
                for i in range(3):
                    arr_mean, arr_std   = arr_lab[:,:,i].mean(), arr_lab[:,:,i].std()
                    samp_mean, samp_std = samp_lab[:,:,i].mean(), samp_lab[:,:,i].std()
                    arr_lab[:,:,i] = (arr_lab[:,:,i] - arr_mean) * (samp_std / (arr_std+1e-5)) + samp_mean
                arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
                canvas = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2RGB)
            except Exception:
                pass
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
                IO.Float.Input("angle", default=90.0, min=-360.0, max=360.0, step=1.0, tooltip="회전 각도 (도 단위). 음수로 입력시 반대로 회전합니다."),
                IO.String.Input("pad_color", default="128, 128, 128", tooltip="RGB 값을 콤마로 구분하여 입력합니다 (예: 128, 128, 128). 비워둘 경우 중립 회색으로 처리됩니다."),
                IO.Boolean.Input("preview_mode", default=False, tooltip="자른 결과를 노드에서 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="회전된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, angle, pad_color="128, 128, 128", preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        fill_color = (128, 128, 128)
        try:
            if pad_color and pad_color.strip():
                parts = [int(p.strip()) for p in pad_color.split(',')]
                if len(parts) == 3:
                    fill_color = tuple(parts)
                elif len(parts) == 1:
                    val = parts[0]
                    fill_color = (val, val, val)
        except Exception:
            # Maintains a safe neutral gray state upon parsing failure
            fill_color = (128, 128, 128)

        # Use 'expand=True' to prevent cropping when rotating, and use 'fillcolor' to handle empty corners
        rotated = pil_img.rotate(angle, expand=True, fillcolor=fill_color)

        output = to_tensor_output(rotated)
        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))
        return IO.NodeOutput(output)

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
                IO.Combo.Input("mode", options=["horizontal","vertical"], default="horizontal", tooltip="출력 모드 선택(horizontal:가로방향, vertical:세로방향)"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="자른 결과를 노드에서 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="뒤집힌 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, mode="horizontal", preview_mode=False) -> IO.NodeOutput:

        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)

        if mode == "horizontal":  # horizontal
            flipped = pil_img.transpose(Image.FLIP_LEFT_RIGHT)
        else:          # vertical
            flipped = pil_img.transpose(Image.FLIP_TOP_BOTTOM)

        output = to_tensor_output(flipped)
        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))
        return IO.NodeOutput(output)

# -------------------------------

class IRL_Guidance_Crop(IO.ComfyNode):
    
    @classmethod
    def imagetocv(cls, image) -> np.ndarray:
        if isinstance(image, torch.Tensor):
            arr = image.cpu().numpy()
        else:
            arr = np.array(image)

        # 4dim [B, H, W, C] or [B, C, H, W] -> [H, W, C]
        if arr.ndim == 4:
            arr = arr[0]
            if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0))  # [C, H, W] -> [H, W, C]
        
        # 3dim -> check to [C, H, W] or [H, W, C]
        elif arr.ndim == 3:
            if arr.shape[0] <= 4 and arr.shape[0] < arr.shape[1]:
                arr = np.transpose(arr, (1, 2, 0)) #[H, W, C]
        
        # 2dim [H, W]) -> [H, W, 1]
        elif arr.ndim == 2:
            arr = np.expand_dims(arr, axis=-1)

        # C = 1 -> C = 3
        if arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)
        elif arr.shape[-1] > 3:
            arr = arr[..., :3]  # del alpha

        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255).astype(np.uint8)
            else:
                arr = arr.astype(np.uint8)

        return arr

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Guidance_Crop",
            display_name="가이던스 크롭",
            description="이미지를 지정된 사각형 영역으로 크롭합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="크롭 처리할 이미지"),
                IO.Int.Input("x", default=0, min=0, tooltip="크롭 기준점 X 좌표"),
                IO.Int.Input("y", default=0, min=0, tooltip="크롭 기준점 Y 좌표"),
                IO.Int.Input("width", default=256, min=1, tooltip="크롭 영역의 너비"),
                IO.Int.Input("height", default=256, min=1, tooltip="크롭 영역의 높이"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="크롭될 부분을 노드에서 미리보기. 기준선이 보이지 않는 경우 에러가 발생할 위험이 있습니다."),
                IO.Combo.Input("guide_color", options=["red", "blue", "green", "yellow", "black"], default="red", tooltip="자른 결과를 노드에서 미리보기"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="크롭된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, x, y, width, height, preview_mode=False, guide_color = "red") -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        w, h = pil_img.size

        x1 = min(max(x, 0), w)
        y1 = min(max(y, 0), h)
        x2 = min(max(x + width, 1), w)
        y2 = min(max(y + height, 1), h)
        cropped = pil_img.crop((x1, y1, x2, y2))
        output = to_tensor_output(cropped)
        if preview_mode:
            canvas_cv = cls.imagetocv(arr)
            if canvas_cv.dtype != np.uint8:
                canvas_cv = (canvas_cv * 255).astype(np.uint8)
            canvas_bgr = cv2.cvtColor(canvas_cv, cv2.COLOR_RGB2BGR)
            if canvas_bgr.dtype != np.uint8:
                canvas_bgr = (canvas_bgr * 255).astype(np.uint8)
            
            color_map = {
                "red": (0, 0, 255),      # OpenCV Red (BGR)
                "blue": (255, 0, 0),     # OpenCV Blue (BGR)
                "green": (0, 255, 0),    # OpenCV Green (BGR)
                "yellow": (0, 255, 255), # OpenCV Yellow (BGR)
                "black": (0, 0, 0)       # OpenCV Black (BGR)
            }

            draw_color = color_map.get(guide_color.lower(), (0, 0, 255))
            complementary_color = (255 - draw_color[0], 255 - draw_color[1], 255 - draw_color[2])
            if canvas_bgr.shape[-1] == 3:
                x_step = w / 9.0
                y_step = h / 9.0

                line_thickness = max(2, int(min(w, h) / 500))
                grid_color = (160, 160, 160)
                for i in range(1, 9):
                    gx = int(x_step * i)
                    gy = int(y_step * i)

                    cv2.line(canvas_bgr, (gx, 0), (gx, h), grid_color, line_thickness)
                    cv2.putText(canvas_bgr, str(gx), (gx + 4, 18), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

                    cv2.line(canvas_bgr, (0, gy), (w, gy), grid_color, line_thickness)
                    cv2.putText(canvas_bgr, str(gy), (5, gy - 4), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 200), 1, cv2.LINE_AA)

                # lines = guide_color
                cv2.rectangle(canvas_bgr, (x1, y1), (x2, y2), draw_color, 2)
                cv2.line(canvas_bgr, (x2, y2), (0, y2), complementary_color, 2)   # Extend to the far left
                cv2.line(canvas_bgr, (x2, y1), (x2, 0), complementary_color, 2)   # Extend to the upper end
                canvas_cv = cv2.cvtColor(canvas_bgr, cv2.COLOR_BGR2RGB)

            preview_tensor = to_tensor_output(canvas_cv)
            return IO.NodeOutput(output,ui=UI.PreviewImage(preview_tensor))
        return IO.NodeOutput(output)

# -------------------------------

class IRL_CropMargins(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_CropMargins",
            display_name="크롭 마진",
            description="이미지 중앙을 기준으로 각 면의 모서리로부터 픽셀단위 이미지 자르기를 수행합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="크롭 처리할 이미지"),
                IO.Int.Input("left", default=0, min=0, tooltip="왼쪽에서 자를 픽셀 수"),
                IO.Int.Input("right", default=0, min=0, tooltip="오른쪽에서 자를 픽셀 수"),
                IO.Int.Input("top", default=0, min=0, tooltip="위쪽에서 자를 픽셀 수"),
                IO.Int.Input("bottom", default=0, min=0, tooltip="아래쪽에서 자를 픽셀 수"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="자른 결과를 노드에서 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="크롭된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, left, right, top, bottom, preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        pil_img = Image.fromarray(arr)
        w, h = pil_img.size
        cx, cy = w // 2, h // 2
        max_left = cx
        max_right = w - cx
        max_top = cy
        max_bottom = h - cy
        
        left = min(max(left, 0), max_left)
        top = min(max(top, 0), max_top)
        right = min(max(right, 0), max_right)
        bottom = min(max(bottom, 0), max_bottom)

        # Validation
        if (left + right) >= w:
            raise ValueError(f"[IRL_CropMargins Error] 좌우 크롭 값(left: {left}, right: {right}) 합이 (Width:{w})이상입니다. 유효한 크롭 영역이 없습니다.")

        if (top + bottom) >= h:
            raise ValueError(f"[IRL_CropMargins Error] 상하 크롭 값(top: {top}, bottom: {bottom}) 합이 (Height:{h})이상입니다. 유효한 크롭 영역이 없습니다.")
        
        base_left = cx   # The default maximum distance to be applied when equal
        base_right = w - cx
        base_top = cy
        base_bottom = h - cy
        w_left = base_left - left
        w_right = base_right - right
        h_top = base_top - top
        h_bottom = base_bottom - bottom

        if w_left + w_right > w:
            print(f"[IRL_CropMargins] Warning: The sum of left({left}) + right({right}) is already greater than the image width ({w}). Adjust the crop area.")
            # A safety mechanism to reduce to the appropriate ratio or to leave only the minimal area (1 pixel) visible
            scale = (w - 1) / (w_left + w_right) if (w_left + w_right) > 0 else 1
            wi_left = int(w_left * scale)
            w_right = w - 1 - wi_left
            w_left = wi_left
        elif w_left < 0 or w_right < 0:
            raise ValueError(f"[IRL_CropMargins] The left-right crop value (left: {left}, right: {right}) was too large and exceeded the valid crop region. (Result: negative occurrence)")
        else:
            pass

        if h_top + h_bottom > h:
            print(f"[IRL_CropMargins] Warning: The sum of top({top}) and bottom({bottom}) is at least the image height({h}). Adjust the crop area.")
            scale = (h - 1) / (h_top + h_bottom) if (h_top + h_bottom) > 0 else 1
            he_top = int(h_top * scale)
            h_bottom = h - 1 - he_top
            h_top = he_top
        elif h_top < 0 or h_bottom < 0:
            raise ValueError(f"[IRL_CropMargins] The top and bottom crop values (top: {top}, bottom: {bottom}) were too large and exceeded the valid crop region. (Result: negative occurrence)")
        else:
            pass
            
        x1 = max(cx - w_left, 0) 
        y1 = max(cy - h_top, 0)
        x2 = min(cx + w_right, w)
        y2 = min(cy + h_bottom, h)

        cropped = pil_img.crop((x1, y1, x2, y2))
        output = to_tensor_output(cropped)
        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))
        return IO.NodeOutput(output)

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
                IO.Combo.Input("padding_color", options=["neutral_gray", "white", "black"], default="neutral_gray", tooltip="왜곡 여백 색상"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="퍼스펙티브 왜곡이 적용된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, src_p1_x, src_p1_y, src_p2_x, src_p2_y, src_p3_x, src_p3_y, 
                src_p4_x, src_p4_y, dst_p1_x, dst_p1_y, dst_p2_x, dst_p2_y, dst_p3_x, dst_p3_y, 
                dst_p4_x, dst_p4_y, padding_color) -> IO.NodeOutput:

        arr = to_numpy_image(image)
        h, w = arr.shape[:2]

        # Padding color mapping
        color_map = {
            "neutral_gray": (128, 128, 128),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        bg_rgb = color_map.get(padding_color, (128, 128, 128))

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
        warped = cv2.warpPerspective(arr, matrix, (w, h), borderValue=bg_rgb)

        return IO.NodeOutput(to_tensor_output(Image.fromarray(warped)))

# -------------------------------

class IRL_MaskColorFill(IO.ComfyNode):

    @classmethod
    def ensure_mask_tensor(cls, t: torch.Tensor) -> torch.Tensor:
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

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_MaskColorFill",
            display_name="마스크 색상 채우기",
            description="이미지에서 마스크 영역을 특정 색상으로 덮습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="원본 이미지"),
                IO.Mask.Input("mask", tooltip="작업 영역 마스크"),
                IO.String.Input("fill_color", default="128, 128, 128", tooltip="RGB 값을 콤마로 구분하여 입력합니다 (예: 128, 128, 128). 비워둘 경우 중립 회색으로 처리됩니다."),
                IO.Boolean.Input("preview_mode", default=False, tooltip="자른 결과를 노드에서 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="회전된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, mask, fill_color="128, 128, 128", preview_mode=False) -> IO.NodeOutput:
        image_tensor = ensure_image_tensor(image)  # [B, C, H, W] (Channels-First)
        mask_tensor = cls.ensure_mask_tensor(mask) # [B, 1, H, W]
        f_color = (128, 128, 128)
        try:
            if fill_color and fill_color.strip():
                parts = [int(p.strip()) for p in fill_color.split(',')]
                if len(parts) == 3:
                    f_color = tuple(parts)
                elif len(parts) == 1:
                    val = parts[0]
                    f_color = (val, val, val)
        except Exception:
            # Maintains a safe neutral gray state upon parsing failure
            f_color = (128, 128, 128)

        r, g, b = f_color
        fill_val = torch.tensor([r / 255.0, g / 255.0, b / 255.0], dtype=image_tensor.dtype, device=image_tensor.device)
        fill_val = fill_val.view(1, 3, 1, 1)

        # Safe resizing in case of different resolutions
        if mask_tensor.shape[2:] != image_tensor.shape[2:]:
            mask_tensor = torch.nn.functional.interpolate(
                mask_tensor,
                size=image_tensor.shape[2:],
                mode='bilinear',
                align_corners=False
            )

        mask_tensor = torch.clamp(mask_tensor, 0.0, 1.0)

        # Masked area blending operation (fill the area with mask=1 with the fill color, and keep the original area with mask=0)
        output_tensor = image_tensor * (1.0 - mask_tensor) + fill_val * mask_tensor
        
        output = output_tensor.permute(0, 2, 3, 1) #B, C, H, W -> B, H, W, C
        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))
        return IO.NodeOutput(output)

# ----------------------------------------
# Grid Perspective Transformer Node (with Smart Color Control)
# ----------------------------------------

class IRL_GridGuidancePerspectiveWarp(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_GridGuidancePerspectiveWarp",
            display_name="그리드가이던스 퍼스펙티브 왜곡",
            description="원본과 대상 좌표를 사용하여 이미지에 원근 왜곡을 적용합니다. 프리뷰 가이던스를 통해 어떤 식으로 왜곡될지를 볼 수 있습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="왜곡할 이미지"),
                # Destination points (0.0 ~ 1.0 ratio)
                IO.Float.Input("dst_p1_x", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P1 X 비율"),
                IO.Float.Input("dst_p1_y", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P1 Y 비율"),
                IO.Float.Input("dst_p2_x", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P2 X 비율"),
                IO.Float.Input("dst_p2_y", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P2 Y 비율"),
                IO.Float.Input("dst_p3_x", default=0.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P3 X 비율"),
                IO.Float.Input("dst_p3_y", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P3 Y 비율"),
                IO.Float.Input("dst_p4_x", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P4 X 비율"),
                IO.Float.Input("dst_p4_y", default=1.0, min=0.0, max=1.0, step=0.01, tooltip="대상 P4 Y 비율"),
                IO.Combo.Input("padding_color", options=["neutral_gray", "white", "black"], default="neutral_gray", tooltip="왜곡 여백 색상"),
                IO.Combo.Input("method", default="Bicubic", options=["Lanczos", "Bicubic", "Nearest", "PixelBox"], tooltip="보간법"),
                IO.Boolean.Input("preview_mode", default=True, tooltip="9분할 그리드와 왜곡 가이드라인 미리보기"),
                IO.Combo.Input("guide_color", options=["red", "blue", "green", "yellow", "black"], default="red", tooltip="프리뷰 가이드라인 색상"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="퍼스펙티브 왜곡이 적용된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, dst_p1_x, dst_p1_y, dst_p2_x, dst_p2_y, dst_p3_x, dst_p3_y, 
                dst_p4_x, dst_p4_y, padding_color, method, preview_mode, guide_color) -> IO.NodeOutput:

        arr = to_numpy_image(image)
        h, w = arr.shape[:2]

        # Color and Interpolation maps
        color_map = {
            "neutral_gray": (128, 128, 128),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        bg_rgb = color_map.get(padding_color, (128, 128, 128))
        cv_interp_map = {
            "Bicubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4,
            "Nearest": cv2.INTER_NEAREST,
            "PixelBox": cv2.INTER_AREA
        }
        cv_method = cv_interp_map.get(method, cv2.INTER_CUBIC)

        # Source and destination points

        # Source points
        src = np.float32([
            [0.0, 0.0],
            [w, 0.0],
            [0.0, h],
            [w, h]
        ])

        # destination points
        dst = np.float32([
            [dst_p1_x * w, dst_p1_y * h],
            [dst_p2_x * w, dst_p2_y * h],
            [dst_p3_x * w, dst_p3_y * h],
            [dst_p4_x * w, dst_p4_y * h]
        ])

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(arr, matrix, (w, h), borderValue=bg_rgb, flags=cv_method)

        output = to_tensor_output(Image.fromarray(warped))

        if preview_mode:
            preview_arr = warped.copy()
            slot_w = w / 9.0
            slot_h = h / 9.0

            # 1) 9-division grid + scale labels
            grid_color = (180, 180, 180)
            label_color = (0, 255, 255)
            for i in range(1, 9):
                gx = int(i * slot_w)
                gy = int(i * slot_h)
                cv2.line(preview_arr, (gx, 0), (gx, h), grid_color, 1, cv2.LINE_AA)
                cv2.line(preview_arr, (0, gy), (w, gy), grid_color, 1, cv2.LINE_AA)
                cv2.putText(preview_arr, f"{i}:{gx}", (gx + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
                cv2.putText(preview_arr, f"{i}:{gy}", (2, gy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

            cv2.putText(preview_arr, f"0:0", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
            cv2.putText(preview_arr, f"9:{w}", (max(0, w - 55), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
            cv2.putText(preview_arr, f"9:{h}", (2, max(12, h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

            guide_color_map = {
                "red": (255, 0, 0),
                "blue": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (255, 255, 0),
                "black": (0, 0, 0)
            }
            line_rgb = guide_color_map.get(guide_color, (255, 0, 0))

            # 2) Target (dst) rectangle outline
            pts_int = np.int32(dst)
            cv2.polylines(preview_arr, [pts_int], isClosed=True, color=line_rgb, thickness=3, lineType=cv2.LINE_AA)

            # 3) Original -> Destination arrow and P1~P4 corner number/coordinate display
            arrow_color = (255 - line_rgb[0], 255 - line_rgb[1], 255 - line_rgb[2])
            for idx, (s_pt, d_pt) in enumerate(zip(src, dst), start=1):
                s_int = tuple(np.int32(s_pt))
                d_int = tuple(np.int32(d_pt))
                
                sx, sy = s_int[0], s_int[1]
                s_offset_x = 10 if sx < w / 2 else -35
                s_offset_y = 20 if sy < h / 2 else -10
                cv2.putText(preview_arr, f"P{idx}", (sx + s_offset_x, sy + s_offset_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

                if s_int != d_int:
                    cv2.arrowedLine(preview_arr, s_int, d_int, arrow_color, 2, cv2.LINE_AA, tipLength=0.15)

                
                # Show original points
                cv2.circle(preview_arr, s_int, 4, arrow_color, -1)
                
                # Display the actual coordinate text with numbers P1 to P4 at the target (dst) corner
                dx, dy = int(d_pt[0]), int(d_pt[1])
                cv2.circle(preview_arr, (dx, dy), 5, line_rgb, -1)
                cv2.putText(preview_arr, f"P{idx}({dx},{dy})", (dx + 5, dy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview_arr, f"P{idx}({dx},{dy})", (dx + 5, dy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_rgb, 1, cv2.LINE_AA)
            
            poutput = to_tensor_output(Image.fromarray(preview_arr))
            return IO.NodeOutput(output, ui=UI.PreviewImage(poutput))
 

        return IO.NodeOutput(output)

# ----------------------------------------

class IRL_DragGridGuidancePerspectiveWarp(IO.ComfyNode):

    @classmethod
    def parse_point_str(cls, s, default=(0.0, 0.0)):
        m = re.search(r"P\d+\{([-+]?\d*\.?\d+),\s*([-+]?\d*\.?\d+)\}", s)
        if m:
            return float(m.group(1)), float(m.group(2))
        return default

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_DragGridGuidancePerspectiveWarp",
            display_name="그리드가이던스 퍼스펙티브 왜곡(드래그모드)",
            description="드래그한 좌표들을 기준으로 이미지에 원근 왜곡을 적용합니다.\n"
                        "노드 자체의 실행키를 한번 누르고 나면 입력 프롬프트 위젯이 사라지며, 드래그로 제어점을 움직이는 형태가 됩니다.\n"
                        "프리뷰 가이던스를 통해 어떤 식으로 왜곡될지를 예상할 수 있고, 노드 실행키를 켜면 미리보기로 변형이미지를 볼 수 있습니다.\n"
                        "노드위젯 2.0모드일 경우, 좌표를 실시간으로 이동하는 것의 구현이 실패해서 좌표를 움직이고 나서 리프레시 포인트를 기동할 필요가 있습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="왜곡할 이미지"),
                # Destination points (0.0 ~ 1.0 ratio)
                IO.String.Input("dst_p1", default="P1{0.0,0.0}", tooltip="대상 P1 좌표 (비율)"),
                IO.String.Input("dst_p2", default="P2{1.0,0.0}", tooltip="대상 P2 좌표 (비율)"),
                IO.String.Input("dst_p3", default="P3{0.0,1.0}", tooltip="대상 P3 좌표 (비율)"),
                IO.String.Input("dst_p4", default="P4{1.0,1.0}", tooltip="대상 P4 좌표 (비율)"),
                IO.Combo.Input("padding_color", options=["neutral_gray", "white", "black"], default="neutral_gray", tooltip="왜곡 여백 색상"),
                IO.Combo.Input("method", default="Bicubic", options=["Lanczos", "Bicubic", "Nearest", "PixelBox"], tooltip="보간법"),
                IO.Boolean.Input("preview_mode", default=True, tooltip="9분할 그리드와 왜곡 가이드라인 미리보기"),
                IO.Combo.Input("guide_color", options=["red", "blue", "green", "yellow", "black"], default="red", tooltip="프리뷰 가이드라인 색상"),
                IO.Boolean.Input("refresh_points", default=False, tooltip="스위치를 껐다 다시 켤 때마다 P_Points가 드래그로 이동 안된 것이 강제 갱신됩니다."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="퍼스펙티브 왜곡이 적용된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, dst_p1, dst_p2, dst_p3, dst_p4, padding_color, method, preview_mode, guide_color, refresh_points) -> IO.NodeOutput:

        arr = to_numpy_image(image)
        h, w = arr.shape[:2]

        # Color and Interpolation maps
        color_map = {
            "neutral_gray": (128, 128, 128),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        bg_rgb = color_map.get(padding_color, (128, 128, 128))
        cv_interp_map = {
            "Bicubic": cv2.INTER_CUBIC,
            "Lanczos": cv2.INTER_LANCZOS4,
            "Nearest": cv2.INTER_NEAREST,
            "PixelBox": cv2.INTER_AREA
        }
        cv_method = cv_interp_map.get(method, cv2.INTER_CUBIC)

        # Source and destination points
        p1 = cls.parse_point_str(dst_p1, (0.0,0.0))
        p2 = cls.parse_point_str(dst_p2, (1.0,0.0))
        p3 = cls.parse_point_str(dst_p3, (0.0,1.0))
        p4 = cls.parse_point_str(dst_p4, (1.0,1.0))

        # Source points
        src = np.float32([
            [0.0, 0.0],
            [w, 0.0],
            [0.0, h],
            [w, h]
        ])

        # Destination points
        dst = np.float32([
            [p1[0]*w, p1[1]*h],
            [p2[0]*w, p2[1]*h],
            [p3[0]*w, p3[1]*h],
            [p4[0]*w, p4[1]*h]
        ])

        # Perspective transform
        matrix = cv2.getPerspectiveTransform(src, dst)
        warped = cv2.warpPerspective(arr, matrix, (w, h), borderValue=bg_rgb, flags=cv_method)

        output = to_tensor_output(Image.fromarray(warped))

        if preview_mode:
            preview_arr = warped.copy()
            slot_w = w / 9.0
            slot_h = h / 9.0

            # 1) 9-division grid + scale labels
            grid_color = (180, 180, 180)
            label_color = (0, 255, 255)
            for i in range(1, 9):
                gx = int(i * slot_w)
                gy = int(i * slot_h)
                cv2.line(preview_arr, (gx, 0), (gx, h), grid_color, 1, cv2.LINE_AA)
                cv2.line(preview_arr, (0, gy), (w, gy), grid_color, 1, cv2.LINE_AA)
                cv2.putText(preview_arr, f"{i}:{gx}", (gx + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
                cv2.putText(preview_arr, f"{i}:{gy}", (2, gy - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

            cv2.putText(preview_arr, f"0:0", (2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
            cv2.putText(preview_arr, f"9:{w}", (max(0, w - 55), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)
            cv2.putText(preview_arr, f"9:{h}", (2, max(12, h - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.35, label_color, 1, cv2.LINE_AA)

            guide_color_map = {
                "red": (255, 0, 0),
                "blue": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (255, 255, 0),
                "black": (0, 0, 0)
            }
            line_rgb = guide_color_map.get(guide_color, (255, 0, 0))

            # 2) Target (dst) rectangle outline
            pts_int = np.int32(dst)
            cv2.polylines(preview_arr, [pts_int], isClosed=True, color=line_rgb, thickness=3, lineType=cv2.LINE_AA)

            # 3) Original -> Destination arrow and P1~P4 corner number/coordinate display
            arrow_color = (255 - line_rgb[0], 255 - line_rgb[1], 255 - line_rgb[2])
            for idx, (s_pt, d_pt) in enumerate(zip(src, dst), start=1):
                s_int = tuple(np.int32(s_pt))
                d_int = tuple(np.int32(d_pt))
                
                sx, sy = s_int[0], s_int[1]
                s_offset_x = 10 if sx < w / 2 else -35
                s_offset_y = 20 if sy < h / 2 else -10
                cv2.putText(preview_arr, f"P{idx}", (sx + s_offset_x, sy + s_offset_y), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 2, cv2.LINE_AA)

                if s_int != d_int:
                    cv2.arrowedLine(preview_arr, s_int, d_int, arrow_color, 2, cv2.LINE_AA, tipLength=0.15)

                
                # Show original points
                cv2.circle(preview_arr, s_int, 4, arrow_color, -1)
                
                # Display the actual coordinate text with numbers P1 to P4 at the target (dst) corner
                dx, dy = int(d_pt[0]), int(d_pt[1])
                cv2.circle(preview_arr, (dx, dy), 5, line_rgb, -1)
                cv2.putText(preview_arr, f"P{idx}({dx},{dy})", (dx + 5, dy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview_arr, f"P{idx}({dx},{dy})", (dx + 5, dy - 5), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, line_rgb, 1, cv2.LINE_AA)
            
            poutput = to_tensor_output(Image.fromarray(preview_arr))
            return IO.NodeOutput(output, ui=UI.PreviewImage(poutput))
 

        return IO.NodeOutput(output)

# ----------------------------------------
# Grid Spline Warp Node (Dynamic Edge Ratio Tracking + Rbf)
# ----------------------------------------

class IRL_GridSplineWarp(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_GridSplineWarp",
            display_name="그리드 스플라인 왜곡",
            description="모서리(P1, P3, P6, P8)의 이동에 따라 변의 중점과 중앙점이 유동적으로 연동되는 스플라인 왜곡 노드입니다.\n"
                        "제어점에 기입되어 있는 프롬프트는 0.0~1.0 사이의 숫자로만 교체해주시면 됩니다.\n"
                        "위젯의 이용을 최소화하기 위한 간이 설계라 Nan값이 발생할 경우 이미지가 원하는 대로의 변형이 되지 않을 가능성이 있습니다.\n"
                        "다이나믹 엣지를 킬 경우 각 변의 중간점은 입력이 무시되며, 자동으로 다른 점들에 맞춰 좌표가 지정됩니다.",
            inputs=[
                IO.Image.Input("image", tooltip="왜곡할 이미지"),
                # 8 individual control point input fields (ratios 0.0 ~ 1.0)
                IO.String.Input("P1", default="0.0, 0.0", tooltip="좌상단 모서리 (x, y)"),
                IO.String.Input("P2", default="0.5, 0.0", tooltip="중상단 제어점 (x, y)"),
                IO.String.Input("P3", default="1.0, 0.0", tooltip="우상단 모서리 (x, y)"),
                IO.String.Input("P4", default="0.0, 0.5", tooltip="좌중앙 제어점 (x, y)"),
                IO.String.Input("P5", default="1.0, 0.5", tooltip="우중앙 제어점 (x, y)"),
                IO.String.Input("P6", default="0.0, 1.0", tooltip="좌하단 모서리 (x, y)"),
                IO.String.Input("P7", default="0.5, 1.0", tooltip="중하단 제어점 (x, y)"),
                IO.String.Input("P8", default="1.0, 1.0", tooltip="우하단 모서리 (x, y)"),
                
                IO.Boolean.Input("dynamic_edges", default=True, tooltip="체크 시, 모서리(P1,P3,P6,P8)의 변위와 거리에 맞춰 P2,P4,P5,P7,CC가 유동적으로 자동 계산됩니다."),
                IO.Combo.Input("padding_color", options=["neutral_gray", "white", "black"], default="neutral_gray", tooltip="왜곡 여백 색상"),
                IO.Combo.Input("method", default="cubic", options=["linear", "cubic", "nearest"], tooltip="스플라인 보간 방식"),
                IO.Boolean.Input("preview_mode", default=True, tooltip="9점 메시 그리드와 제어점 미리보기"),
                IO.Combo.Input("guide_color", options=["red", "blue", "green", "yellow", "black"], default="red", tooltip="프리뷰 가이드라인 색상"),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="스플라인 왜곡이 적용된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, P1, P2, P3, P4, P5, P6, P7, P8, dynamic_edges, padding_color, method, preview_mode, guide_color) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        h, w = arr.shape[:2]

        def parse_point(pt_str, default_x, default_y):
            try:
                nums = re.findall(r"[-+]?\d*\.?\d+", pt_str)
                if len(nums) >= 2:
                    return float(nums[0]), float(nums[1])
            except:
                pass
            return default_x, default_y

        # Parsing user input
        p1 = np.array(parse_point(P1, 0.0, 0.0), dtype=np.float32)
        p2_user = np.array(parse_point(P2, 0.5, 0.0), dtype=np.float32)
        p3 = np.array(parse_point(P3, 1.0, 0.0), dtype=np.float32)
        p4_user = np.array(parse_point(P4, 0.0, 0.5), dtype=np.float32)
        p5_user = np.array(parse_point(P5, 1.0, 0.5), dtype=np.float32)
        p6 = np.array(parse_point(P6, 0.0, 1.0), dtype=np.float32)
        p7_user = np.array(parse_point(P7, 0.5, 1.0), dtype=np.float32)
        p8 = np.array(parse_point(P8, 1.0, 1.0), dtype=np.float32)

        if dynamic_edges:
            # Flow calculation according to the distance of each side and the vector ratio (Linear Interpolation along side vectors)
            # Top edge (P1 -> P3): default ratio 0.5 position
            p2 = p1 + 0.5 * (p3 - p1)
            # Left side edge (P1 -> P6)
            p4 = p1 + 0.5 * (p6 - p1)
            # Right side edge (P3 -> P8)
            p5 = p3 + 0.5 * (p8 - p3)
            # Lower edge (P6 -> P8)
            p7 = p6 + 0.5 * (p8 - p6)
            # Center point (CC): diagonal intersection or the average of the upper/lower and left/right midpoints
            cc = (p2 + p7 + p4 + p5) / 4.0
        else:
            p2 = p2_user
            p4 = p4_user
            p5 = p5_user
            p7 = p7_user
            cc = (p1 + p3 + p6 + p8) / 4.0 # Even in manual mode, the default center point maintains the edge average

        # Reassemble in order on a 3x3 grid
        dst_pts = np.array([
            p1, p2, p3,
            p4, cc, p5,
            p6, p7, p8
        ], dtype=np.float32)

        # Generate original grid coordinates (3x3 regular 9-point grid)
        src_x = np.array([0.0, 0.5, 1.0]) * w
        src_y = np.array([0.0, 0.5, 1.0]) * h
        gx, gy = np.meshgrid(src_x, src_y, indexing='xy')
        src_pts = np.vstack([gx.ravel(), gy.ravel()]).T # (9, 2)

        # Convert the target control point coordinates to pixel units
        dst_pts_px = dst_pts.copy()
        dst_pts_px[:, 0] *= w
        dst_pts_px[:, 1] *= h

        # 2) Pixel Remapping Using SciPy RBF (Spline Distortion)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        try:
            rbf_x = Rbf(dst_pts_px[:, 0], dst_pts_px[:, 1], src_pts[:, 0], function='thin_plate')
            rbf_y = Rbf(dst_pts_px[:, 0], dst_pts_px[:, 1], src_pts[:, 1], function='thin_plate')
            
            map_x = rbf_x(grid_x, grid_y).astype(np.float32)
            map_y = rbf_y(grid_x, grid_y).astype(np.float32)
        except Exception:
            map_x = griddata(dst_pts_px, src_pts[:, 0], (grid_x, grid_y), method='cubic', fill_value=np.nan)
            map_y = griddata(dst_pts_px, src_pts[:, 1], (grid_x, grid_y), method='cubic', fill_value=np.nan)

        # Missed (NaN) defense processing
        mask_nan = np.isnan(map_x) | np.isnan(map_y)
        if np.any(mask_nan):
            map_x_near = griddata(dst_pts_px, src_pts[:, 0], (grid_x, grid_y), method='nearest')
            map_y_near = griddata(dst_pts_px, src_pts[:, 1], (grid_x, grid_y), method='nearest')
            map_x[mask_nan] = map_x_near[mask_nan]
            map_y[mask_nan] = map_y_near[mask_nan]

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        color_map = {
            "neutral_gray": (128, 128, 128),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        bg_rgb = color_map.get(padding_color, (128, 128, 128))

        warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_rgb)
        output = to_tensor_output(Image.fromarray(warped))

        # 3) Preview mode handling
        if preview_mode:
            preview_arr = warped.copy()
            
            guide_color_map = {
                "red": (255, 0, 0),
                "blue": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (255, 255, 0),
                "black": (0, 0, 0)
            }
            line_rgb = guide_color_map.get(guide_color, (255, 0, 0))
            grid_color = (180, 180, 180)

            pts_2d = dst_pts_px.reshape(3, 3, 2).astype(np.int32)

            # 9-division scale grid
            for i in range(1, 9):
                gx = int(i * w / 9.0)
                gy = int(i * h / 9.0)
                cv2.line(preview_arr, (gx, 0), (gx, h), grid_color, 1, cv2.LINE_AA)
                cv2.line(preview_arr, (0, gy), (w, gy), grid_color, 1, cv2.LINE_AA)

            # 3x3 control point real distortion mesh line
            for r in range(3):
                row_pts = np.ascontiguousarray(pts_2d[r, :, :])
                cv2.polylines(preview_arr, [row_pts], isClosed=False, color=line_rgb, thickness=2, lineType=cv2.LINE_AA)
            for c in range(3):
                col_pts = np.ascontiguousarray(pts_2d[:, c, :])
                cv2.polylines(preview_arr, [col_pts], isClosed=False, color=line_rgb, thickness=2, lineType=cv2.LINE_AA)

            # 3x3 Safety Zone Sub-Cell Box
            for r in range(2):
                for c in range(2):
                    cell_pts = np.array([
                        pts_2d[r, c],       
                        pts_2d[r, c+1],     
                        pts_2d[r+1, c+1],   
                        pts_2d[r+1, c]      
                    ], dtype=np.int32)
                    cv2.polylines(preview_arr, [cell_pts], isClosed=True, color=(120, 220, 120), thickness=1, lineType=cv2.LINE_AA)

            labels = ["P1", "P2", "P3", "P4", "CC", "P5", "P6", "P7", "P8"]

            for pt, label in zip(dst_pts_px, labels):
                px, py = int(pt[0]), int(pt[1])
                pt_color = (255, 255, 0) if label == "CC" else line_rgb
                cv2.circle(preview_arr, (px, py), 6, pt_color, -1)
                
                text_x = max(5, px - 25)
                text_y = max(15, py - 10)
                cv2.putText(preview_arr, f"{label}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview_arr, f"{label}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, pt_color, 1, cv2.LINE_AA)

            poutput = to_tensor_output(Image.fromarray(preview_arr))
            return IO.NodeOutput(output, ui=UI.PreviewImage(poutput))

        return IO.NodeOutput(output)
# -------------------------------

class IRL_DragGridSplineWarp(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_DragGridSplineWarp",
            display_name="그리드 스플라인 왜곡(드래그모드)",
            description="드래그를 통해 모서리(P1, P3, P6, P8)의 이동에 따라 변의 중점과 중앙점이 유동적으로 연동되는 스플라인 왜곡 노드입니다.\n"
                        "드래그로 제어점을 누른 뒤 이동하는 만큼 좌표가 실시간으로 이동되며, 노드 자체의 실행키를 누르면 적용되는 형태가 됩니다.\n"
                        "다이나믹 엣지를 킬 경우 각 변의 중간점은 입력이 무시되며, 자동으로 다른 점들에 맞춰 좌표가 지정됩니다.\n"
                        "노드 2.0에서는 버그가 있어 포인트 마커가 실시간으로 이동되지 않지만, 리프레쉬포인트나 다이나믹 엣지 스위치를 켰다 끄면 마커가 적용됩니다.",
            inputs=[
                IO.Image.Input("image", tooltip="왜곡할 이미지"),
                # 8 individual control point input fields (ratios 0.0 ~ 1.0)
                IO.String.Input("p_point", default="P1{0.0,0.0}, P2{0.5,0.0}, P3{1.0,0.0}, P4{0.0,0.5}, P5{1.0,0.5}, P6{0.0,1.0}, P7{0.5,1.0}, P8{1.0,1.0}", 
                                tooltip="8점 좌표. 드래그 위젯이 이동시 자동 갱신됩니다. P1{0.0,0.0}, P2{0.5,0.0}, P3{1.0,0.0}, P4{0.0,0.5}, P5{1.0,0.5}, P6{0.0,1.0}, P7{0.5,1.0}, P8{1.0,1.0}형태로 인식합니다."),                
                IO.Boolean.Input("dynamic_edges", default=True, tooltip="체크 시, 모서리(P1,P3,P6,P8)의 변위와 거리에 맞춰 P2,P4,P5,P7,CC가 유동적으로 자동 계산됩니다."),
                IO.Combo.Input("padding_color", options=["neutral_gray", "white", "black"], default="neutral_gray", tooltip="왜곡 여백 색상"),
                IO.Combo.Input("method", default="cubic", options=["linear", "cubic", "nearest"], tooltip="스플라인 보간 방식"),
                IO.Boolean.Input("preview_mode", default=True, tooltip="9점 메시 그리드와 제어점 미리보기"),
                IO.Combo.Input("guide_color", options=["red", "blue", "green", "yellow", "black"], default="red", tooltip="프리뷰 가이드라인 색상"),
                IO.Boolean.Input("refresh_points", default=False, tooltip="스위치를 껐다 다시 켤 때마다 P_Points가 드래그로 이동 안된 것이 강제 갱신됩니다."),
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="스플라인 왜곡이 적용된 이미지"),
            ],
            category="이미지 리파이너/변형"
        )

    @classmethod
    def execute(cls, image, p_point, dynamic_edges, padding_color, method, preview_mode, guide_color, refresh_points=False) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        h, w = arr.shape[:2]

        def parse_point(pt_str, default_x, default_y):
            try:
                nums = re.findall(r"[-+]?\d*\.?\d+", pt_str)
                if len(nums) >= 2:
                    return float(nums[0]), float(nums[1])
            except:
                pass
            return default_x, default_y

        # Parsing user input
        parsed = parse_points(p_point)

        # Extract edge coordinates
        p1 = parsed.get("P1", np.array([0.0, 0.0]))
        p2_user = parsed.get("P2", np.array([0.5, 0.0]))
        p3 = parsed.get("P3", np.array([1.0, 0.0]))
        p4_user = parsed.get("P4", np.array([0.0, 0.5]))
        p5_user = parsed.get("P5", np.array([1.0, 0.5]))
        p6 = parsed.get("P6", np.array([0.0, 1.0]))
        p7_user = parsed.get("P7", np.array([0.5, 1.0]))
        p8 = parsed.get("P8", np.array([1.0, 1.0]))

        if dynamic_edges:
            # Flow calculation according to the distance of each side and the vector ratio (Linear Interpolation along side vectors)
            # Top edge (P1 -> P3): default ratio 0.5 position
            p2 = p1 + 0.5 * (p3 - p1)
            # Left side edge (P1 -> P6)
            p4 = p1 + 0.5 * (p6 - p1)
            # Right side edge (P3 -> P8)
            p5 = p3 + 0.5 * (p8 - p3)
            # Lower edge (P6 -> P8)
            p7 = p6 + 0.5 * (p8 - p6)
            # Center point (CC): diagonal intersection or the average of the upper/lower and left/right midpoints
            cc = (p2 + p7 + p4 + p5) / 4.0
        else:
            p2 = p2_user
            p4 = p4_user
            p5 = p5_user
            p7 = p7_user
            cc = (p1 + p3 + p6 + p8) / 4.0 # Even in manual mode, the default center point maintains the edge average

        # Reassemble in order on a 3x3 grid
        dst_pts = np.array([
            p1, p2, p3,
            p4, cc, p5,
            p6, p7, p8
        ], dtype=np.float32)

        # Generate original grid coordinates (3x3 regular 9-point grid)
        src_x = np.array([0.0, 0.5, 1.0]) * w
        src_y = np.array([0.0, 0.5, 1.0]) * h
        gx, gy = np.meshgrid(src_x, src_y, indexing='xy')
        src_pts = np.vstack([gx.ravel(), gy.ravel()]).T # (9, 2)

        # Convert the target control point coordinates to pixel units
        dst_pts_px = dst_pts.copy()
        dst_pts_px[:, 0] *= w
        dst_pts_px[:, 1] *= h

        # Pixel Remapping Using SciPy RBF (Spline Distortion)
        grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h), indexing='xy')
        try:
            rbf_x = Rbf(dst_pts_px[:, 0], dst_pts_px[:, 1], src_pts[:, 0], function='thin_plate')
            rbf_y = Rbf(dst_pts_px[:, 0], dst_pts_px[:, 1], src_pts[:, 1], function='thin_plate')
            
            map_x = rbf_x(grid_x, grid_y).astype(np.float32)
            map_y = rbf_y(grid_x, grid_y).astype(np.float32)
        except Exception:
            map_x = griddata(dst_pts_px, src_pts[:, 0], (grid_x, grid_y), method='cubic', fill_value=np.nan)
            map_y = griddata(dst_pts_px, src_pts[:, 1], (grid_x, grid_y), method='cubic', fill_value=np.nan)

        # Missed (NaN) defense processing
        mask_nan = np.isnan(map_x) | np.isnan(map_y)
        if np.any(mask_nan):
            map_x_near = griddata(dst_pts_px, src_pts[:, 0], (grid_x, grid_y), method='nearest')
            map_y_near = griddata(dst_pts_px, src_pts[:, 1], (grid_x, grid_y), method='nearest')
            map_x[mask_nan] = map_x_near[mask_nan]
            map_y[mask_nan] = map_y_near[mask_nan]

        map_x = map_x.astype(np.float32)
        map_y = map_y.astype(np.float32)

        color_map = {
            "neutral_gray": (128, 128, 128),
            "white": (255, 255, 255),
            "black": (0, 0, 0)
        }
        bg_rgb = color_map.get(padding_color, (128, 128, 128))

        warped = cv2.remap(arr, map_x, map_y, interpolation=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=bg_rgb)
        output = to_tensor_output(Image.fromarray(warped))

        # 3) Preview mode handling
        if preview_mode:
            preview_arr = warped.copy()
            
            guide_color_map = {
                "red": (255, 0, 0),
                "blue": (0, 0, 255),
                "green": (0, 255, 0),
                "yellow": (255, 255, 0),
                "black": (0, 0, 0)
            }
            line_rgb = guide_color_map.get(guide_color, (255, 0, 0))
            grid_color = (180, 180, 180)

            pts_2d = dst_pts_px.reshape(3, 3, 2).astype(np.int32)

            # 9-division scale grid
            for i in range(1, 9):
                gx = int(i * w / 9.0)
                gy = int(i * h / 9.0)
                cv2.line(preview_arr, (gx, 0), (gx, h), grid_color, 1, cv2.LINE_AA)
                cv2.line(preview_arr, (0, gy), (w, gy), grid_color, 1, cv2.LINE_AA)

            # 3x3 control point real distortion mesh line
            for r in range(3):
                row_pts = np.ascontiguousarray(pts_2d[r, :, :])
                cv2.polylines(preview_arr, [row_pts], isClosed=False, color=line_rgb, thickness=2, lineType=cv2.LINE_AA)
            for c in range(3):
                col_pts = np.ascontiguousarray(pts_2d[:, c, :])
                cv2.polylines(preview_arr, [col_pts], isClosed=False, color=line_rgb, thickness=2, lineType=cv2.LINE_AA)

            # 3x3 Safety Zone Sub-Cell Box
            for r in range(2):
                for c in range(2):
                    cell_pts = np.array([
                        pts_2d[r, c],       
                        pts_2d[r, c+1],     
                        pts_2d[r+1, c+1],   
                        pts_2d[r+1, c]      
                    ], dtype=np.int32)
                    cv2.polylines(preview_arr, [cell_pts], isClosed=True, color=(120, 220, 120), thickness=1, lineType=cv2.LINE_AA)

            labels = ["P1", "P2", "P3", "P4", "CC", "P5", "P6", "P7", "P8"]

            for pt, label in zip(dst_pts_px, labels):
                px, py = int(pt[0]), int(pt[1])
                pt_color = (255, 255, 0) if label == "CC" else line_rgb
                cv2.circle(preview_arr, (px, py), 6, pt_color, -1)
                
                text_x = max(5, px - 25)
                text_y = max(15, py - 10)
                cv2.putText(preview_arr, f"{label}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.putText(preview_arr, f"{label}", (text_x, text_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, pt_color, 1, cv2.LINE_AA)

            poutput = to_tensor_output(Image.fromarray(preview_arr))
            return IO.NodeOutput(output, ui=UI.PreviewImage(poutput))

        return IO.NodeOutput(output)
# -------------------------------



TRANSFORM_NODE_CLASS_MAPPINGS = {
    "IRL_Resize": IRL_Resize,
    "IRL_VecterResize": IRL_VecterResize,
    "IRL_Resize_Upsize_only": IRL_Resize_Upsize_only,
    "IRL_Resize_downsize_only": IRL_Resize_downsize_only,
    "IRL_Rotate": IRL_Rotate,
    "IRL_Flip": IRL_Flip,
    "IRL_Guidance_Crop": IRL_Guidance_Crop,
    "IRL_CropMargins": IRL_CropMargins,
    "IRL_PerspectiveWarp": IRL_PerspectiveWarp,
    "IRL_MaskColorFill": IRL_MaskColorFill,
    "IRL_GridGuidancePerspectiveWarp": IRL_GridGuidancePerspectiveWarp,
    "IRL_DragGridGuidancePerspectiveWarp": IRL_DragGridGuidancePerspectiveWarp,
    "IRL_GridSplineWarp": IRL_GridSplineWarp,
    "IRL_DragGridSplineWarp": IRL_DragGridSplineWarp,
}

TRANSFORM_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_Resize": "리사이즈",
    "IRL_VecterResize": "벡터 리사이즈",
    "IRL_Resize_Upsize_only": "리사이즈(업사이즈 전용)",
    "IRL_Resize_downsize_only": "리사이즈(다운사이즈 전용)",
    "IRL_Rotate": "회전",
    "IRL_Flip": "플립",
    "IRL_Guidance_Crop": "가이던스 크롭",
    "IRL_CropMargins": "크롭 마진",
    "IRL_PerspectiveWarp": "퍼스펙티브 왜곡",
    "IRL_MaskColorFill": "마스크 색상 채우기",
    "IRL_GridGuidancePerspectiveWarp": "그리드가이던스 퍼스펙티브 왜곡",
    "IRL_DragGridGuidancePerspectiveWarp": "그리드가이던스 퍼스펙티브 왜곡(드래그모드)",
    "IRL_GridSplineWarp": "그리드 스플라인 왜곡 (9점)",
    "IRL_DragGridSplineWarp": "그리드 스플라인 왜곡(드래그모드)",
}

# -------------------------------
