# -------------------------------
# IR Lite — Analysis Node
# (LOCALE-based multilingual description support included)
# -------------------------------

import os
import sys
import platform
import torch
import numpy as np
import cv2
from PIL import Image, ImageDraw, ImageFont

from comfy_api.latest import IO, UI
import matplotlib.pyplot as plt
from io import BytesIO

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

def to_tensor_mask(mask: Image.Image):
    arr = np.array(mask).astype(np.float32) / 255.0
    arr = arr[None, ..., None]  # add batch + add color channel
    return torch.from_numpy(arr)

def to_numpy_mask(mask):
    if isinstance(mask, torch.Tensor):
        arr = mask[0].cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0,255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return arr.squeeze()
    elif isinstance(mask, Image.Image):
        return np.array(mask.convert("L"))
    elif isinstance(mask, np.ndarray):
        return mask.astype(np.uint8)
    else:
        raise TypeError("Unsupported mask type")
        
def get_font(font_size=48):
    system = platform.system()

    try:
        if system == "Windows":
            return ImageFont.truetype("C:/Windows/Fonts/arial.ttf", font_size)
        elif system == "Linux":
            return ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", font_size)
        elif system == "Darwin":  # macOS
            return ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", font_size)
        elif custom_path and os.path.exists(custom_path):
            # Custom font path input Example: path = "/ / /fontname.ttf"
            path = custom_path

            return ImageFont.truetype(path, font_size)
        else:
            # Unitified OS → Pillow base font
            return ImageFont.load_default()

    except OSError:
        # OS Check Error → Pillow base font
        return ImageFont.load_default()


# -------------------------------

class IRL_RGBSplit(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_RGBSplit",
            display_name="이미지 3채널 색상 분리",
            description="이미지의 RGB 채널을 분리하여 출력합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="이미지 채널을 분리할 이미지"),
            ],
            outputs=[
                IO.Image.Output("red", tooltip="적색 채널 이미지"),
                IO.Image.Output("green", tooltip="녹색 채널 이미지"),
                IO.Image.Output("blue", tooltip="청색 채널 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        arr = to_numpy_image(image)

        # extraction Channel Color Points

        red   = np.zeros_like(arr); red[...,0] = arr[...,0]
        green = np.zeros_like(arr); green[...,1] = arr[...,1]
        blue  = np.zeros_like(arr); blue[...,2] = arr[...,2]

        # PIL Img Conversion
        img_r = Image.fromarray(red)
        img_g = Image.fromarray(green)
        img_b = Image.fromarray(blue)

        return IO.NodeOutput(
            to_tensor_output(img_r),
            to_tensor_output(img_g),
            to_tensor_output(img_b)
        )
# -------------------------------

class IRL_HistogramPlot(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_HistogramPlot",
            display_name="이미지 히스토그램 그래프",
            description="이미지의 RGB 채널을 분리하여 히스토그램 그래프 형태로 출력합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="히스토그램을 계산할 이미지"),
            ],
            outputs=[
                IO.Image.Output("histogram", tooltip="RGB 히스토그램 그래프 출력"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image) -> IO.NodeOutput:
        arr = to_numpy_image(image)

        # extraction Channel Color Points Histogram Calculate
        r_hist = np.histogram(arr[...,0], bins=256, range=(0,255))[0]
        g_hist = np.histogram(arr[...,1], bins=256, range=(0,255))[0]
        b_hist = np.histogram(arr[...,2], bins=256, range=(0,255))[0]

        # Drawing graph

        plt.figure(figsize=(6,4))
        plt.plot(r_hist, color="red", label="Red")
        plt.plot(g_hist, color="green", label="Green")
        plt.plot(b_hist, color="blue", label="Blue")
        plt.legend()
        plt.title("RGB Histogram")
        plt.xlabel("Pixel value")
        plt.ylabel("Frequency")

        # Img Conversion
        buf = BytesIO()
        plt.savefig(buf, format="PNG")
        plt.close()
        buf.seek(0)
        img = Image.open(buf).convert("RGB")

        return IO.NodeOutput(to_tensor_output(img))


# -------------------------------
class IRL_ImageMeanStd(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageMeanStd",
            display_name="이미지 평균 & 표준편차",
            description="이미지 픽셀값의 평균값과 표준편차를 계산합니다.\n"
                        "운영체제가 윈도우가 아닐 경우 폰트가 경로에 없어 사이즈가 처리 안 될 가능성이 있습니다.\n"
                        "동봉된 윈도우즈 기본 폰트를 설치하는걸 권장합니다.\n"
                        "혹은 이 파일을 수정해 공용함수의 커스텀 경로를 지정하면 지정된 경로의 해당 폰트를 사용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="분석할 이미지"),
                IO.Int.Input("font_size", default=48, min=30, max=54, tooltip="폰트 크기 (30~54)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="평균과 표준편차가 표시된 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=48) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        mean_value, std_value = float(gray.mean()), float(gray.std())

        # Display text on the canvas
        canvas = Image.new("RGB", (512, 256), "black")

        draw = ImageDraw.Draw(canvas)
        # Arial pont, size 48
        font = get_font(font_size)

        draw.text((10, 40), f"MEAN: {mean_value:.4f}", fill=(255, 255, 0), font=font)
        draw.text((10, 140), f"STD:  {std_value:.4f}", fill=(255, 255, 0), font=font)

        return IO.NodeOutput(to_tensor_output(canvas))

# -------------------------------
class IRL_ImageMinMax(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageMinMax",
            display_name="이미지 최소 & 최대",
            description="이미지 픽셀값의 최소값과 최대값을 계산합니다.\n"
                        "운영체제가 윈도우가 아닐 경우 폰트가 경로에 없어 사이즈가 처리 안 될 가능성이 있습니다.\n"
                        "동봉된 윈도우즈 기본 폰트를 설치하는걸 권장합니다.\n"
                        "혹은 이 파일을 수정해 공용함수의 커스텀 경로를 지정하면 지정된 경로의 해당 폰트를 사용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="분석할 이미지"),
                IO.Int.Input("font_size", default=48, min=30, max=54, tooltip="폰트 크기 (30~54)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="최소/최대값이 표시된 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=48) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        min_value, max_value = float(gray.min()), float(gray.max())

        # Display text on the canvas
        canvas = Image.new("RGB", (512, 256), "black")
        draw = ImageDraw.Draw(canvas)

        # Arial pont, size 48
        font = get_font(font_size)

        draw.text((10, 40), f"MIN: {min_value:.4f}", fill=(255, 255, 0), font=font)
        draw.text((10, 140), f"MAX: {max_value:.4f}", fill=(255, 255, 0), font=font)

        return IO.NodeOutput(to_tensor_output(canvas))


# -------------------------------

class IRL_ImageEdgeMap(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageEdgeMap",
            display_name="이미지 에지 맵",
            description="소벨 연산자를 사용하여 이미지의 에지 맵을 생성합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="엣지맵을 추출할 이미지"),
                IO.Float.Input("edge_scale", default=1.0, min=0.1, max=5.0, step=0.1, tooltip="엣지 강도 스케일")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="엣지 맵 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, edge_scale=1.0) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        h, w = arr.shape[:2]

        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        gray = gray.astype(np.uint8)

        # Sobel Operation
        sobel_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0)
        sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1)
        edge = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Weight Scaling
        edge = edge / edge.max() * 255
        edge = edge * edge_scale
        
        edge = np.clip(edge, 0, 255).astype(np.uint8)
        canvas = Image.fromarray(edge)

        return IO.NodeOutput(to_tensor_output(canvas))
        
# -------------------------------

class IRL_ImageBrightnessContrast(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageBrightnessContrast",
            display_name="밝기 & 대비",
            description="이미지의 밝기와 대비를 계산합니다.\n"
                        "운영체제가 윈도우가 아닐 경우 폰트가 경로에 없어 사이즈가 처리 안 될 가능성이 있습니다.\n"
                        "동봉된 윈도우즈 기본 폰트를 설치하는걸 권장합니다.\n"
                        "혹은 이 파일을 수정해 공용함수의 커스텀 경로를 지정하면 지정된 경로의 해당 폰트를 사용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="분석할 이미지"),
                IO.Int.Input("font_size", default=48, min=30, max=54, tooltip="폰트 크기 (30~54)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="밝기값과 대비값이 표시된 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=48) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        brightness_value, contrast_value = float(gray.mean()), float(gray.std())

        # Display text on the canvas
        canvas = Image.new("RGB", (512, 256), "black")
        draw = ImageDraw.Draw(canvas)

        # Arial pont, size 48
        font = get_font(font_size)

        draw.text((10, 40), f"Brightness: {brightness_value:.4f}", fill=(255, 255, 0), font=font)
        draw.text((10, 140), f"Contrast:   {contrast_value:.4f}", fill=(255, 255, 0), font=font)

        return IO.NodeOutput(to_tensor_output(canvas))



# -------------------------------
class IRL_CannyEdgeStats(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_CannyEdgeStats",
            display_name="캐니 에지 통계",
            description="캐니 에지 검출을 통해 에지 밀도와 평균을 계산합니다.\n"
                        "운영체제가 윈도우가 아닐 경우 폰트가 경로에 없어 사이즈가 처리 안 될 가능성이 있습니다.\n"
                        "동봉된 윈도우즈 기본 폰트를 설치하는걸 권장합니다.\n"
                        "혹은 이 파일을 수정해 공용함수의 커스텀 경로를 지정하면 지정된 경로의 해당 폰트를 사용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="케니 엣지를 검출해 분석할 이미지"),
                IO.Int.Input("font_size", default=40, min=30, max=40, tooltip="폰트 크기 (30~54)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="케니엣지 밀도와 평균통계가 표시된 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=40) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8) if arr.ndim == 3 else arr.astype(np.uint8)

        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200).astype(np.float32)
        edge_density = float((edges > 0).mean())
        edge_mean = float(edges.mean() / 255.0)

        # Display text on the canvas
        canvas = Image.new("RGB", (512, 256), "black")
        draw = ImageDraw.Draw(canvas)

        # Arial pont, size 40
        font = get_font(font_size)

        draw.text((10, 40), f"EDGE_DENSITY: {edge_density:.4f}", fill=(255, 255, 0), font=font)
        draw.text((10, 140), f"EDGE_MEAN:    {edge_mean:.4f}", fill=(255, 255, 0), font=font)

        return IO.NodeOutput(to_tensor_output(canvas))


# -------------------------------

class IRL_DepthStats(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_DepthStats",
            display_name="깊이 통계",
            description="이미지의 깊이 평균과 표준편차를 계산합니다.\n"
                        "운영체제가 윈도우가 아닐 경우 폰트가 경로에 없어 사이즈가 처리 안 될 가능성이 있습니다.\n"
                        "동봉된 윈도우즈 기본 폰트를 설치하는걸 권장합니다.\n"
                        "혹은 이 파일을 수정해 공용함수의 커스텀 경로를 지정하면 지정된 경로의 해당 폰트를 사용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="뎁스 평균값과 표준편차를 분석할 이미지"),
                IO.Int.Input("font_size", default=40, min=30, max=40, tooltip="폰트 크기 (30~54)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="뎁스 평균값과 표준편차 통계가 표시된 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=40) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        arr = gray.astype(np.float32) / 255.0
        depth_mean, depth_std = float(arr.mean()), float(arr.std())

        # Display text on the canvas
        canvas = Image.new("RGB", (512, 256), "black")
        draw = ImageDraw.Draw(canvas)

        # Arial pont, size 40
        font = get_font(font_size)

        draw.text((10, 40), f"DEPTH_MEAN: {depth_mean:.4f}", fill=(255, 255, 0), font=font)
        draw.text((10, 140), f"DEPTH_STD:  {depth_std:.4f}", fill=(255, 255, 0), font=font)

        return IO.NodeOutput(to_tensor_output(canvas))


# -------------------------------

ANALYSIS_NODE_CLASS_MAPPINGS = {
    "IRL_RGBSplit": IRL_RGBSplit,
    "IRL_HistogramPlot": IRL_HistogramPlot,
    "IRL_ImageMeanStd": IRL_ImageMeanStd,
    "IRL_ImageMinMax": IRL_ImageMinMax,
    "IRL_ImageEdgeMap": IRL_ImageEdgeMap,
    "IRL_ImageBrightnessContrast": IRL_ImageBrightnessContrast,
    "IRL_CannyEdgeStats": IRL_CannyEdgeStats,
    "IRL_DepthStats": IRL_DepthStats,
}

ANALYSIS_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_RGBSplit": "이미지 3채널 색상 분리",
    "IRL_HistogramPlot": "이미지 히스토그램 그래프",
    "IRL_ImageMeanStd": "이미지 평균 & 표준편차",
    "IRL_ImageMinMax": "이미지 최소 & 최대",
    "IRL_ImageEdgeMap": "이미지 에지 맵",
    "IRL_ImageBrightnessContrast": "밝기 & 대비",
    "IRL_CannyEdgeStats": "캐니 에지 통계",
    "IRL_DepthStats": "깊이 통계",
}

# -------------------------------
