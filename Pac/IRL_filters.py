# -------------------------------
# IR Lite — Filters Nodes
# (LOCALE-based multilingual description support included)
# -------------------------------
import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from skimage import exposure

# ComfyUI 최신 API
from comfy_api.latest import IO, UI




# -------------------------------
# 공통 유틸 함수
def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]
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
    arr = arr[None, ..., None]
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

# -------------------------------

class IRL_GaussianBlur(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_GaussianBlur",
            display_name="가우시안 블러",
            description="커널 크기와 시그마 값을 사용하여 이미지에 가우시안 블러를 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="블러를 적용할 이미지"),
                IO.Int.Input("kernel_size", default=3, min=1, max=99, tooltip="블러 커널의 크기"),
                IO.Float.Input("sigma", default=1.00, min=0.00, max=200.00, step=0.01, tooltip="가우시안 블러의 시그마 값"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="블러가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, kernel_size, sigma) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = cv2.GaussianBlur(arr, (k, k), sigma)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(blurred)))
        
# -------------------------------

class IRL_MedianBlur(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_MedianBlur",
            display_name="미디언 블러",
            description="커널 크기를 사용하여 이미지에 미디언 블러를 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="블러를 적용할 이미지"),
                IO.Int.Input("kernel_size", default=3, min=1, max=99, tooltip="미디언 블러 커널의 크기"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="블러가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, kernel_size) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = cv2.medianBlur(arr, k)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(blurred)))

# -------------------------------

class IRL_BilateralFilter(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_BilateralFilter",
            display_name="양방향 필터",
            description="지름과 시그마 값을 사용하여 이미지에 양방향 필터를 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="필터를 적용할 이미지"),
                IO.Int.Input("diameter", default=9, min=1, max=50, tooltip="필터 커널의 지름"),
                IO.Float.Input("sigma_color", default=75.0, min=0.0, max=200.0, step=1.0, tooltip="색상 공간에서의 시그마 값"),
                IO.Float.Input("sigma_space", default=75.0, min=0.0, max=200.0, step=1.0, tooltip="좌표 공간에서의 시그마 값"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="필터가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, diameter, sigma_color, sigma_space) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        filtered = cv2.bilateralFilter(arr, diameter, sigma_color, sigma_space)
        return IO.NodeOutput(to_tensor_output(Image.fromarray(filtered)))

# -------------------------------
class IRL_Sharpen(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Sharpen",
            display_name="샤픈",
            description="조정 가능한 양으로 이미지를 선명하게 합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="샤픈을 적용할 이미지"),
                IO.Float.Input("amount", default=0.000, min=0.000, max=2.000, step=0.001, tooltip="샤픈 효과의 강도"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="샤픈이 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, amount) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)

        # 언샤프 마스크: 원본과 블러 이미지를 섞어서 샤픈
        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(arr, 1 + amount, blurred, -amount, 0)

        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        return IO.NodeOutput(to_tensor_output(Image.fromarray(sharpened)))

# -------------------------------
class IRL_HighPass(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_HighPass",
            display_name="하이패스 필터",
            description="에지와 세부 사항을 강조하기 위해 하이패스 필터를 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="필터를 적용할 이미지"),
                IO.Int.Input("radius", default=3, min=0, max=31, tooltip="하이패스 필터의 반경"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="하이패스 필터가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, radius) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)
        if radius <= 0:
            return IO.NodeOutput(image)

        k = radius if radius % 2 == 1 else radius + 1

        blurred = cv2.GaussianBlur(arr, (k, k), 0)
        highpass = arr - blurred + 128
        highpass = np.clip(highpass, 0, 255).astype(np.uint8)

        return IO.NodeOutput(to_tensor_output(Image.fromarray(highpass)))
        
# -------------------------------

FILTERS_NODE_CLASS_MAPPINGS = {
    "IRL_GaussianBlur": IRL_GaussianBlur,
    "IRL_MedianBlur": IRL_MedianBlur,
    "IRL_BilateralFilter": IRL_BilateralFilter,
    "IRL_Sharpen": IRL_Sharpen,
    "IRL_HighPass": IRL_HighPass,
}

FILTERS_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_GaussianBlur": "가우시안 블러",
    "IRL_MedianBlur": "미디언 블러",
    "IRL_BilateralFilter": "양방향 필터",
    "IRL_Sharpen": "샤픈",
    "IRL_HighPass": "하이패스 필터",
}

# -------------------------------