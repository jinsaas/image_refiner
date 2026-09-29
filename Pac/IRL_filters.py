# -------------------------------
# IR Lite — Filters Nodes
# -------------------------------

import numpy as np
import torch
import cv2
from PIL import Image, ImageDraw
from skimage import exposure

from comfy_api.latest import IO, UI

# ---------------------------------------
# Header Utils
#----------------------------------------

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

def apply_mask(mask, target_shape):

    mask_arr = to_numpy_mask(mask)
    mask_arr = cv2.resize(mask_arr, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_NEAREST) # Resize the mask to fit the image
    mask_arr = mask_arr.astype(np.float32) / 255.0        #Normalization
    mask_arr = cv2.cvtColor(mask_arr, cv2.COLOR_GRAY2BGR) # Gradient Mask
    return mask_arr

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
                IO.Mask.Input("mask", optional=True, tooltip="선택 영역에만 필터를 적용합니다. 연결하지 않으면 전체 이미지에 적용합니다."),
                IO.Int.Input("kernel_size", default=3, min=1, max=99, step=1, tooltip="블러 커널의 크기"),
                IO.Float.Input("sigma", default=1.00, min=0.00, max=200.00, step=0.01, tooltip="가우시안 블러의 시그마 값"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="블러가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, mask=None, kernel_size=3, sigma=1.00, preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = cv2.GaussianBlur(arr, (k, k), sigma)

        if mask is not None:
            mask_arr = apply_mask(mask, arr.shape[:2])  # Resize the mask to fit the image
            blurred = (blurred * mask_arr + arr * (1.0 - mask_arr)).astype(np.uint8)
        else:
            pass

        output = to_tensor_output(Image.fromarray(blurred))

        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))

        return IO.NodeOutput(output)

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
                IO.Mask.Input("mask", optional=True, tooltip="선택 영역에만 필터를 적용합니다. 연결하지 않으면 전체 이미지에 적용합니다."),
                IO.Int.Input("kernel_size", default=3, min=1, max=99, step=1, tooltip="미디언 블러 커널의 크기"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="블러가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, mask=None, kernel_size=3, preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        k = kernel_size if kernel_size % 2 == 1 else kernel_size + 1
        blurred = cv2.medianBlur(arr, k)

        if mask is not None:
            mask_arr = apply_mask(mask, arr.shape[:2])  # Resize the mask to fit the image
            blurred = (blurred * mask_arr + arr * (1.0 - mask_arr)).astype(np.uint8)
        else:
            pass

        output = to_tensor_output(Image.fromarray(blurred))

        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))

        return IO.NodeOutput(output)

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
                IO.Mask.Input("mask", optional=True, tooltip="선택 영역에만 필터를 적용합니다. 연결하지 않으면 전체 이미지에 적용합니다."),
                IO.Int.Input("diameter", default=9, min=1, max=50, step=1, tooltip="필터 커널의 지름"),
                IO.Float.Input("sigma_color", default=75.0, min=0.0, max=200.0, step=1.0, tooltip="색상 공간에서의 시그마 값"),
                IO.Float.Input("sigma_space", default=75.0, min=0.0, max=200.0, step=1.0, tooltip="좌표 공간에서의 시그마 값"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="필터가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, mask=None, diameter=9, sigma_color=75.0, sigma_space=75.0, preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        filtered = cv2.bilateralFilter(arr, diameter, sigma_color, sigma_space)

        if mask is not None:
            mask_arr = apply_mask(mask, arr.shape[:2])  # Resize the mask to fit the image
            filtered = (filtered * mask_arr + arr * (1.0 - mask_arr)).astype(np.uint8)
        else:
            pass

        output = to_tensor_output(Image.fromarray(filtered))

        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))

        return IO.NodeOutput(output)

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
                IO.Mask.Input("mask", optional=True, tooltip="선택 영역에만 필터를 적용합니다. 연결하지 않으면 전체 이미지에 적용합니다."),
                IO.Float.Input("amount", default=0.000, min=0.000, max=2.000, step=0.001, tooltip="샤픈 효과의 강도"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="샤픈이 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, mask=None, amount=0.000, preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)

        blurred = cv2.GaussianBlur(arr, (0, 0), sigmaX=3)
        sharpened = cv2.addWeighted(arr, 1 + amount, blurred, -amount, 0)
        sharpened = np.clip(sharpened, 0, 255).astype(np.uint8)

        if mask is not None:
            mask_arr = apply_mask(mask, arr.shape[:2])  # Resize the mask to fit the image
            sharpened = (sharpened * mask_arr + arr * (1.0 - mask_arr)).astype(np.uint8)
        else:
            pass

        output = to_tensor_output(Image.fromarray(sharpened))

        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))

        return IO.NodeOutput(output)

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
                IO.Mask.Input("mask", optional=True, tooltip="선택 영역에만 필터를 적용합니다. 연결하지 않으면 전체 이미지에 적용합니다."),
                IO.Int.Input("radius", default=3, min=0, max=31, step=1, tooltip="하이패스 필터의 반경"),
                IO.Float.Input("filter_str", default=1.00, min=0.00, max=2.00, step=0.01, tooltip="하이패스 필터 처리도"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="하이패스 필터가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, mask=None, radius=3, filter_str=1.00, preview_mode=False) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)
        if radius <= 0:
            return IO.NodeOutput(image)

        k = radius if radius % 2 == 1 else radius + 1

        blurred = cv2.GaussianBlur(arr, (k, k), 0)
        highpass = arr - blurred + 128
        highpass = 128 + (highpass - 128) * filter_str
        highpass = np.clip(highpass, 0, 255).astype(np.uint8)

        if mask is not None:
            mask_arr = apply_mask(mask, arr.shape[:2])  # Resize the mask to fit the image
            highpass = (highpass * mask_arr + arr * (1.0 - mask_arr)).astype(np.uint8)
        else:
            pass

        output = to_tensor_output(Image.fromarray(highpass))

        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))

        return IO.NodeOutput(output)

# -------------------------------

class IRL_MaskedFilter(IO.ComfyNode):

    @staticmethod
    def preprocess(mask, dilate_iter=0, blur_size=0):

        if dilate_iter > 0:
            kernel = np.ones((3,3), np.uint8)
            mask = cv2.dilate(mask, kernel, iterations=dilate_iter)

        if blur_size > 0:
            k = blur_size if blur_size % 2 == 1 else blur_size + 1
            mask = cv2.GaussianBlur(mask, (k, k), 0)

        return mask  # [H,W,3]

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_MaskedFilter",
            display_name="마스크드 필터링",
            description="이미지와 마스크를 받아 지정된 영역에 필터링을 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="필터를 적용할 이미지"),
                IO.Mask.Input("mask", tooltip="영역 마스크"),
                IO.Combo.Input("filter_mode", ["Gaussian","Median","Bilateral","Sharpen","HighPass"], default="Gaussian", tooltip="필터 처리"),
                IO.Int.Input("kernel", default=1, min=0, max=200, step=1, tooltip="반경/커널값."),
                IO.Int.Input("value1", default=1, min=0, max=200, step=1, tooltip="제어치 기준값"),
                IO.Int.Input("value2", default=1, min=0, max=200, step=1, tooltip="조정치 기준값"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="필터가 적용된 이미지"),
            ],
            category="이미지 리파이너/필터"
        )

    @classmethod
    def execute(cls, image, mask, filter_mode, kernel, value1, value2, preview_mode) -> IO.NodeOutput:

        # -------------------------------------------------
        # Value
        # -------------------------------------------------

        kernel = int(kernel)

        if kernel <= 0:
            return IO.NodeOutput(image)

        # -------------------------------------------------
        # Image
        # -------------------------------------------------

        original = to_numpy_image(image).astype(np.float32)

        # -------------------------------------------------
        # Mask
        # -------------------------------------------------

        H, W = original.shape[:2]

        base_mask = apply_mask(mask, (H, W))   # [H,W,3]


        # -------------------------------------------------
        # Filter
        # -------------------------------------------------

        if filter_mode == "Gaussian":
            k = kernel if kernel % 2 == 1 else kernel + 1
            b_val = value1/10
            if value2 > 0:
                if value2 > 15:
                    print (f"mask blur kernel vaiue limit over. set max value:15")
                    value2 = 15
                mask_arr = cls.preprocess(base_mask, blur_size=value2)
            else:
                mask_arr =  base_mask
            filtered = cv2.GaussianBlur(original, (k, k), b_val)

        elif filter_mode == "Median":
            k = kernel if kernel % 2 == 1 else kernel + 1

            if value1 == 0 and value2 == 0 :
                mask_arr =  base_mask
            else:
                if value1 > 10:
                    print (f"dilate strength vaiue limit over. set max value:10")
                    value1 = 10
                if value2 > 15:
                    print (f"mask blur kernel vaiue limit over. set max value:15")
                    value2 = 15
                mask_arr = cls.preprocess(base_mask, dilate_iter=value1, blur_size=value2)

            filtered = cv2.medianBlur(original.astype(np.uint8), k).astype(np.float32)

        elif filter_mode == "Sharpen":
            k = kernel if kernel % 2 == 1 else kernel + 1
            amount = min(2.0, value1 / 100.0)
            if value2 > 0:
                if value2 > 15:
                    print (f"mask blur kernel vaiue limit over. set max value:15")
                    value2 = 15
                mask_arr = cls.preprocess(base_mask, blur_size=value2)
            else:
                mask_arr =  base_mask
            blurred = cv2.GaussianBlur(original, (k, k), sigmaX=3)
            filtered = cv2.addWeighted(original, 1.0 + amount, blurred, -amount, 0)

        elif filter_mode == "HighPass":
            if kernel > 31:
                print (f"highpass kernel vaiue limit over. set max value:31")
                kernel = 31
            k = kernel if kernel % 2 == 1 else kernel + 1

            if value2 > 0:
                if value2 > 15:
                    print (f"mask blur kernel vaiue limit over. set max value:15")
                    value2 = 15
                mask_arr = cls.preprocess(base_mask, blur_size=value2)
            else:
                mask_arr =  base_mask

            blurred = cv2.GaussianBlur(original, (k, k), 0)
            filtered = original - blurred + 128
            scale = min(2.0, value1 / 100.0)
            filtered = 128 + (filtered - 128) * scale
            filtered = np.clip(filtered, 0, 255)

        elif filter_mode == "Bilateral":
            diameter = min(kernel, 15)
            sigmaColor = value1
            sigmaSpace = value2
            
            filtered = cv2.bilateralFilter(original.astype(np.uint8), diameter, sigmaColor, sigmaSpace).astype(np.float32)
            mask_arr = base_mask

        else:
            filtered = original.copy()
            mask_arr = base_mask

        # -------------------------------------------------
        # Mask Blend
        # -------------------------------------------------

        result = (original * (1.0 - mask_arr) + filtered * mask_arr)

        result = np.clip(result, 0, 255).astype(np.uint8)

        # -------------------------------------------------
        # Output
        # -------------------------------------------------

        output = to_tensor_output(Image.fromarray(result))

        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))

        return IO.NodeOutput(output)

# -------------------------------

FILTERS_NODE_CLASS_MAPPINGS = {
    "IRL_GaussianBlur": IRL_GaussianBlur,
    "IRL_MedianBlur": IRL_MedianBlur,
    "IRL_BilateralFilter": IRL_BilateralFilter,
    "IRL_Sharpen": IRL_Sharpen,
    "IRL_HighPass": IRL_HighPass,
    "IRL_MaskedFilter": IRL_MaskedFilter,
}

FILTERS_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_GaussianBlur": "가우시안 블러",
    "IRL_MedianBlur": "미디언 블러",
    "IRL_BilateralFilter": "양방향 필터",
    "IRL_Sharpen": "샤픈",
    "IRL_HighPass": "하이패스 필터",
    "IRL_MaskedFilter": "마스크드 필터링"
}

# -------------------------------
