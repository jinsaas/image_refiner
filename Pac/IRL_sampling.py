# -------------------------------
# IR Lite — AdjustmentsEX Nodes
# (LOCALE-based multilingual description support included)
# -------------------------------

import os
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import cv2
from PIL import Image
from skimage import exposure
import random


# ComfyUI 최신 API
import comfy
from comfy_api.latest import IO, UI
from node_helpers import conditioning_set_values
import comfy.sample
import comfy.samplers
from comfy.samplers import KSampler
import comfy.utils
import latent_preview
import comfy.hooks
import comfy.context_windows
import comfy.cli_args
import nodes
import node_helpers
import comfy.model_management as model_management

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

def parse_seed(seed_str: str) -> int:
    MAX_SEED = 2**64 - 1  # 안전 범위 (4294967295)

    # None이나 빈 문자열 처리
    if not seed_str:
        return 0  # 랜덤 시드

    try:
        seed_val = int(seed_str)
        if 0 <= seed_val <= MAX_SEED:
            return seed_val
        # 범위 초과 → 해시 변환
        seed_str = str(seed_val)
    except (ValueError, TypeError):
        seed_str = str(seed_str)

    # SHA256 해시 → 64비트 정수 변환
    hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed_val = int.from_bytes(hash_bytes[:8], "big")  # 상위 8바이트 사용
    return seed_val


def resize_image(image_tensor, size):
    """
    image_tensor: torch.Tensor (batch, height, width, channels)
    size: (new_w, new_h) 튜플
    """
    arr = to_numpy_image(image_tensor)  # 텐서를 numpy로 변환
    new_w, new_h = size
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"잘못된 리사이즈 크기: {(new_w, new_h)}")
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return to_tensor_output(Image.fromarray(resized))


# -------------------------------
# Node definitions
# -------------------------------

class IRL_ImgResampler(IO.ComfyNode):
    
    @staticmethod
    def build_global_prompt(pos_text=None, quality=None):
        """
        주어진 텍스트 요소들을 조합하여 글로벌 프롬프트 문자열 생성.
        'basic' 값이나 None/빈 문자열은 제외.
        """
        parts = []

        def add_part(value, skip_basic=False):
            if value and isinstance(value, str):
                val = value.strip()
                if val and (not skip_basic or val.lower() != "basic"):
                    parts.append(val)

        if pos_text and pos_text.strip():
            parts.append(pos_text.strip())
        else:
            parts.append("preserve style")


        # quality 처리
        add_part(quality, skip_basic=True)


        return ",".join(parts)
        
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImgResampler",
            display_name="이미지 리샘플러",
            category="이미지 리파이너/이미지조정",
            description="이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다.\n"
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.",
            inputs=[
                IO.Model.Input("model", tooltip="참고할 모델"),
                IO.Clip.Input("clip", tooltip="참고할 clip"),
                IO.Vae.Input("vae", tooltip="참고할 vae 객체"),
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Float.Input("noise_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="노이즈 강도"),
                IO.Float.Input("denoise", default=0.30, min=0.00, max=1.00, step=0.01,
                               tooltip="디노이즈 처리"),
                IO.Combo.Input("noise_mode", options=["normal", "Small_spread", "big_spread"], default="normal", tooltip="노이즈 방식"),
                IO.Int.Input("steps", default=20, min=1, max=100, tooltip="디노이즈 스텝 수"),
                IO.Float.Input("cfg", default=7.0, min=1.0, max=20.0, step=0.1, tooltip="CFG 스케일"),
                IO.Combo.Input("sampler_name", options=["euler", "ddim", "dpmpp"], default="euler", tooltip="샘플러 방식"),
                IO.Combo.Input("scheduler", options=["normal", "karras"], default="normal", tooltip="스케줄러 방식"),
                IO.Combo.Input("quality", options=["basic", "high_resolution", "masterpiece"],
                               default="basic", tooltip="이미지 퀄리티 프리셋-긍정용", optional=True),
                IO.String.Input("pos_text", multiline=True, tooltip="긍정 프롬프트 텍스트", optional=True),
                IO.Combo.Input("bad_qual", options=["bad_quality", "low_resolution", "basic"],
                               default="basic", tooltip="이미지 퀄리티 프리셋-부정용", optional=True),
                IO.String.Input("neg_text", multiline=True, tooltip="부정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.Combo.Input("device_set", options=["cpu", "cuda", "rocm"], default="cpu", tooltip="실행 장치")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, model, clip, vae, image, noise_str=0.00, denoise=0.30, noise_mode="normal", 
                steps=20, cfg=7.0, sampler_name="euler", scheduler="normal", quality="basic",
                pos_text=None, bad_qual="basic", neg_text=None, seed=0, device_set="cpu") -> IO.NodeOutput:

        if device_set == "cpu":
            device = "cpu"
        elif device_set == "cuda":
            device = "cuda"
        elif device_set == "rocm":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "cuda"
            else:
                device = "cpu"  # fallback
        parsed_seed = parse_seed(seed)

        if parsed_seed == 0:
            base_seed = random.randint(1, 2**31 - 1)
        else:
            base_seed = parsed_seed
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print("사용된 시드:", base_seed)

        latent_image = None

        batch, h, w, c = image.shape

        new_h = max(64, (h // 64) * 64)
        new_w = max(64, (w // 64) * 64)

        if (new_h, new_w) != (h, w):
            image = resize_image(image, (new_w, new_h))

        latent = vae.encode(image)

        latent_image = latent


        if noise_mode.lower() == "normal":
            noise = torch.randn_like(latent) * noise_str
        elif noise_mode.lower() == "small_spread":
            noise = torch.randn_like(latent) * (noise_str * 0.5)
        elif noise_mode.lower() == "big_spread":
            noise = torch.randn_like(latent) * (noise_str * 2.0)
        else:
            noise = torch.zeros_like(latent)
            

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        global_prompt = cls.build_global_prompt(pos_text, quality)
        global_tokens = clip.tokenize(global_prompt)
        global_cond = clip.encode_from_tokens_scheduled(global_tokens)
        positive = conditioning_set_values(global_cond, {"strength": cfg})

        negative_keywords = []
        if bad_qual != "basic":
            negative_keywords.append(bad_qual)

        if neg_text and neg_text.strip():
            negative_keywords.append(neg_text.strip())

        negative_prompt = ", ".join(negative_keywords)
        print("global_prompt:", global_prompt)
        print("negative_prompt:", negative_prompt)

        negative_tokens = clip.tokenize(negative_prompt)
        negative_cond = clip.encode_from_tokens_scheduled(negative_tokens)
        negative = conditioning_set_values(negative_cond, {"strength": cfg * 0.5})

        latent_image = latent_image.to(device)
        noise = noise.to(device)
        positive = [p.to(device) if hasattr(p, "to") else p for p in positive]
        negative = [n.to(device) if hasattr(n, "to") else n for n in negative]
        latent_refined = comfy.sample.sample(model, noise, steps, cfg, sampler_name, scheduler, positive, negative, latent_image,
                                        denoise, disable_noise=False, start_step=0, last_step=10000, force_full_denoise=True,
                                        noise_mask=None, callback=callback, disable_pbar=False, seed=base_seed)

        
        decoded = vae.decode(latent_refined)

        arr = to_numpy_image(decoded)

        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))
        
# -------------------------------

class IRL_ImgDetailer(IO.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImgDetailer",
            display_name="이미지 디테일러",
            category="이미지 리파이너/이미지조정",
            description="이미지 재처리를 통해 품질 향상을 시도합니다.\n"
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.",
            inputs=[
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Float.Input("sharpen_strength", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="샤프닝 강도"),
                IO.Combo.Input("equalize_hist", options=["off", "equalize", "clahe"], default="off", tooltip="히스토그램 평활화 적용 여부"),
                IO.Float.Input("hist_strength", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="히스토그램 평활화 강도"),
                IO.Float.Input("color_str", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="색상 강조"),
                IO.Float.Input("soften_strength", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="유연화 강도"),
                IO.Float.Input("line_strength", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="라인 강조")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, sharpen_strength=0.00, equalize_hist="off", hist_strength=0.00, 
                color_str=0.00, soften_strength=0.00, line_strength=0.00) -> IO.NodeOutput:

                
        arr = to_numpy_image(image)
        channels = cv2.split(arr)
        hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        edges = cv2.Canny(arr, 100, 200)


        if sharpen_strength > 0.00:
            blur = cv2.GaussianBlur(arr, (5,5), 2)
            arr = cv2.addWeighted(arr, 1.00 + sharpen_strength, blur, -sharpen_strength, 0)

        if equalize_hist.lower() == "equalize":
            eq_channels = [cv2.equalizeHist(c) for c in channels]
            eq_arr = cv2.merge(eq_channels)
            arr = cv2.addWeighted(arr, 1.0 - hist_strength, eq_arr, hist_strength, 0)

        elif equalize_hist.lower() == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0 * max(hist_strength, 0.1), tileGridSize=(8,8))
            eq_channels = [clahe.apply(c) for c in channels]
            arr = cv2.merge(eq_channels)

        if color_str > 0.00:
            s = cv2.addWeighted(s, 1.0 + color_str, s, 0, 0)
            hsv = cv2.merge([h, s, v])
            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


        if soften_strength > 0.00:
            blur = cv2.GaussianBlur(arr, (5,5), 2)
            arr = cv2.addWeighted(arr, 1.00 - soften_strength, blur, soften_strength, 0)
            
        if line_strength > 0.00:
            edges = cv2.Canny(arr, 150, 250)
            edges = cv2.GaussianBlur(edges, (3,3), 0)
            edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2RGB)
            edges_colored = cv2.multiply(edges_colored, 0.5)
            arr = cv2.addWeighted(arr, 1.0, edges_colored, line_strength * 0.5, 0)

         
        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))
        
# -------------------------------
SAMPLING_NODE_CLASS_MAPPINGS = {
    "IRL_ImgResampler": IRL_ImgResampler,
    "IRL_ImgDetailer": IRL_ImgDetailer,
}

SAMPLING_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_ImgResampler": "이미지 리샘플러",
    "IRL_ImgDetailer": "이미지 디테일러",
}

# -------------------------------
