# -------------------------------
# IR Lite — AdjustmentsEX Nodes
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
import hashlib
import math
import re
import gc
from safetensors.torch import load_file, save_file, safe_open
import folder_paths
from typing import List, Dict

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

#----------------------------------------
# Header Utils
#----------------------------------------

def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]  
    return torch.from_numpy(arr)
    
def to_tensor_imgoutput(canvas: Image.Image) -> torch.Tensor:
    arr = np.array(canvas).astype(np.float32) / 255.0  # (H,W,C)
    if arr.ndim == 2:  # grayscale
        arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[...,:3]
    arr = arr.transpose(2,0,1)  # (C,H,W)
    arr = np.expand_dims(arr, axis=0)  # (1,C,H,W)
    return torch.from_numpy(arr).float()



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
    input->Torch float32 [0,1] tensor
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

def par_seed(seed_str: str) -> int:
    MAX_SEED = 2**64 - 1 


    if not seed_str:
        return 0 

    try:
        seed_val = int(seed_str)
        if 0 <= seed_val <= MAX_SEED:
            return seed_val

        seed_str = str(seed_val)
    except (ValueError, TypeError):
        seed_str = str(seed_str)

    hash_bytes = hashlib.sha256(seed_str.encode("utf-8")).digest()
    seed_val = int.from_bytes(hash_bytes[:8], "big")
    return seed_val


def resize_image(image_tensor, size):
    """
    image_tensor: torch.Tensor (batch, height, width, channels)
    size: (new_w, new_h) 
    """
    arr = to_numpy_image(image_tensor)
    new_w, new_h = size
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"잘못된 리사이즈 크기: {(new_w, new_h)}")
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return to_tensor_output(Image.fromarray(resized))


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

def get_mask_bbox(mask_tensor: torch.Tensor):
    """
    mask_tensor: shape (1,1,H,W) 또는 (H,W)
    return: (x_min, y_min, x_max, y_max)
    """
    if mask_tensor.dim() == 4:
        mask_tensor = mask_tensor.squeeze(0).squeeze(0)

    coords = torch.nonzero(mask_tensor > 0.5)
    if coords.numel() == 0:
        return None

    y_min = int(coords[:,0].min().item())
    y_max = int(coords[:,0].max().item())
    x_min = int(coords[:,1].min().item())
    x_max = int(coords[:,1].max().item())

    return (x_min, y_min, x_max, y_max)



def resize_mask_to_latent(mask, latent_shape=(64,64)):
    if mask is None:
        return None
    mask_resized = F.interpolate(mask, size=latent_shape, mode="nearest")
    return mask_resized.squeeze(0).squeeze(0)


def scale_bbox_to_latent(bbox, orig_size, latent_size=(64,64)):
    x_min, y_min, x_max, y_max = bbox
    H, W = orig_size
    h_lat, w_lat = latent_size
    return (
        int(x_min * w_lat / W),
        int(y_min * h_lat / H),
        int(x_max * w_lat / W),
        int(y_max * h_lat / H)
    )

def apply_mask_mode(mask_tensor: torch.Tensor, mask_set: str, mask_mode: str, target_size: tuple[int,int]) -> torch.Tensor:
    H, W = target_size


    if mask_set.lower() == "invert":
        mask_tensor = 1.0 - mask_tensor
    else:
        pass
        
    if mask_mode == "light_spread":
        mask_blr = F.max_pool2d(mask_tensor, 3, stride=1, padding=1)
    elif mask_mode == "small_spread":
        mask_blr = F.max_pool2d(mask_tensor, 5, stride=1, padding=2)
    elif mask_mode == "spread":
        mask_blr = F.max_pool2d(mask_tensor, 7, stride=1, padding=3)
    elif mask_mode == "big_spread":
        mask_blr = F.max_pool2d(mask_tensor, 9, stride=1, padding=4)
    elif mask_mode == "hard_spread":
        mask_blr = F.max_pool2d(mask_tensor, 11, stride=1, padding=5)
    elif mask_mode == "veryhard_spread":
        mask_blr = F.max_pool2d(mask_tensor, 13, stride=1, padding=6)
    elif mask_mode == "cutoff":
        mask_blr = F.max_pool2d(mask_tensor, 15, stride=1, padding=7)
    else:
        mask_blr = mask_tensor

    h, w = mask_blr.shape[2:]
        
    if h > H or w > W:
        mask_tensor = mask_blr[:, :, :H, :W]
    elif h < H or w < W:
        mask_tensor = F.interpolate(mask_blr, size=(H, W), mode="bilinear")
    else:
        mask_tensor = mask_blr
    mask_tensor = (mask_tensor > 0.5).float()

    return mask_tensor
    
def reblend_images(a: torch.Tensor, b: torch.Tensor, mask: torch.Tensor,
                         mode: str = "off", strength: float = 0.5) -> torch.Tensor:
    if mask is None or mode.lower() == "off":
        return a

    a_ch, b_ch, m_ch = a, b, mask

    if mode == "Blend":
        blended = (a_ch * (1.0 - strength) + b_ch * strength).clamp(0.0, 1.0)
    elif mode == "Overlay":
        blended = torch.where(a_ch < 0.5, 2 * a_ch * b_ch, 1 - 2 * (1 - a_ch) * (1 - b_ch))
    elif mode == "Add":
        blended = (a_ch + b_ch).clamp(0.0, 1.0)
    elif mode == "Multiply":
        blended = (a_ch * b_ch).clamp(0.0, 1.0)
    elif mode == "Difference":
        blended = torch.abs(a_ch - b_ch).clamp(0.0, 1.0)
    else:
        blended = a_ch

    inv_mask = torch.ones_like(m_ch) - m_ch
    result = m_ch * blended + inv_mask * a_ch

    return result

def encode_promptSamples(clip, prompt):
    if not prompt: return []
    try:
        match = re.match(r"\(([a-zA-Z0-9_-]+):([0-9.]+)\)", prompt.strip())
        if match:
            keyword, weight_str = match.groups()
            weight = float(weight_str)
            tokens = clip.tokenize(keyword)
            cond = clip.encode_from_tokens_scheduled(tokens)
            cond_scaled = [[item[0] * weight, item[1] if len(item) > 1 else {}] for item in cond]
            return cond_scaled
        else:
            tokens = clip.tokenize(prompt)
            return clip.encode_from_tokens_scheduled(tokens)
    except Exception as e:
        print(f"[IRL_ImgResampler_textencoding] 인코딩 실패: {e}")
        return []



def build_Posset_prompt(pos_text=None, quality=None):

    parts = []

    def add_part(value, skip_basic=False):
        if value and isinstance(value, str):
            val = value.strip()
            if val and (not skip_basic or val.lower() != "basic"):
                parts.append(val)

    if pos_text and pos_text.strip():
        pos_lines = [line.strip() for line in pos_text.splitlines() if line.strip()]
        parts.extend(pos_lines)
    else:
        parts.append("preserve style")

         
    add_part(quality, skip_basic=True)
    
    return ",".join(parts)

def build_negset_prompt(neg_text=None, bad_qual=None):

    parts = []

    def add_part(value, skip_basic=False):
        if value and isinstance(value, str):
            val = value.strip()
            if val and (not skip_basic or val.lower() != "basic"):
                parts.append(val)

    if neg_text and neg_text.strip():
        neg_lines = [line.strip() for line in neg_text.splitlines() if line.strip()]
        parts.extend(neg_lines)

    add_part(bad_qual, skip_basic=True)
    
    return ",".join(parts) if parts else ""

def apply_mask_mode_numpy(mask_arr: np.ndarray, mask_set: str, mask_mode: str, target_size: tuple[int,int]) -> np.ndarray:
    H, W = target_size

    # invert
    if mask_set.lower() == "invert":
        mask_arr = 1.0 - mask_arr

    # spread Set(OpenCV dilate)
    if mask_mode == "light_spread":
        kernel = np.ones((3,3), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
    elif mask_mode == "small_spread":
        kernel = np.ones((5,5), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
    elif mask_mode == "spread":
        kernel = np.ones((7,7), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
    elif mask_mode == "big_spread":
        kernel = np.ones((9,9), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
    elif mask_mode == "hard_spread":
        kernel = np.ones((11,11), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
    elif mask_mode == "veryhard_spread":
        kernel = np.ones((13,13), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
    elif mask_mode == "cutoff":
        kernel = np.ones((15,15), np.uint8)
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)

    # Resize
    mask_arr = cv2.resize(mask_arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)

    # Threshold to binary
    mask_arr = (mask_arr > 0).astype(np.uint8)

    # Expand to 1 channels (HxWx1)
    mask_arr = np.repeat(mask_arr[..., None], 1, axis=2)

    return mask_arr

lora_dir = folder_paths.get_folder_paths("loras")

#----------------------------------------
# Resampler-Average
#----------------------------------------


class IRL_ImgResampler(IO.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImgResampler",
            display_name="이미지 리샘플러",
            category="이미지 리파이너/이미지조정",
            description="이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다.\n"
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.\n"
                        "입력 이미지의 가로·세로 크기는 8의 배수 형태를 권장합니다.\n"
                        "(ex:512,512)",
            inputs=[
                IO.Model.Input("model", tooltip="참고할 모델"),
                IO.Clip.Input("clip", tooltip="참고할 clip"),
                IO.Vae.Input("vae", tooltip="참고할 vae 객체"),
                IO.Image.Input("image", tooltip="조정할 대상 이미지", optional=True),
                IO.Mask.Input("mask", tooltip="참고할 대상 마스크", optional=True),
                IO.Image.Input("re_sample_palete", tooltip="노이즈샘플링 추가용 샘플링 팔레트 이미지", optional=True),
                IO.Float.Input("noise_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="노이즈 강도. noise_set이 base_only이면 수치가 적용되지 않습니다."),
                IO.Float.Input("denoise", default=0.05, min=0.00, max=1.00, step=0.01,
                               tooltip="디노이즈 처리. noise_set이 base_only이면 절대 값을 0.41 이상 쓰면 안됩니다. 원본이미지를 날릴 가능성이 높습니다."),
                IO.String.Input("seedset", default=0, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Combo.Input("noise_mode", options=["normal", "Small_spread", "big_spread"], default="normal", tooltip="노이즈 방식. 블러를 주거나, 노이즈를 뿌려서 처리합니다.\n"
                               "normal일 경우 기본 노이즈로 처리합니다"),
                IO.Int.Input("steps", default=12, min=1, max=100, tooltip="디노이즈 스텝 수.\n"
                             "base_only + cpu 모드라면 스텝수는 20 이상을 적으면 안됩니다. base_only + cpu 모드 추천은 12 이하입니다."),
                IO.Float.Input("cfg", default=2.0, min=1.0, max=20.0, step=0.1, tooltip="CFG 스케일.\n"
                               "올릴수록 프롬프트 영향이 강해지고, 낮출수록 프롬프트 영향이 줄어듭니다."),
                IO.Float.Input("neg_str", default=0.01, min=0.01, max=0.50, step=0.01, tooltip="부정 조건 강도 페널티 스케일. 낮출수록 부정 텍스트의 영향을 낮춥니다.\n"
                               "noise_set이 base_only이면 값을 낮출수록 이미지 증발 현상이 줄어듭니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 방식"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러 방식"),
                IO.Combo.Input("quality", options=["basic", "high_resolution", "masterpiece"],
                               default="basic", tooltip="이미지 퀄리티 프리셋-긍정용. basic으로 두면 공백으로 넘깁니다.", optional=True),
                IO.String.Input("pos_text", multiline=True, tooltip="긍정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.Combo.Input("bad_qual", options=["bad_quality", "low_resolution", "basic"],
                               default="basic", tooltip="이미지 퀄리티 프리셋-부정용. basic으로 두면 공백으로 넘깁니다.", optional=True),
                IO.String.Input("neg_text", multiline=True, tooltip="부정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.Int.Input("latent_size_x", default=512, min=8, max=2048, tooltip="이미지를 넣지 않았을 시 사용되는 라텐트 캔버스.\n"
                             "이미지가 있을 경우는 무시됩니다. Y축과 맞춰주는게 좋습니다."),
                IO.Int.Input("latent_size_y", default=512, min=8, max=2048, tooltip="이미지를 넣지 않았을 시 사용되는 라텐트 캔버스.\n"
                             "이미지가 있을 경우는 무시됩니다. X축과 맞춰주는게 좋습니다."),
                IO.Int.Input("latent_batch", default=1, min=1, max=10, tooltip="잠재 이미지 생성 횟수. 생성 횟수가 많을수록 과부하가 걸릴 수 있습니다."),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="실행 장치"),
                IO.Combo.Input("clear_cache", options=["off", "on"], default="off", tooltip="남아있는 샘플링 잔여 기록을 정리합니다.\n"
                               "이미지가 붕괴되는게 개선되지 않을 경우 사용해보는걸 권장합니다."),
                IO.Combo.Input("noise_set", options=["base_only", "random_set"], default="base_only", tooltip="노이즈의 활성화 정도를 적용합니다.\n"
                               "base_only의 경우 원본이미지에서 살짝 변화를 주는 정도지만,\n"
                               "random_set 경우 큰 변화를 노릴 수 있으나 원본이미지의 보존이 까다롭습니다."),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, model, clip, vae, image=None, mask=None, re_sample_palete=None, noise_str=0.00, denoise=0.05, seedset=0, noise_mode="normal", 
                steps=12, cfg=2.0, neg_str=0.01, sampler_name="euler", scheduler="simple", quality="basic",
                pos_text=None, bad_qual="basic", neg_text=None, seed=0, latent_size_x=512, latent_size_y=512, latent_batch=1, device_set="cpu", clear_cache="off",noise_set="base_only") -> IO.NodeOutput:

        # Select Device

        if device_set == "cpu":
            device = "cpu"
        elif device_set == "nvidia":
            device = "cuda"
        elif device_set == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "rocm"
            else:
                device = "cpu"
        else:
            device = "cpu"

        if clear_cache.lower() == "on":
            if device_set in ["nvidia", "amd"]:
                try:
                    torch.cuda.empty_cache()   # GPU clear cache
                    gc.collect()               # CPU/Python clear cache
                    print("GPU/CPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패, CPU 캐시만 정리:", e)
                    gc.collect()
            else:
                gc.collect()                   # CPU clear cache
                print("CPU 캐시 초기화 완료")


        # Seed Settings
        parsed_seed = par_seed(seedset)

        if parsed_seed == 0:
            base_seed = random.randint(1, 2**31 - 1)
        else:
            base_seed = parsed_seed
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print("사용된 시드:", base_seed)

        # Latent Image Settings
        latent_image = None

        if image is not None:
            batch, h, w, c = image.shape
            new_h = max(64, (h // 64) * 64)
            new_w = max(64, (w // 64) * 64)
            if (new_h, new_w) != (h, w):
                image = resize_image(image, (new_w, new_h))

            # re_sample_palete Settings
            if re_sample_palete is not None:
                # mean correction to palette (RGB)

                mean_color = re_sample_palete.mean(dim=(0,1,2), keepdim=True)
                mean_orig = image.mean(dim=(0,1,2), keepdim=True)

                # Mean Correction(point Add)
                correction = mean_color - mean_orig
                corrected_image = image + correction
                corrected_image = torch.clamp(corrected_image, 0.0, 1.0)

                latent_image = vae.encode(corrected_image)

                # noise_set switch
                if noise_set == "base_only":
                    noise_base = torch.zeros_like(latent_image)
                elif noise_set == "random_set":
                    noise_base = torch.randn_like(latent_image, generator=generator) * noise_str
                else:
                    noise_base = torch.zeros_like(latent_image)

            else:
                # Not using palette mixing
                latent_image = vae.encode(image)
                if noise_set == "base_only":
                    noise_base = torch.zeros_like(latent_image)
                else:
                    noise_base = torch.randn_like(latent_image, generator=generator) * noise_str

        elif re_sample_palete is not None:
            # Only palette → palette latent to base
            palete_resized = resize_image(re_sample_palete, (latent_size_x, latent_size_y))
            latent_image = vae.encode(palete_resized)
            if noise_set == "base_only":
                noise_base = torch.zeros_like(latent_image)
            else:
                noise_base = torch.randn_like(latent_image, generator=generator) * noise_str

        else:
            # No base image & no palette mixing
            latent_image = torch.randn((latent_batch, 4, latent_size_y//8, latent_size_x//8),
                                       generator=generator, device=device)
            if noise_set == "base_only":
                noise_base = torch.zeros_like(latent_image)
            else:
                noise_base = torch.randn_like(latent_image, generator=generator) * noise_str

        # Noise Mask Settings
        if noise_mode.lower() == "normal":
            noise = noise_base
        elif noise_mode.lower() == "small_spread":
            noise = noise_base * 0.5
        elif noise_mode.lower() == "big_spread":
            noise = noise_base * 2.0
        else:
            noise = noise_base

        # Noise Mask
        noise_mask = None
        if mask is not None:
            mask_t = ensure_mask_tensor(mask)
            noise_mask = resize_mask_to_latent(mask_t, latent_shape=latent_image.shape[-2:])

        # Promot Encode (With Weight settings)
        Posset_prompt = build_Posset_prompt(pos_text, quality)
        positive = encode_promptSamples(clip, Posset_prompt)
        positive = conditioning_set_values(positive, {"strength": cfg})

        negative_keywords = []
        # neg_str Clampings (0.01 ~ 0.50)
        neg_str = max(0.01, min(neg_str, 0.50))
        negative_prompt = build_negset_prompt(neg_text, bad_qual)
        print("positive_prompt:", Posset_prompt)
        print("negative_prompt:", negative_prompt)

        negative = encode_promptSamples(clip, negative_prompt)
        negative = conditioning_set_values(negative, {"strength": cfg * neg_str})

        # Change Device
        latent_image = latent_image.to(device)
        noise = noise.to(device)
        positive = [p.to(device) if hasattr(p, "to") else p for p in positive]
        negative = [n.to(device) if hasattr(n, "to") else n for n in negative]

        # Sampling
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        latent_refined = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise, disable_noise=False, start_step=0, last_step=10000,
            force_full_denoise=True, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=base_seed
        )

        # Decode
        decoded = vae.decode(latent_refined)
        arr = to_numpy_image(decoded)

        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))
        
#----------------------------------------
# Resampler-Mixing
#----------------------------------------

class IRL_ImgResamplerMix(IO.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImgResamplerMix",
            display_name="이미지 리샘플러(믹스)",
            category="이미지 리파이너/이미지조정",
            description="이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다.\n"
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.\n"
                        "입력 이미지의 가로·세로 크기는 8의 배수 형태를 권장합니다.\n"
                        "(ex:512,512)",
            inputs=[
                IO.Model.Input("model", tooltip="참고할 모델"),
                IO.Clip.Input("clip", tooltip="참고할 clip"),
                IO.Vae.Input("vae", tooltip="참고할 vae 객체"),
                IO.Image.Input("image", tooltip="조정할 대상 이미지", optional=True),
                IO.Mask.Input("mask", tooltip="참고할 대상 마스크", optional=True),
                IO.Image.Input("re_sample_palete", tooltip="노이즈샘플링 추가용 샘플링 팔레트 이미지", optional=True),
                IO.Combo.Input("palete_inject", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"], default="0", 
                               tooltip="샘플링 팔레트 이미지 정보 주입량. 참고 색상량은 증가하지만,\n"
                               "과하면 원본이미지에 영향이 갈 수 있습니다."),
                IO.Float.Input("noise_str", default=0.05, min=0.00, max=1.00, step=0.01,
                               tooltip="노이즈 강도"),
                IO.Float.Input("denoise", default=0.10, min=0.00, max=1.00, step=0.01,
                               tooltip="디노이즈 처리"),
                IO.String.Input("seedset", default=0, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Combo.Input("noise_mode", options=["normal", "Small_spread", "big_spread"], default="normal", tooltip="노이즈 방식"),
                IO.Int.Input("steps", default=12, min=1, max=100, tooltip="디노이즈 스텝 수"),
                IO.Float.Input("cfg", default=2.0, min=1.0, max=20.0, step=0.1, tooltip="CFG 스케일"),
                IO.Float.Input("neg_str", default=0.01, min=0.01, max=0.50, step=0.01, tooltip="부정 조건 강도 페널티 스케일. 낮출수록 부정 텍스트의 영향을 낮춥니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 방식"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러 방식"),
                IO.String.Input("pos_text", multiline=True, default="illustration style, global illumination, sharp focus, vivid colors, color balanced",
                                 tooltip="긍정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.String.Input("neg_text", multiline=True, default="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)",
                                 tooltip="부정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.Int.Input("latent_size_x", default=512, min=8, max=2048, tooltip="이미지를 넣지 않았을 시 사용되는 라텐트 캔버스.\n"
                             "이미지가 있을 경우는 무시됩니다. Y축과 맞춰주는게 좋습니다."),
                IO.Int.Input("latent_size_y", default=512, min=8, max=2048, tooltip="이미지를 넣지 않았을 시 사용되는 라텐트 캔버스.\n"
                             "이미지가 있을 경우는 무시됩니다. X축과 맞춰주는게 좋습니다."),
                IO.Int.Input("latent_batch", default=1, min=1, max=10, tooltip="잠재 이미지 생성 횟수. 생성 횟수가 많을수록 과부하가 걸릴 수 있습니다."),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="실행 장치"),
                IO.Combo.Input("clear_cache", options=["off", "on"], default="off", tooltip="남아있는 샘플링 잔여 기록을 정리합니다.\n"
                               "이미지가 붕괴되는게 개선되지 않을 경우 사용해보는걸 권장합니다."),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, model, clip, vae, image=None, mask=None, re_sample_palete=None, palete_inject="0", noise_str=0.05, denoise=0.10, seedset=0, noise_mode="normal", 
                steps=12, cfg=2.0, neg_str=0.01, sampler_name="euler", scheduler="simple", pos_text="illustration style, global illumination, sharp focus, vivid colors, color balanced", 
                neg_text="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)", seed=0, latent_size_x=512, latent_size_y=512,
                latent_batch=1, device_set="cpu", clear_cache="off") -> IO.NodeOutput:

        # Select Device

        if device_set == "cpu":
            device = "cpu"
        elif device_set == "nvidia":
            device = "cuda"
        elif device_set == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "rocm"
            else:
                device = "cpu"
        else:
            device = "cpu"

        if clear_cache.lower() == "on":
            if device_set in ["nvidia", "amd"]:
                try:
                    torch.cuda.empty_cache()   # GPU clear cache
                    gc.collect()               # CPU/Python clear cache
                    print("GPU/CPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패, CPU 캐시만 정리:", e)
                    gc.collect()
            else:
                gc.collect()                   # CPU clear cache
                print("CPU 캐시 초기화 완료")


        # Seed Settings
        parsed_seed = par_seed(seedset)

        if parsed_seed == 0:
            base_seed = random.randint(1, 2**31 - 1)
        else:
            base_seed = parsed_seed
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print("사용된 시드:", base_seed)

        # Latent Image Settings
        latent_image = None

        if image is not None:
            batch, h, w, c = image.shape
            new_h = max(64, (h // 64) * 64)
            new_w = max(64, (w // 64) * 64)
            if (new_h, new_w) != (h, w):
                image = resize_image(image, (new_w, new_h))
            latent_image = vae.encode(image)
        else:
            latent_image = torch.randn((latent_batch, 4, latent_size_y//8, latent_size_x//8),
                                       generator=generator, device=device)

        # re_sample_palete Settings
        if re_sample_palete is not None:
            # Palette inject mapping
            
            palete_inject = int(palete_inject) * 0.05
            palete_inject = max(0.0, min(palete_inject, 0.70))

            if image is not None:
                palete_resized = resize_image(re_sample_palete, (new_w, new_h))
            else:
                # Not base image : latent size
                palete_resized = resize_image(
                    re_sample_palete,
                    (latent_size_x, latent_size_y)
                )

            
            # palete to latent Mixing(With Mask)
            palete_latent = vae.encode(palete_resized)
            palete_latent = palete_latent / (palete_latent.std() + 1e-6)
            base_noise = latent_image + palete_latent * palete_inject
            noise_base = torch.randn_like(base_noise) * noise_str

        else:
            # Not using palete
            noise_base = torch.randn_like(latent_image) * noise_str


        # Noise Mask Settings
        if noise_mode.lower() == "normal":
            noise = noise_base
        elif noise_mode.lower() == "small_spread":
            noise = noise_base * 0.5
        elif noise_mode.lower() == "big_spread":
            noise = noise_base * 2.0
        else:
            noise = noise_base

        # Noise Mask
        noise_mask = None
        if mask is not None:
            mask_t = ensure_mask_tensor(mask)
            noise_mask = resize_mask_to_latent(mask_t, latent_shape=latent_image.shape[-2:])

        # Promot Encode (With Weight settings)
        Posset_prompt =  ", ".join([line.strip() for line in pos_text.splitlines() if line.strip()])

        positive = encode_promptSamples(clip, Posset_prompt)
        positive = conditioning_set_values(positive, {"strength": cfg})

        negative_prompt = ", ".join([line.strip() for line in neg_text.splitlines() if line.strip()])

        print("positive_prompt:", Posset_prompt)
        print("negative_prompt:", negative_prompt)

        negative = encode_promptSamples(clip, negative_prompt)
        negative = conditioning_set_values(negative, {"strength": cfg * neg_str})

        # Change Device
        latent_image = latent_image.to(device)
        noise = noise.to(device)
        positive = [p.to(device) if hasattr(p, "to") else p for p in positive]
        negative = [n.to(device) if hasattr(n, "to") else n for n in negative]

        # Sampling
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        latent_refined = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise, disable_noise=False, start_step=0, last_step=10000,
            force_full_denoise=True, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=base_seed
        )

        # Decode
        decoded = vae.decode(latent_refined)
        arr = to_numpy_image(decoded)

        return IO.NodeOutput(to_tensor_output(Image.fromarray(arr)))
       
#----------------------------------------

class IRL_ImgResamplerAnd(IO.ComfyNode):
    
    @classmethod
    def define_schema(cls):

        files = []
        base_dirs = folder_paths.get_folder_paths("loras")
        for d in base_dirs:
            if not os.path.exists(d):
                continue
            for root, dirs, filenames in os.walk(d):
                for f in filenames:
                    if f.endswith(".safetensors"):
                        rel_path = os.path.relpath(os.path.join(root, f), d)
                        files.append(os.path.splitext(rel_path)[0])


        return IO.Schema(
            node_id="IRL_ImgResamplerAnd",
            display_name="이미지 리샘플러(로라로딩)",
            category="이미지 리파이너/이미지조정",
            description="이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다.\n"
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.\n"
                        "입력 이미지의 가로·세로 크기는 8의 배수 형태를 권장합니다.\n"
                        "(ex:512,512)",
            inputs=[
                IO.Model.Input("model", tooltip="참고할 모델"),
                IO.Clip.Input("clip", tooltip="참고할 clip"),
                IO.Vae.Input("vae", tooltip="참고할 vae 객체"),
                IO.Image.Input("image", tooltip="조정할 대상 이미지", optional=True),
                IO.Combo.Input("lora_name", options=files, default=files[0] if files else "", tooltip="적용할 LoRA 파일을 선택하세요"),
                IO.Float.Input("lora_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="로라 처리 강도"),
                IO.Float.Input("clip_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="클립 적용 강도"),
                IO.Int.Input("clip_skip", default=0, min=0, max=24, step=1,
                               tooltip="클립스킵횟수"),
                IO.Float.Input("denoise", default=0.05, min=0.00, max=1.00, step=0.01,
                               tooltip="디노이즈 처리. noise_set이 base_only이면 절대 값을 0.41 이상 쓰면 안됩니다. 원본이미지를 날릴 가능성이 높습니다."),
                IO.String.Input("seedset", default=0, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Combo.Input("noise_mode", options=["normal", "Small_spread", "big_spread"], default="normal", tooltip="노이즈 방식. 블러를 주거나, 노이즈를 뿌려서 처리합니다.\n"
                               "normal일 경우 기본 노이즈로 처리합니다"),
                IO.Int.Input("steps", default=12, min=1, max=100, tooltip="디노이즈 스텝 수.\n"
                             "base_only + cpu 모드라면 스텝수는 20 이상을 적으면 안됩니다. base_only + cpu 모드 추천은 12 이하입니다."),
                IO.Float.Input("cfg", default=2.0, min=1.0, max=20.0, step=0.1, tooltip="CFG 스케일.\n"
                               "올릴수록 프롬프트 영향이 강해지고, 낮출수록 프롬프트 영향이 줄어듭니다."),
                IO.Float.Input("neg_str", default=0.01, min=0.01, max=0.50, step=0.01, tooltip="부정 조건 강도 페널티 스케일. 낮출수록 부정 텍스트의 영향을 낮춥니다.\n"
                               "noise_set이 base_only이면 값을 낮출수록 이미지 증발 현상이 줄어듭니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 방식"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러 방식"),
                IO.String.Input("pos_text", multiline=True, default="illustration style, global illumination, sharp focus, vivid colors, color balanced",
                                 tooltip="긍정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.String.Input("neg_text", multiline=True, default="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)",
                                 tooltip="부정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.Int.Input("latent_size_x", default=512, min=8, max=2048, tooltip="이미지를 넣지 않았을 시 사용되는 라텐트 캔버스.\n"
                             "이미지가 있을 경우는 무시됩니다. Y축과 맞춰주는게 좋습니다."),
                IO.Int.Input("latent_size_y", default=512, min=8, max=2048, tooltip="이미지를 넣지 않았을 시 사용되는 라텐트 캔버스.\n"
                             "이미지가 있을 경우는 무시됩니다. X축과 맞춰주는게 좋습니다."),
                IO.Int.Input("latent_batch", default=1, min=1, max=10, tooltip="잠재 이미지 생성 횟수. 생성 횟수가 많을수록 과부하가 걸릴 수 있습니다."),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="실행 장치"),
                IO.Combo.Input("clear_cache", options=["off", "on"], default="off", tooltip="남아있는 샘플링 잔여 기록을 정리합니다.\n"
                               "이미지가 붕괴되는게 개선되지 않을 경우 사용해보는걸 권장합니다."),
                IO.Combo.Input("noise_set", options=["base_only", "random_set"], default="base_only", tooltip="노이즈의 활성화 정도를 적용합니다.\n"
                               "base_only의 경우 원본이미지에서 살짝 변화를 주는 정도지만,\n"
                               "random_set 경우 큰 변화를 노릴 수 있으나 원본이미지의 보존이 까다롭습니다."),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, model, clip, vae, image=None, lora_name="", lora_str=0.00, clip_str=0.00, clip_skip=0, denoise=0.05, seedset=0, noise_mode="normal", 
                steps=12, cfg=2.0, neg_str=0.01, sampler_name="euler", scheduler="simple", quality="basic", 
                pos_text="illustration style, global illumination, sharp focus, vivid colors, color balanced", 
                neg_text="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)", 
                seed=0, latent_size_x=512, latent_size_y=512, latent_batch=1, device_set="cpu", clear_cache="off",noise_set="base_only") -> IO.NodeOutput:

        # Select Device

        if device_set == "cpu":
            device = "cpu"
        elif device_set == "nvidia":
            device = "cuda"
        elif device_set == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "rocm"
            else:
                device = "cpu"
        else:
            device = "cpu"

        if clear_cache.lower() == "on":
            if device_set in ["nvidia", "amd"]:
                try:
                    torch.cuda.empty_cache()   # GPU clear cache
                    gc.collect()               # CPU/Python clear cache
                    print("GPU/CPU 캐시 초기화 완료")
                except Exception as e:
                    print("GPU 캐시 초기화 실패, CPU 캐시만 정리:", e)
                    gc.collect()
            else:
                gc.collect()                   # CPU clear cache
                print("CPU 캐시 초기화 완료")


        # Seed Settings
        parsed_seed = par_seed(seedset)

        if parsed_seed == 0:
            base_seed = random.randint(1, 2**31 - 1)
        else:
            base_seed = parsed_seed
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print("사용된 시드:", base_seed)
        
        #Clip_skip setting
        clip_skip = max(0, min(clip_skip, 24))
        if clip_skip > 0:
            clip = clip.clone()
            clip.clip_layer(-clip_skip)

        # LoRA activate
        if lora_name and (lora_str != 0.0 or clip_str != 0.0):
            lora_path = folder_paths.get_full_path_or_raise("loras", lora_name + ".safetensors")
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model, clip = comfy.sd.load_lora_for_models(model, clip, lora, lora_str, clip_str)

        # Latent Image Settings
        latent_image = None

        if image is not None:
            batch, h, w, c = image.shape
            new_h = max(64, (h // 64) * 64)
            new_w = max(64, (w // 64) * 64)
            if (new_h, new_w) != (h, w):
                image = resize_image(image, (new_w, new_h))


            latent_image = vae.encode(image)
            if noise_set == "base_only":
                noise_base = torch.zeros_like(latent_image)
            else:
                noise_base = torch.randn_like(latent_image, generator=generator)

        else:
            # No base image
            latent_image = torch.randn((latent_batch, 4, latent_size_y//8, latent_size_x//8),
                                       generator=generator, device=device)
            if noise_set == "base_only":
                noise_base = torch.zeros_like(latent_image)
            else:
                noise_base = torch.randn_like(latent_image, generator=generator)

        # Noise Mask Settings
        if noise_mode.lower() == "normal":
            noise = noise_base
        elif noise_mode.lower() == "small_spread":
            noise = noise_base * 0.5
        elif noise_mode.lower() == "big_spread":
            noise = noise_base * 2.0
        else:
            noise = noise_base
        
        noise_mask = None

        # Promot Encode (With Weight settings)
        Posset_prompt =  ", ".join([line.strip() for line in pos_text.splitlines() if line.strip()])

        positive = encode_promptSamples(clip, Posset_prompt)
        positive = conditioning_set_values(positive, {"strength": cfg})

        negative_prompt = ", ".join([line.strip() for line in neg_text.splitlines() if line.strip()])

        print("positive_prompt:", Posset_prompt)
        print("negative_prompt:", negative_prompt)

        negative = encode_promptSamples(clip, negative_prompt)
        negative = conditioning_set_values(negative, {"strength": cfg * neg_str})

        # Change Device
        latent_image = latent_image.to(device)
        noise = noise.to(device)
        positive = [p.to(device) if hasattr(p, "to") else p for p in positive]
        negative = [n.to(device) if hasattr(n, "to") else n for n in negative]

        # Sampling
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        latent_refined = comfy.sample.sample(
            model, noise, steps, cfg, sampler_name, scheduler,
            positive, negative, latent_image,
            denoise, disable_noise=False, start_step=0, last_step=10000,
            force_full_denoise=True, noise_mask=noise_mask,
            callback=callback, disable_pbar=disable_pbar, seed=base_seed
        )

        # Decode
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
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.\n"
                        "마스크 리블렌딩 모드와 샤프닝/히스토그램/유연화/밝기/대비/광원보정은 따로 적용됩니다.",
            inputs=[
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Mask.Input("mask", tooltip="참고할 대상 마스크", optional=True),
                IO.Combo.Input("re_blend_mode", options=["off", "Blend", "Overlay", "Add", "Multiply", "Difference"], default="off", tooltip="마스크 영역 재처리 방식"),
                IO.Float.Input("blend_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="마스크 리블렌딩 효과 강도"),
                IO.Combo.Input("mask_set", options=["off", "Normal", "invert"],
                               default="off", tooltip="마스크 적용 방식"),
                IO.Combo.Input("mask_mode", options=["off", "light_spread", "small_spread", "spread", "big_spread", "hard_spread", "veryhard_spread", "cutoff"],
                               default="off", tooltip="마스크 처리 방식"),
                IO.Float.Input("sharpen_strength", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="샤프닝 강도"),
                IO.Combo.Input("equalize_hist", options=["off", "equalize", "clahe"], default="off", tooltip="히스토그램 평활화 적용 여부"),
                IO.Float.Input("hist_strength", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="히스토그램 평활화 강도"),
                IO.Float.Input("color_str", default=0.00, min=0.00, max=2.00, step=0.01, tooltip="색상 강조"),
                IO.Float.Input("soften_strength", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="유연화 강도"),
                IO.Float.Input("line_strength", default=0.00, min=0.00, max=3.00, step=0.01, tooltip="라인 강조"),
                IO.String.Input("line_color", default="#000000", tooltip="라인 색상 (HEX 코드)"),
                IO.Float.Input("brightness_strength", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="밝기 조절 강도"),
                IO.Float.Input("contrast_strength", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="대비 조절 강도"),
                IO.Float.Input("light_balance", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="광원 보정 강도"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, mask=None, re_blend_mode="off", blend_str=0.00, mask_set="off", mask_mode="off", sharpen_strength=0.00, equalize_hist="off", hist_strength=0.00, 
                color_str=0.00, soften_strength=0.00, line_strength=0.00, line_color="#000000", brightness_strength=1.00, contrast_strength=1.00, light_balance=1.00) -> IO.NodeOutput:

        arr = ensure_image_tensor(image)
        H, W = arr.shape[2:]

        if mask is not None and re_blend_mode.lower() != "off":
            original = arr.clone()
            mask_arr = ensure_mask_tensor(mask)
            mask_arr = apply_mask_mode(mask_arr, mask_set, mask_mode, (H, W))
            arr = reblend_images(arr, original, mask_arr, re_blend_mode, blend_str)
        else:
            pass


        arr = arr[0].permute(1,2,0).cpu().numpy()
        
        arr = (arr * 255).clip(0,255).astype(np.uint8)

        base_edges = cv2.Canny(arr, 100, 200)
        base_hsv   = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        base_h, base_s, base_v = cv2.split(base_hsv)
        base_channels = cv2.split(arr)

        if sharpen_strength > 0.00:
            blur = cv2.GaussianBlur(arr, (5,5), 2)
            arr = cv2.addWeighted(arr, 1.00 + sharpen_strength, blur, -sharpen_strength, 0)

        if equalize_hist.lower() == "equalize":
            eq_channels = [cv2.equalizeHist(c) for c in base_channels]
            eq_arr = cv2.merge(eq_channels)
            arr = cv2.addWeighted(arr, 1.0 - hist_strength, eq_arr, hist_strength, 0)

        elif equalize_hist.lower() == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0 * max(hist_strength, 0.1), tileGridSize=(8,8))
            eq_channels = [clahe.apply(c) for c in base_channels]
            arr = cv2.merge(eq_channels)

        if color_str > 0.00:
            s = cv2.addWeighted(base_s, 1.0 + color_str, base_s, 0, 0)
            hsv = cv2.merge([base_h, s, base_v])
            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        if soften_strength > 0.00:
            blur = cv2.GaussianBlur(arr, (5,5), 2)
            arr = cv2.addWeighted(arr, 1.00 - soften_strength, blur, soften_strength, 0)

        if line_strength > 0.00:
            edges = cv2.Canny(arr, 150, 250)
            edges = cv2.GaussianBlur(edges, (3,3), 0)
            
            # HEX → RGB Trans
            hex_color = line_color.lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            edges_colored = np.zeros_like(arr)
            edges_colored[edges > 0] = rgb_color
            arr = cv2.addWeighted(arr, 1.0, edges_colored, line_strength, 0)


        if brightness_strength != 1.0:
            beta = (brightness_strength - 1.0) * 255.0
            arr = cv2.convertScaleAbs(arr, alpha=1.0, beta=beta)


        if contrast_strength != 1.0:
            alpha = contrast_strength
            arr = cv2.convertScaleAbs(arr, alpha=alpha, beta=0)

        if light_balance != 1.0:
            hsv = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            v = v * light_balance
            v = np.clip(v, 0, 255).astype(np.uint8)
            hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v])
            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
                   
        tensor_out = to_tensor_output(arr)
               
        return IO.NodeOutput(tensor_out)

#----------------------------------------
class IRL_NoiseCleaner(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_NoiseCleaner",
            display_name="노이즈 제거기",
            category="이미지 리파이너/이미지조정",
            description="이미지의 노이즈를 제거합니다. 마스크를 지정하면 해당 영역만 처리합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="노이즈 제거 대상 이미지"),
                IO.Mask.Input("mask", tooltip="노이즈 제거 영역 마스크", optional=True),
                IO.Combo.Input("mask_set", options=["off", "Normal", "invert"],
                               default="off", tooltip="마스크 적용 방식"),
                IO.Combo.Input("mask_mode", options=["off", "light_spread", "small_spread", "spread", "big_spread", "hard_spread", "veryhard_spread", "cutoff"],
                               default="off", tooltip="마스크 처리 방식"),
                IO.String.Input("seedset", tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Combo.Input("noise_level", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14"], default="0", 
                               tooltip="랜덤 노이즈 강도 단계. 값이 커질수록 주입 노이즈가 강해집니다."),
                IO.Float.Input("strength", default=0.5, min=0.0, max=1.0, step=0.05,
                               tooltip="노이즈 제거 강도"),
                IO.Float.Input("color_str", default=0.00, min=0.00, max=2.00, step=0.01,
                               tooltip="색상 강조"),
                IO.Float.Input("line_str", default=0.00, min=0.00, max=3.00, step=0.01,
                               tooltip="라인 강조"),
                IO.String.Input("line_color", default="#000000", tooltip="라인 색상 (HEX 코드)"),
                IO.Combo.Input("method", options=["gaussian", "median", "autoencoder"],
                               default="autoencoder", tooltip="노이즈 제거 방식"),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="실행 장치"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="노이즈 제거 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, mask=None, mask_set="off", mask_mode="off", seedset=0, noise_level="0", strength=0.5, color_str=0.00, line_str=0.00, line_color="#000000",
                method="autoencoder", device_set="cpu") -> IO.NodeOutput:
                    
        # Select Device

        if device_set == "cpu":
            device = "cpu"
        elif device_set == "nvidia":
            device = "cuda"
        elif device_set == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "rocm"
            else:
                device = "cpu"
        else:
            device = "cpu"

        # img to numpy
        arr = ensure_image_tensor(image)
        H, W = arr.shape[2:]
        print("Torch image shape:", arr.shape)
        
        arr = arr[0].permute(1,2,0).cpu().numpy()
        arr = (arr * 255).clip(0,255).astype(np.uint8)
        print("Numpy image shape:", arr.shape)
        original = arr.copy()
        
        mask_arr = None
        if mask is not None:
            mask_arr = ensure_mask_tensor(mask)
            print("Torch mask shape:", mask_arr.shape)
            mask_arr = mask_arr[0,0].cpu().numpy()
            mask_arr = apply_mask_mode_numpy(mask_arr, mask_set, mask_mode, (H, W))
            print("Numpy mask shape:", mask_arr.shape)

        # Seed Settings
        parsed_seed = par_seed(seedset)
        inject_noise = int(noise_level) * 0.05
        inject_noise = max(0.0, min(inject_noise, 0.70))

        if parsed_seed == 0:
            base_seed = random.randint(1, 2**31 - 1)
        else:
            base_seed = parsed_seed
        generator = torch.Generator(device=device).manual_seed(base_seed)
        print("사용된 시드:", base_seed)

        rng = np.random.default_rng(base_seed)
        noise = rng.normal(loc=0.0, scale=inject_noise * 255, size=arr.shape).astype(np.int16)
        noisy_arr = np.clip(arr.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        
        # Denoise Settings

        denoised = noisy_arr.copy()

        if method == "gaussian":
            ksize = max(1, int(strength * 5)) | 1
            denoised = cv2.GaussianBlur(noisy_arr, (ksize, ksize), strength * 2)

        elif method == "median":
            ksize = max(1, int(strength * 5)) | 1
            denoised = cv2.medianBlur(noisy_arr, ksize)

        elif method == "autoencoder":
            tmp = noisy_arr.astype(np.float32) / 255.0
            tmp = cv2.GaussianBlur(tmp, (5, 5), strength * 5)
            denoised = np.clip(tmp * 255.0, 0, 255).astype(np.uint8)

        processed = denoised.copy()

        # Color_Str
        if color_str > 0.0:
            hsv = cv2.cvtColor(processed, cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[...,1] = np.clip(hsv[...,1] * (1.0 + color_str), 0, 255)
            processed = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
            
        # Line_Str            
        if line_str > 0.0:
            edges = cv2.Canny(processed, 150, 250)
            edges = cv2.GaussianBlur(edges, (3,3), 0)
            
            # HEX → RGB Trans
            hex_color = line_color.lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))


            edges_colored = np.zeros_like(processed)
            edges_colored[edges > 0] = rgb_color  #Colored Edges
            processed = cv2.addWeighted(processed, 1.0, edges_colored, line_str, 0)

        # Mask area Fix
        if mask_arr is not None:
            arr = np.where(mask_arr > 0, processed, original)
        else:
            arr = processed

        if arr.ndim == 2:  # HxW → HxWx3
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 1:  # HxWx1 → HxWx3
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)

        # output
        
        tensor_out = to_tensor_output(arr)
        return IO.NodeOutput(tensor_out)

#----------------------------------------
SAMPLING_NODE_CLASS_MAPPINGS = {
    "IRL_ImgResampler": IRL_ImgResampler,
    "IRL_ImgResamplerMix": IRL_ImgResamplerMix,
    "IRL_ImgResamplerAnd": IRL_ImgResamplerAnd,
    "IRL_ImgDetailer": IRL_ImgDetailer,
    "IRL_NoiseCleaner": IRL_NoiseCleaner,
}

SAMPLING_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_ImgResampler": "이미지 리샘플러",
    "IRL_ImgResamplerMix": "이미지 리샘플러(믹스)",
    "IRL_ImgResamplerAnd": "이미지 리샘플러(로라로딩)",
    "IRL_ImgDetailer": "이미지 디테일러",
    "IRL_NoiseCleaner": "노이즈 제거기",
}

#----------------------------------------
