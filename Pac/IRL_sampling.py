# -------------------------------
# IR Lite — Edit&Sampling Nodes
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
from . import IRL_noise

import comfy
from comfy_api.latest import IO, UI
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
import comfy.hooks
import comfy.context_windows
import comfy.cli_args
import nodes
import node_helpers
import comfy.model_management as model_management
from comfy_api.latest._io_public import ComfyTypeIO, comfytype, Custom

#----------------------------------------
# Header Utils
#----------------------------------------
RESET = "\033[0m"
BOLD = "\033[1m"
CYAN = "\033[36m"
YELLOW = "\033[33m"
GREEN = "\033[32m"


NoisePack = Custom("noise_pack")

@comfytype(io_type="noise_pack")
class NoisePack(ComfyTypeIO):
    Type = str

def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]  
    return torch.from_numpy(arr)
    
def to_tensor_imgoutput(canvas: Image.Image) -> torch.Tensor:
    if hasattr(canvas, "numpy"):
        arr = canvas.numpy()
    elif isinstance(canvas, np.ndarray):
        arr = canvas
    else:
        arr = np.array(canvas)

    arr = np.squeeze(arr)
    if arr.ndim == 4:
        arr = arr[0]

    arr = arr.astype(np.float32) / 255.0  # (H,W,C)
    if arr.ndim == 2:  # grayscale
        arr = np.expand_dims(arr, axis=-1)  # (H,W,1)
    if arr.shape[-1] == 4:  # RGBA → RGB
        arr = arr[...,:3]

    if arr.ndim == 3:
        arr = np.expand_dims(arr, axis=0)  # (1,H,W,C)

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

def to_numpy_image_out(image):
    if len(image.shape) == 5:
        decoded = image.reshape(-1, image.shape[-3], image.shape[-2], image.shape[-1])
        arr = to_numpy_image(decoded)
    else:
        arr = to_numpy_image(image)
    if isinstance(image, torch.Tensor):
        if arr.ndim == 4 and arr.shape[0] == 1:
            arr = arr[0]     # (1,C,H,W) → (C,H,W)
            if arr.shape[0] in (1,3):
                arr = np.transpose(arr, (1,2,0))  # (C,H,W) → (H,W,C)
        elif arr.ndim == 3 and arr.shape[0] in (1,3):
            arr = np.transpose(arr, (1,2,0))      # (C,H,W) → (H,W,C)

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
    MAX_SEED = 2**31 - 1


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
    seed_val =  max(0, min(int.from_bytes(hash_bytes[:8], "big"), MAX_SEED))
    return seed_val


def resize_image(image_tensor, size):
    """
    image_tensor: torch.Tensor (batch, height, width, channels)
    size: (new_w, new_h) 
    """
    arr = to_numpy_image(image_tensor)
    new_w, new_h = size
    if new_w <= 0 or new_h <= 0:
        raise ValueError(f"Wrong Resize Value: {(new_w, new_h)}")
    resized = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return to_tensor_output(Image.fromarray(resized))


def resize_latent_safe(latent, target_shape):
    if latent.dim() == 3:
        latent = latent.unsqueeze(1)
    if latent.dim() == 4: # base sd model VAE (N,C,H,W), check (H,W)
        return F.interpolate(latent, size=target_shape[-2:], mode="bilinear", align_corners=False)
    elif latent.dim() == 5: # FLOW model (N,C,D,H,W), check (D,H,W)
        return F.interpolate(latent, size=target_shape[-3:], mode="trilinear", align_corners=False)
    else:
        raise ValueError(f"Unexpected latent shape: {latent.shape}")

def standardize_latent(latent):
    return latent / (latent.std() + 1e-6)

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


def resize_mask_to_latent(mask, latent, ndim=None):

    if ndim is None:
        ndim = latent.ndim

    if mask is None:
        return None
    if ndim == 4:  # 2D latent (B,C,H,W)
        target_shape = latent.shape[2:]  # (H,W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0)  # (B,C,H,W)
        mask_resized = F.interpolate(mask, size=target_shape, mode="bilinear")
    elif ndim == 5:  # 3D latent (B,C,D,H,W)
        target_shape = latent.shape[2:]  # (D,H,W)
        if mask.ndim == 3:
            mask = mask.unsqueeze(0).unsqueeze(2)  # (B,C,1,H,W)
        elif mask.ndim == 4:
            mask = mask.unsqueeze(2)  # (B,C,1,H,W)
        mask_resized = F.interpolate(mask, size=target_shape, mode="trilinear")
    else:
        raise ValueError(f"Unsupported latent dimension: {ndim}")
    return mask_resized



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

def make_circular_kernel(ksize: int) -> torch.Tensor:

    center = ksize // 2
    y, x = torch.meshgrid(torch.arange(ksize), torch.arange(ksize), indexing="ij")
    dist = torch.sqrt((x - center)**2 + (y - center)**2)
    radius = ksize / 2.0
    kernel = (dist <= radius).float()
    return kernel

def apply_mask_mode(mask_tensor: torch.Tensor, mask_set: str, mask_mode: str, mask_style, target_size: tuple[int,int]) -> torch.Tensor:
    H, W = target_size


    if mask_set.lower() == "invert":
        mask_tensor = 1.0 - mask_tensor
    else:
        pass
    if mask_mode == "light_spread":
        ksize, padding = 3, 1
    elif mask_mode == "small_spread":
        ksize, padding = 5, 2
    elif mask_mode == "spread":
        ksize, padding = 7, 3
    elif mask_mode == "big_spread":
        ksize, padding = 9, 4
    elif mask_mode == "hard_spread":
        ksize, padding = 11, 5
    elif mask_mode == "veryhard_spread":
        ksize, padding = 13, 6
    elif mask_mode == "cutoff":
        ksize, padding = 15, 7
    else:
        ksize, padding = None, None

    if mask_style == "square" and ksize is not None:
        mask_blr = F.max_pool2d(mask_tensor, ksize, stride=1, padding=padding)
    elif mask_style == "circle" and ksize is not None:
        kernel = make_circular_kernel(ksize).to(mask_tensor.device)
        kernel = kernel.unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        mask_blr = F.conv2d(mask_tensor, kernel, padding=padding)
        mask_blr = (mask_blr > 0).float()
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

def build_conditioning(clip, text, weight, device):

    if not text or not str(text).strip():
        model = clip.patcher.model
    
        if hasattr(model, 'embed_tokens'):
            clip_dim = model.embed_tokens.weight.shape[1]
        elif hasattr(model, 'token_embedding'):
            clip_dim = model.token_embedding.weight.shape[1]
        elif hasattr(model, 'config') and hasattr(model.config, 'hidden_size'):
            clip_dim = model.config.hidden_size
        return [[torch.zeros((1, 77, clip_dim), device=device), {"pooled_output": torch.zeros((1, 1280), device=device), "weight": 0}]]

    tokens = clip.tokenize(text)
    
    cond = clip.encode_from_tokens_scheduled(tokens)

    cond_scaled = []
    for item in cond:
        tensor = item[0]
        cond_dict = item[1].copy() if len(item) > 1 else {}
        
        scaled_tensor = tensor * weight
        
        cond_dict["weight"] = weight

        if "pooled_output" not in cond_dict or cond_dict["pooled_output"] is None:
            cond_dict["pooled_output"] = torch.zeros((1, 1280), device=device)
            
        cond_scaled.append([scaled_tensor, cond_dict])
    
    return cond_scaled

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

def apply_feathering(mask_tensor: torch.Tensor, feather_size: int, feather_strength: float) -> torch.Tensor:
    if feather_size <= 0:
        return mask_tensor

    if mask_tensor.ndim == 3:
        mask_tensor = mask_tensor.unsqueeze(1)

    kernel_size = feather_size * 2 + 1
    sigma = feather_size / 2.0
    
    k = kernel_size // 2
    x = torch.arange(-k, k + 1, dtype=mask_tensor.dtype, device=mask_tensor.device)
    gauss = torch.exp(-(x**2) / (2 * sigma**2))
    gauss = gauss / gauss.sum()
    kernel2d = (gauss.unsqueeze(1) @ gauss.unsqueeze(0)).unsqueeze(0).unsqueeze(0)

    channels = mask_tensor.shape[1]
    kernel2d = kernel2d.repeat(channels, 1, 1, 1)

    # 4. apply Conv2d
    blurred = F.conv2d(mask_tensor, kernel2d, padding=k, groups=channels)
    
    # 5. Strength and Clamp
    if feather_strength != 1.0:
        blurred = torch.clamp(blurred * feather_strength, 0.0, 1.0)
        
    return blurred

def apply_mask_mode_numpy(mask_arr: np.ndarray, mask_set: str, mask_mode: str, target_size: tuple[int,int]) -> np.ndarray:
    H, W = target_size

    if mask_arr.ndim == 4:
        mask_arr = mask_arr.squeeze(0).squeeze(0)
    elif mask_arr.ndim == 3:
        mask_arr = mask_arr.squeeze(0)
    elif mask_arr.ndim == 2:
        pass
    else:
        raise ValueError(f"Unsupported mask shape: {mask_arr.shape}")

    # invert
    if mask_set.lower() == "invert":
        mask_arr = 1.0 - mask_arr

    mask_2d = mask_arr.squeeze() # (H, W)
    if mask_2d.max() <= 1: # 0~1 사이라면 255 곱하기
        mask_arr = (mask_2d * 255).astype(np.uint8)
    else:
        mask_arr = mask_2d.astype(np.uint8)

    # spread Set(OpenCV dilate)
    if mask_mode == "basic":
        mask_arr = (mask_arr > 0.5).astype(np.uint8)
    elif mask_mode == "light_spread":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_arr = cv2.erode(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (3, 3), 0)
    elif mask_mode == "small_spread":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_arr = cv2.erode(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (3, 3), 0)
    elif mask_mode == "spread":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3,3))
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (3, 3), 0)
    elif mask_mode == "big_spread":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (5, 5), 0)
    elif mask_mode == "hard_spread":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7,7))
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (7, 7), 0)
    elif mask_mode == "veryhard_spread":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9,9))
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (9, 9), 0)
    elif mask_mode == "cutoff":
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11,11))
        mask_arr = cv2.dilate(mask_arr.astype(np.uint8), kernel, iterations=1)
        mask_arr = cv2.GaussianBlur(mask_arr, (11, 11), 0)

    # Resize
    mask_arr = cv2.resize(mask_arr.astype(np.uint8), (W, H), interpolation=cv2.INTER_LINEAR)

    # Expand to 1 channels (HxWx1)
    mask_arr = mask_arr[..., None]

    return mask_arr

lora_dir = folder_paths.get_folder_paths("loras")

def get_equalized_sampled_colors(noise_map):

    noise_min = np.min(noise_map)
    noise_max = np.max(noise_map)
    
    if (noise_max - noise_min) > 1e-8:
        normalized = (noise_map - noise_min) / (noise_max - noise_min)
    else:
        normalized = np.zeros_like(noise_map)
    normalized_img = normalized * 255.0
    if normalized_img.ndim == 2:
        normalized_img = np.stack([normalized_img] * 3, axis=-1)
    return normalized.astype(np.float32)

def apply_blend(arr_small, arr_small_masked, palette_mode, palette_inject_val, mask_bool_small_3c):
    if arr_small_masked is None:
        return arr_small.copy()

    result = arr_small.copy()

    if palette_mode == "overlayblend":
        blended = cv2.addWeighted(arr_small, 1.0 - palette_inject_val,
                                  arr_small_masked, palette_inject_val, 0)
        result[mask_bool_small_3c] = blended[mask_bool_small_3c]

    elif palette_mode == "saturationblend":
        hsv_arr = cv2.cvtColor(arr_small, cv2.COLOR_BGR2HSV)
        hsv_pal = cv2.cvtColor(arr_small_masked, cv2.COLOR_BGR2HSV)

        hsv_arr[..., 1] = (1.0 - palette_inject_val) * hsv_arr[..., 1] + palette_inject_val * hsv_pal[..., 1]

        blended = cv2.cvtColor(hsv_arr, cv2.COLOR_HSV2BGR)
        result[mask_bool_small_3c] = blended[mask_bool_small_3c]

    elif palette_mode == "averageblend":
        blended = (arr_small + arr_small_masked) / 2.0
        result[mask_bool_small_3c] = blended[mask_bool_small_3c]

    elif palette_mode == "overwrite":
        weight_mask = min(1.0, palette_inject_val * 1.2)
        weight_orig = 1.0 - weight_mask
        blended = cv2.addWeighted(arr_small, weight_orig, arr_small_masked, weight_mask, 0)
        result[mask_bool_small_3c] = blended[mask_bool_small_3c]

    elif palette_mode == "color_pallete":
        result[mask_bool_small_3c] = arr_small_masked[mask_bool_small_3c]

    return np.clip(result, 0.0, 255.0).astype(np.float32)

def apply_noise_with_palette(arr_small, pal_arr_small, palette_mode, noise_style, palette_inject_val, inject_noise, noise_level, mask_bool_small_3c, rng):

    arr_f = arr_small.astype(np.float32)
    h, w, c = arr_f.shape

    if inject_noise <= 0.0:
        return arr_f.copy()

    blur_strength = int(noise_level) * 4 + 1  # (5, 9, 13, 17...)
    blurred_img = cv2.GaussianBlur(arr_f, (blur_strength, blur_strength), 0)


    colors = pal_arr_small.reshape(-1, 3).astype(np.float32)
    blank_color = colors[0].astype(np.float32)
    
    sampled_color = np.zeros_like(arr_f)
    sampled_color[:] = blank_color

    overlay_weight = min(1.0, inject_noise)
    guided_canvas = cv2.addWeighted(blurred_img, 1.0 - overlay_weight, sampled_color, overlay_weight, 0)

    arr_small_masked = arr_f.copy()
    arr_small_masked[mask_bool_small_3c] = guided_canvas[mask_bool_small_3c]

    return apply_blend(arr_f, arr_small_masked, palette_mode, palette_inject_val, mask_bool_small_3c)

def apply_noise_no_palette(arr_small, palette_mode, noise_style,
                           palette_inject_val, inject_noise,
                           noise_level, mask_bool_small_3c, rng):

    arr_f = arr_small.astype(np.float32)
    h, w, c = arr_f.shape

    if noise_style == "skipnoise" or inject_noise <= 0.0:
        return arr_f.copy()

    level = int(noise_level)
    if level <= 0:
        return arr_f.copy()

    blur_strength = int(noise_level) * 4 + 1  # (5, 9, 13, 17...)
    blurred_canvas = cv2.GaussianBlur(arr_f, (blur_strength, blur_strength), 0)

    overlay_weight = min(1.0, inject_noise)
    guided_canvas = cv2.addWeighted(arr_f, 1.0 - overlay_weight, blurred_canvas, overlay_weight, 0)

    arr_small_masked = arr_f.copy()
    arr_small_masked[mask_bool_small_3c] = guided_canvas[mask_bool_small_3c]

    return apply_blend(arr_f, arr_small_masked, palette_mode, palette_inject_val, mask_bool_small_3c)

def run_upscale_with_progress(upscale_model, in_img, tile=512, overlap=32):
    tile = int(512 / 4)      # 128
    overlap = int(32 / 4)      # 8

    # numpy(H, W, C) → torch(N, C, H, W)
    if isinstance(in_img, np.ndarray):
        in_img = torch.from_numpy(in_img).float() / 255.0
        in_img = in_img.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)

    steps = in_img.shape[0] * comfy.utils.get_tiled_scale_steps(
        in_img.shape[3], in_img.shape[2], tile_x=tile, tile_y=tile, overlap=overlap
    )
    pbar = comfy.utils.ProgressBar(steps)

    s = comfy.utils.tiled_scale(
        in_img,
        lambda a: upscale_model(a),
        tile_x=tile,
        tile_y=tile,
        overlap=overlap,
        upscale_amount=upscale_model.scale
,
        pbar=pbar
    )
    # numpy(H, W, C)
    inpainted = s[0].permute(1, 2, 0).cpu().numpy()  # (H, W, C)
    inpainted = (inpainted * 255).clip(0, 255).astype(np.uint8)

    return inpainted

def tiled_resize(arr, new_W, new_H, tile_size, overlap_size, interp=cv2.INTER_LANCZOS4):
    H, W, C = arr.shape
    out = np.zeros((new_H, new_W, C), dtype=np.uint8)

    # tile loop
    for y in range(0, H, tile_size - overlap_size):
        for x in range(0, W, tile_size - overlap_size):
            tile = arr[y:y+tile_size, x:x+tile_size]
            # tile rescale
            scale_y = new_H / H
            scale_x = new_W / W
            tile_resized = cv2.resize(tile, (int(tile.shape[1]*scale_x), int(tile.shape[0]*scale_y)), interpolation=interp)
            # composite (overlap blends)
            out_y = int(y * scale_y)
            out_x = int(x * scale_x)
            out[out_y:out_y+tile_resized.shape[0], out_x:out_x+tile_resized.shape[1]] = tile_resized
    return out

def adjust_sigma(base_sigma=0.15, strength=6):
    # strength: 1~11
    offset = strength - 6   # default:6
    return base_sigma + (0.01 * offset)

def apply_detail_enhance(img, strength):
    # reference value

    base_sigma_s = 10
    base_sigma_r = 0.15
    offset = int(strength) - 6   # 1~11, default:6
    sigma_s = base_sigma_s + (0.01 * offset)
    sigma_r = base_sigma_r + (0.01 * offset)
    return cv2.detailEnhance(img, sigma_s=sigma_s, sigma_r=sigma_r)

def apply_edge_filter(img, strength):
    # reference value

    base_sigma_s = 60
    base_sigma_r = 0.4
    offset = int(strength) - 6   # 1~11, default:6
    sigma_s = base_sigma_s + (0.01 * offset)
    sigma_r = base_sigma_r + (0.01 * offset)
    return cv2.edgePreservingFilter(img, flags=1, sigma_s=sigma_s, sigma_r=sigma_r)

def match_latent_size(tensor, target_shape):
    if tensor.ndim == 4:  # (B,C,H,W)
        if tensor.shape[2:] != target_shape:
            tensor = F.interpolate(
                tensor, size=target_shape, mode="bilinear", align_corners=False
            )
    elif tensor.ndim == 5:  # (B,C,D,H,W)
        if tensor.shape[2:] != target_shape:
            tensor = F.interpolate(
                tensor, size=target_shape, mode="trilinear", align_corners=False
            )
    else:
        raise ValueError(f"Unsupported tensor dimension: {tensor.ndim}")
    return tensor


def generate_hybrid_texture_noise_2d(width, height, scale=8.0, rng=None):
    if rng is None:
        rng = np.random.default_rng() 

    if scale <= 0:
        scale = 8.0

    # 1. 2D canvas
    grid_w = int(np.ceil(width / scale))
    grid_h = int(np.ceil(height / scale))

    # 2. random gradiant
    gradients = rng.normal(size=(grid_h + 1, grid_w + 1, 2)).astype(np.float32)
    gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8

    # 3. 2Dscale set
    X = np.arange(width, dtype=np.float32) / scale
    Y = np.arange(height, dtype=np.float32) / scale
    grid_x, grid_y = np.meshgrid(X, Y)

    # 4. pixel/frame index 
    xi = np.floor(grid_x).astype(np.int32)
    yi = np.floor(grid_y).astype(np.int32)

    # 5. vector calculate
    xf = grid_x - xi
    yf = grid_y - yi

    # clip index overflow defence
    xi0 = np.clip(xi, 0, grid_w)
    xi1 = np.clip(xi + 1, 0, grid_w)
    yi0 = np.clip(yi, 0, grid_h)
    yi1 = np.clip(yi + 1, 0, grid_h)

    # 6. safe index vector
    g00 = gradients[yi0, xi0]
    g10 = gradients[yi0, xi1]
    g01 = gradients[yi1, xi0]
    g11 = gradients[yi1, xi1]

    # 7. frequency modulation
    n00 = np.sin(xf * g00[..., 0] + yf * g00[..., 1])
    n10 = np.sin((xf - 1.0) * g10[..., 0] + yf * g10[..., 1])
    n01 = np.sin(xf * g01[..., 0] + (yf - 1.0) * g01[..., 1])
    n11 = np.sin((xf - 1.0) * g11[..., 0] + (yf - 1.0) * g11[..., 1])

    # 8. Fade effect
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(a, b, t): return a + t * (b - a)

    u = fade(xf)
    v = fade(yf)

    # 9. bilinear Interpolation
    x1 = lerp(n00, n10, u)
    x2 = lerp(n01, n11, u)
    noise_map = lerp(x1, x2, v)

    # normalize
    noise_map = noise_map - np.mean(noise_map)
    noise_std = np.std(noise_map)
    if noise_std > 1e-8:
        noise_map = noise_map / noise_std

    return noise_map.astype(np.float32)


def generate_hybrid_texture_noise_3d(width, height, depth, scale=8.0, rng=None):
    if rng is None:
        rng = np.random.default_rng()

    if scale <= 0:
        scale = 8.0

    # 1. 3D canvas
    grid_w = int(np.ceil(width / scale))
    grid_h = int(np.ceil(height / scale))
    grid_d = int(np.ceil(depth / scale))

    # 2. random gradiant
    gradients = rng.normal(size=(grid_d + 1, grid_h + 1, grid_w + 1, 3)).astype(np.float32)
    gradients /= np.linalg.norm(gradients, axis=-1, keepdims=True) + 1e-8

    # 3. 3Dscale set
    Z = np.arange(depth, dtype=np.float32) / scale
    Y = np.arange(height, dtype=np.float32) / scale
    X = np.arange(width, dtype=np.float32) / scale
    grid_d_coord, grid_y, grid_x = np.meshgrid(Z, Y, X, indexing='ij')

    # 4. pixel/frame index 
    xi = np.floor(grid_x).astype(np.int32)
    yi = np.floor(grid_y).astype(np.int32)
    zi = np.floor(grid_d_coord).astype(np.int32)

    # 5. vector calculate
    xf = grid_x - xi
    yf = grid_y - yi
    zf = grid_d_coord - zi

    # clip index overflow defence
    xi0 = np.clip(xi, 0, grid_w)
    xi1 = np.clip(xi + 1, 0, grid_w)
    yi0 = np.clip(yi, 0, grid_h)
    yi1 = np.clip(yi + 1, 0, grid_h)
    zi0 = np.clip(zi, 0, grid_d)
    zi1 = np.clip(zi + 1, 0, grid_d)

    # 6. safe index vector
    g000 = gradients[zi0, yi0, xi0]
    g100 = gradients[zi0, yi0, xi1]
    g010 = gradients[zi0, yi1, xi0]
    g110 = gradients[zi0, yi1, xi1]
    g001 = gradients[zi1, yi0, xi0]
    g101 = gradients[zi1, yi0, xi1]
    g011 = gradients[zi1, yi1, xi0]
    g111 = gradients[zi1, yi1, xi1]

    # 7. frequency modulation
    n000 = np.sin(xf * g000[..., 0] + yf * g000[..., 1] + zf * g000[..., 2])
    n100 = np.sin((xf - 1.0) * g100[..., 0] + yf * g100[..., 1] + zf * g100[..., 2])
    n010 = np.sin(xf * g010[..., 0] + (yf - 1.0) * g010[..., 1] + zf * g010[..., 2])
    n110 = np.sin((xf - 1.0) * g110[..., 0] + (yf - 1.0) * g110[..., 1] + zf * g110[..., 2])
    
    n001 = np.sin(xf * g001[..., 0] + yf * g001[..., 1] + (zf - 1.0) * g001[..., 2])
    n101 = np.sin((xf - 1.0) * g101[..., 0] + yf * g101[..., 1] + (zf - 1.0) * g101[..., 2])
    n011 = np.sin(xf * g011[..., 0] + (yf - 1.0) * g011[..., 1] + (zf - 1.0) * g011[..., 2])
    n111 = np.sin((xf - 1.0) * g111[..., 0] + (yf - 1.0) * g111[..., 1] + (zf - 1.0) * g111[..., 2])

    # 8. Fade effect
    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(a, b, t): return a + t * (b - a)

    u = fade(xf)
    v = fade(yf)
    w_fade = fade(zf)

    # Trilinear Interpolation
    x1_0 = lerp(n000, n100, u)
    x2_0 = lerp(n010, n110, u)
    y_0 = lerp(x1_0, x2_0, v)

    x1_1 = lerp(n001, n101, u)
    x2_1 = lerp(n011, n111, u)
    y_1 = lerp(x1_1, x2_1, v)

    noise_map = lerp(y_0, y_1, w_fade)

    # 9. normalize
    noise_map = noise_map - np.mean(noise_map)
    noise_std = np.std(noise_map)
    if noise_std > 1e-8:
        noise_map = noise_map / noise_std

    return noise_map.astype(np.float32)

def inject_noisemode_to_latent(latent_tensor, sigmas, noise_mode, noisepack, np_rng, device, seed):
    dims = latent_tensor.ndim
    orig_shape = latent_tensor.shape

    grid_scale = 4.0 if noise_mode == "small_spread" else (16.0 if noise_mode == "big_spread" else 8.0)
    sigma_scale = sigmas[0]
    noise_ratio = sigma_scale * 0.1

    if dims == 4:  # (B, C, H, W)
        batch_size, num_channels, h, w = orig_shape
        is_3d = False
    elif dims == 5:  # Flow (B, C, D, H, W)
        batch_size, num_channels, depth, h, w = orig_shape
        is_3d = True
    else:
        raise ValueError(f"Unsupported latent dimension reference: {dims}")

    channel_noises = []

    if noisepack == "generate_hybrid_texture_noise":
        if is_3d:
            batch_size, num_channels, depth, h, w = orig_shape
        
            for c in range(num_channels):
                texture_np = generate_hybrid_texture_noise_3d(w, h, depth, scale=grid_scale, rng=np_rng)
                texture_t = torch.from_numpy(texture_np).to(device=device, dtype=latent_tensor.dtype)
                texture_t = texture_t.unsqueeze(0).expand(batch_size, -1, -1, -1)
                channel_noises.append(texture_t)
        else:
            batch_size, num_channels, h, w = orig_shape
        
            for c in range(num_channels):
                texture_np = generate_hybrid_texture_noise_2d(w, h, scale=grid_scale, rng=np_rng)
                texture_t = torch.from_numpy(texture_np).to(device=device, dtype=latent_tensor.dtype)
                texture_t = texture_t.unsqueeze(0).expand(batch_size, -1, -1)
                channel_noises.append(texture_t)
            
        texture_tensor = torch.stack(channel_noises, dim=1)
    else:
        for c in range(num_channels):
            if noisepack == "SaltPepperNoise":
                texture_np = IRL_noise.SaltPepperNoise(w, h, grid_scale, seed, sigma_scale, rng=np_rng)
            elif noisepack == "PerlinNoise":
                texture_np = IRL_noise.PerlinNoise(w, h, grid_scale, seed, sigma_scale, rng=np_rng)
            elif noisepack == "RandomColor":
                texture_np = IRL_noise.RandomColor(w, h, grid_scale, seed, sigma_scale, rng=np_rng)
            elif noisepack == "WhiteNoise":
                texture_np = IRL_noise.WhiteNoise(w, h, grid_scale, seed, sigma_scale, rng=np_rng)
            else:
                texture_np = IRL_noise.GaussianNoise(w, h, grid_scale, seed, sigma_scale, rng=np_rng)

            texture_t = torch.from_numpy(texture_np).to(device=device, dtype=latent_tensor.dtype)
            if is_3d:
                if texture_t.ndim == 2:
                    texture_t = texture_t.unsqueeze(0).expand(depth, h, w)
                texture_t = texture_t.unsqueeze(0).expand(batch_size, depth, h, w)
            else:
                if texture_t.ndim == 2:
                    texture_t = texture_t.unsqueeze(0).expand(batch_size, h, w)
                    
            channel_noises.append(texture_t)
            
        texture_tensor = torch.stack(channel_noises, dim=1)
            
        texture_tensor = torch.stack(channel_noises, dim=1)

    texture_noise = (texture_tensor - texture_tensor.mean()) / (texture_tensor.std() + 1e-6)


    processed_latent = latent_tensor * (1.0 - noise_ratio) + texture_noise * noise_ratio

    return processed_latent

def inject_custom_noise_to_latent(latent_tensor, sigmas, noise_mode, np_rng, device):
    dims = latent_tensor.ndim
    orig_shape = latent_tensor.shape

    grid_scale = 4.0 if noise_mode == "small_spread" else (16.0 if noise_mode == "big_spread" else 8.0)
    sigma_scale = sigmas[0]
    noise_ratio = sigma_scale * 0.1

    channel_noises = []

    if dims == 4:  # (B, C, H, W)
        batch_size, num_channels, h, w = orig_shape
        
        for c in range(num_channels):
            texture_np = generate_hybrid_texture_noise_2d(w, h, scale=grid_scale, rng=np_rng)
            texture_t = torch.from_numpy(texture_np).to(device=device, dtype=latent_tensor.dtype)
            texture_t = texture_t.unsqueeze(0).expand(batch_size, -1, -1)
            channel_noises.append(texture_t)
            
        texture_tensor = torch.stack(channel_noises, dim=1)  # [B, C, H, W]

    elif dims == 5:  # Flow (B, C, D, H, W)
        batch_size, num_channels, depth, h, w = orig_shape
        
        for c in range(num_channels):
            texture_np = generate_hybrid_texture_noise_3d(w, h, depth, scale=grid_scale, rng=np_rng)
            texture_t = torch.from_numpy(texture_np).to(device=device, dtype=latent_tensor.dtype)
            texture_t = texture_t.unsqueeze(0).expand(batch_size, -1, -1, -1)
            channel_noises.append(texture_t)
            
        texture_tensor = torch.stack(channel_noises, dim=1)  # [B, C, D, H, W]
        
    else:
        raise ValueError(f"Unsupported latent dimension reference: {dims}")

    texture_noise = (texture_tensor - texture_tensor.mean()) / (texture_tensor.std() + 1e-6)

    processed_latent = latent_tensor * (1.0 - noise_ratio) + texture_noise * noise_ratio

    return processed_latent

def extract_palette_features(palette_image):

    img = palette_image.permute(0, 3, 1, 2) if palette_image.shape[-1] == 3 else palette_image
    
    kernel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32).view(1, 1, 3, 3).repeat(3, 1, 1, 1).to(img.device)
    edge_x = F.conv2d(img, kernel_x, padding=1)
    edges = torch.abs(edge_x).mean(dim=1, keepdim=True)
    
    colors = img
    
    brightness = 0.299 * img[:, 0, :, :] + 0.587 * img[:, 1, :, :] + 0.114 * img[:, 2, :, :]
    brightness = brightness.unsqueeze(1)
    
    return {
        "edges": edges,
        "colors": colors,
        "brightness": brightness
    }

def get_sigmas_weights(sigmas, start_sigma, end_sigma):
    weights = torch.zeros_like(sigmas)

    mask = (sigmas <= start_sigma) & (sigmas >= end_sigma)
    weights[mask] = 1.0
    
    return weights

def get_style_transfer_hooks(transfer_str, style_type, palette_features):
    mu_p = palette_features["mean"]
    sigma_p = palette_features["std"]
    palette_features = extract_palette_features(palette_image)
    def apply_style_hook(q, k, v, extra_options):
        step_idx = extra_options["transformer_options"].get("current_step", 0)
        current_weight = transfer_str[step_idx]
        
        if style_type == "Edge":
            return q * (1.0 + current_weight * palette_features["edges"]), k, v
        
        elif style_type == "Color":
            return q, k, v + (current_weight * palette_features["colors"])
            
        elif style_type == "Brightness":
            return q, k, v * (1.0 + current_weight * (palette_features["brightness"] - 1.0))

        elif style_type == "AdaIN":
            mu_curr = v.mean(dim=(1, 2), keepdim=True)
            sigma_curr = v.std(dim=(1, 2), keepdim=True) + 1e-6
            
            v_norm = (v - mu_curr) / sigma_curr
            v_ada = v_norm * (sigma_p.view(1, -1, 1, 1) * w + (1 - w)) + (mu_p.view(1, -1, 1, 1) * w + mu_curr * (1 - w))
            return q, k, v_ada
        return q, k, v

    return apply_style_hook

def build_style_transfer_model_options(weights, sigmas, extra_options=None):

    weight_map = weights.to(device=sigmas.device, dtype=torch.float32)
    
    def style_patch(args):
        sigma = args["sigma"]
        idx = torch.argmin(torch.abs(sigmas - sigma))
        
        current_weight = weight_map[idx]
        
        return args["cond_out"] * current_weight
    def input_patch(args):
        sigma = args["sigma"]
        idx = torch.argmin(torch.abs(sigmas - sigma))
        
        current_weight = weight_map[idx]
        return args["cond_out"] * (current_weight * 0.5)

    model_options = {
        "transformer_options": {
            "patches": {
                "middle_block": [style_patch], 
                "input_blocks": [input_patch]  
            }
        }
    }
    
    if extra_options:
        model_options["transformer_options"].update(extra_options)
        
    return model_options

def get_safe_conditioning_hooked(cond, device, model, cfg):
    if cond is None or (isinstance(cond, list) and len(cond) == 0):
        cond = model.cond_stage_model.get_empty_conditioning()

    updated_cond = node_helpers.conditioning_set_values(cond, {"strength": cfg})
    combined_hooks = HookGroup()
    for item in updated_cond:
        if "hooks" in item[1]:
            combined_hooks = combined_hooks.clone_and_combine(item[1]["hooks"])

    hooks = combined_hooks if len(combined_hooks) > 0 else None

    return updated_cond, hooks        

def build_model_options(pos_hook=None, neg_hook=None, extra_options=None):
    model_options = {"hooks": {}, "transformer_options": {}}
    if pos_hook: model_options["hooks"]["pos_conditioning"] = pos_hook
    if neg_hook: model_options["hooks"]["neg_conditioning"] = neg_hook
    if extra_options and isinstance(extra_options, dict):
        model_options["transformer_options"].update(extra_options)
    return model_options

def get_sigmas(model, scheduler, steps, denoise, device="cpu"):

    model_sampling = model.get_model_object("model_sampling")
    
    if denoise < 1.0:
        new_steps = int(steps / denoise)
        sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, new_steps).to(device)
        return sigmas[-(steps + 1):]
    
    sigmas = comfy.samplers.calculate_sigmas(model_sampling, scheduler, steps).to(device)
    return sigmas


def coordinate_latent_amplitude(original_latent, processed_latent, max_amplitude_limit=4.0):
    if torch.std(processed_latent).item() == 0.0:
        return original_latent.clone()

    # copy mean for channels (B, C, ...)
    mean = torch.mean(processed_latent, dim=list(range(2, processed_latent.ndim)), keepdim=True)

    # amplitude clamp
    centered = processed_latent - mean
    centered = torch.tanh(centered / max_amplitude_limit) * max_amplitude_limit

    # mean rematch
    processed_latent = centered + mean

    return processed_latent


def mask_type_check(mask, device):

    if mask is not None:
        if isinstance(mask, dict):
            if "latent_mask" in mask:
                mask = mask["latent_mask"]
            elif "noise_mask" in mask:
                mask = mask["noise_mask"]
            elif "mask" in mask:
                mask = mask["mask"]

        if mask.dtype == torch.bool:
            mask = mask.float()

        return mask.to(device)

    else:
        return None

def noise_str_control(noise_str, noise_burn_guard):
    
    noise_strength = float(noise_str)
    guard_level = int(noise_burn_guard)
    guard_mapping = {
        1: 1.0,
        2: 0.85,
        3: 0.70,
        4: 0.55,
        5: 0.40
    }
    guard_burn = guard_mapping.get(guard_level, 1.0)
    gate_noise = noise_strength * guard_burn
    return gate_noise

def generate_auto_mask(arr_bgr):
    # --- [Tier 1] sd 1.5 pixel collapse ---
    hsv = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    
    # sat >=20 : low saturation collapse
    _, sat_low_mask = cv2.threshold(saturation, 20, 255, cv2.THRESH_BINARY_INV)
    
    # grayscale collapse
    gray = cv2.cvtColor(arr_bgr, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)
    diff = cv2.absdiff(gray, blurred)
    _, texture_broken_mask = cv2.threshold(diff, 10, 255, cv2.THRESH_BINARY_INV)

    # --- [Tier 2] overtue or noise burn ---
    s_channel = hsv[:, :, 1]
    v_channel = hsv[:, :, 2]

    _, s_burn_mask = cv2.threshold(s_channel, 230, 255, cv2.THRESH_BINARY) # over saturation
    _, v_burn_mask = cv2.threshold(v_channel, 240, 255, cv2.THRESH_BINARY) # noise burn
    burn_mask = cv2.bitwise_or(s_burn_mask, v_burn_mask)

    # --- [composite] ---
    combined_mask = cv2.bitwise_or(sat_low_mask, texture_broken_mask)
    combined_mask = cv2.bitwise_or(combined_mask, burn_mask)
    
    # dilated_mask (iterations=3)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    dilated_mask = cv2.dilate(combined_mask, kernel, iterations=3)
    
    # GaussianBlur mask
    auto_mask = cv2.GaussianBlur(dilated_mask, (11, 11), 0)
    
    return auto_mask



def blend_latent_with_mask(latent_image, noise, noise_mask, noise_str):

    erased_latent = latent_image * (1.0 - noise_str)
    target_noise = noise * noise_str
    blended_canvas = erased_latent + target_noise

    processed_latent = blended_canvas * noise_mask + latent_image * (1.0 - noise_mask)
    
    return processed_latent

def prepare_mask_for_opencv(mask_arr: np.ndarray) -> np.ndarray:

    # 1. (0.0 ~ 1.0) float, (0 ~ 255)uint8
    if mask_arr.max() <= 1.0:
        mask_arr = (mask_arr * 255).astype(np.uint8)
    else:
        mask_arr = mask_arr.astype(np.uint8)
    
    # 2. threshold (-> fmm, telea iogic mask)
    _, hard_mask = cv2.threshold(mask_arr, 127, 255, cv2.THRESH_BINARY)
    
    return mask_arr, hard_mask



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

def run_vector_resize(image_np, new_w, new_h, resize_mode="pixelbox"):
    # image_np: numpy array [H,W] or [H,W,3]
    interp_map = {
        "nearest": cv2.INTER_NEAREST,
        "bilinear": cv2.INTER_LINEAR,
        "bicubic": cv2.INTER_CUBIC,
        "lanczos": cv2.INTER_LANCZOS4,
        "pixelbox": cv2.INTER_AREA
    }

    base_resized = cv2.resize(image_np, (new_w, new_h), interpolation=interp_map[resize_mode])

    if image_np.ndim == 3:
        gray = cv2.cvtColor(image_np, cv2.COLOR_RGB2GRAY)
    else:
        gray = image_np
    contours = image_to_vector(image_np)
    valid_contours = [c for c in contours if len(c) > 10]

    if valid_contours:
        scaled_contours = resize_vector(valid_contours, new_w, new_h, image_np.shape[1], image_np.shape[0])
        image_resized_np = vector_to_image(scaled_contours, new_w, new_h,
                                           base_image=base_resized,
                                           draw_color=(255,255,255),
                                           thickness=1)
    else:
        edges = cv2.Canny(gray, 50, 150)
        edgeline = cv2.resize(edges, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        image_resized_np = cv2.bitwise_or(base_resized, edgeline)

    return image_resized_np

#----------------------------------------
# image Enhancer
#----------------------------------------

class IRL_ColorTransfer(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ColorTransfer",
            display_name="컬러 트랜스퍼",
            category="이미지 리파이너/이미지조정",
            description="참고 이미지를 기반으로 색상 통계를 맞춰 컬러 트랜스퍼를 수행합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.Image.Input("samp_image", tooltip="참고 이미지"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="컬러 트랜스퍼 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, samp_image) -> IO.NodeOutput:
        arr = ensure_image_tensor(image)[0].permute(1,2,0).cpu().numpy()
        arr = (arr * 255).clip(0,255).astype(np.uint8)

        samp_arr = ensure_image_tensor(samp_image)[0].permute(1,2,0).cpu().numpy()
        samp_arr = (samp_arr * 255).clip(0,255).astype(np.uint8)

        # Lab color area
        arr_lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
        samp_lab = cv2.cvtColor(samp_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

        # Reinhard Color Transfer
        for i in range(3):  # L, a, b channels
            arr_mean, arr_std   = arr_lab[:,:,i].mean(), arr_lab[:,:,i].std()
            samp_mean, samp_std = samp_lab[:,:,i].mean(), samp_lab[:,:,i].std()
            arr_lab[:,:,i] = (arr_lab[:,:,i] - arr_mean) * (samp_std / (arr_std+1e-5)) + samp_mean

        arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
        arr = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2RGB)

        tensor_out = to_tensor_output(arr)
        return IO.NodeOutput(tensor_out)

# ---------------------------------------------------------------------------------


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
                IO.Image.Input("samp_image", tooltip="참고 이미지", optional=True),
                IO.Combo.Input("re_blend_mode", options=["off", "Blend", "Overlay", "Add", "Multiply", "Difference"], default="off", tooltip="마스크 영역 재처리 방식"),
                IO.Float.Input("blend_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="마스크 리블렌딩 효과 강도"),
                IO.Combo.Input("mask_set", options=["off", "Normal", "invert"],
                               default="off", tooltip="마스크 적용 방식"),
                IO.Combo.Input("mask_mode", options=["off", "light_spread", "small_spread", "spread", "big_spread", "hard_spread", "veryhard_spread", "cutoff"],
                               default="off", tooltip="마스크 처리 방식"),
                IO.Combo.Input("mask_style", options=["off", "square", "circle"],
                               default="off", tooltip="마스크 스타일"),
                IO.Float.Input("sharpen_strength", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="샤프닝 강도"),
                IO.Combo.Input("equalize_hist", options=["off", "equalize", "clahe"], default="off", tooltip="히스토그램 평활화 적용 여부"),
                IO.Float.Input("hist_strength", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="히스토그램 평활화 강도"),
                IO.Float.Input("color_str", default=0.00, min=0.00, max=2.00, step=0.01, tooltip="색상 강조"),
                IO.Float.Input("soften_strength", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="유연화 강도"),
                IO.Float.Input("line_strength", default=0.00, min=0.00, max=2.00, step=0.01, tooltip="라인 강조"),
                IO.Combo.Input("color_mode", options=["off", "transfer"], default="off", tooltip="색상 정렬"),
                IO.String.Input("line_color", default="#000000", tooltip="라인 색상 (HEX 코드)"),
                IO.Float.Input("brightness_strength", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="밝기 조절 강도"),
                IO.Float.Input("contrast_strength", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="대비 조절 강도"),
                IO.Float.Input("light_balance", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="광원 보정 강도"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, image, mask=None, samp_image=None, re_blend_mode="off", blend_str=0.00, mask_set="off", mask_mode="off", mask_style="off", sharpen_strength=0.00, equalize_hist="off", hist_strength=0.00, 
                color_mode="off", color_str=0.00, soften_strength=0.00, line_strength=0.00, line_color="#000000", brightness_strength=1.00, contrast_strength=1.00, light_balance=1.00, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            current_device = model_management.get_torch_device()
            if current_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete") 

        arr = ensure_image_tensor(image)
        H, W = arr.shape[2:]

        if mask is not None and re_blend_mode.lower() != "off":
            original = arr.clone()
            mask_arr = ensure_mask_tensor(mask)
            mask_arr = apply_mask_mode(mask_arr, mask_set, mask_mode, mask_style, (H, W))
            arr = reblend_images(arr, original, mask_arr, re_blend_mode, blend_str)
        else:
            pass

        arr = arr[0].permute(1,2,0).cpu().numpy()
        
        arr = (arr * 255).clip(0,255).astype(np.uint8)
        
        if samp_image is not None and color_mode.lower() == "transfer":
            samp_arr = ensure_image_tensor(samp_image)[0].permute(1,2,0).cpu().numpy()
            samp_arr = (samp_arr * 255).clip(0,255).astype(np.uint8)

            # Lab color area
            arr_lab  = cv2.cvtColor(arr, cv2.COLOR_RGB2LAB).astype(np.float32)
            samp_lab = cv2.cvtColor(samp_arr, cv2.COLOR_RGB2LAB).astype(np.float32)

            # Reinhard Color Transfer
            for i in range(3):  # L, a, b channels
                arr_mean, arr_std   = arr_lab[:,:,i].mean(), arr_lab[:,:,i].std()
                samp_mean, samp_std = samp_lab[:,:,i].mean(), samp_lab[:,:,i].std()
                arr_lab[:,:,i] = (arr_lab[:,:,i] - arr_mean) * (samp_std / (arr_std+1e-5)) + samp_mean

            arr_lab = np.clip(arr_lab, 0, 255).astype(np.uint8)
            arr = cv2.cvtColor(arr_lab, cv2.COLOR_LAB2RGB)

        else:
            pass

        base_edges = cv2.Canny(arr, 100, 200)
        base_hsv   = cv2.cvtColor(arr, cv2.COLOR_RGB2HSV)
        base_h, base_s, base_v = cv2.split(base_hsv)
        base_channels = cv2.split(arr)

        sharpen_strength = float(max(0.0, min(sharpen_strength, 1.0)))
        if sharpen_strength > 0.00:
            blur = cv2.GaussianBlur(arr, (5,5), 2)
            arr = cv2.addWeighted(arr, 1.00 + sharpen_strength, blur, -sharpen_strength, 0)

        hist_strength = float(max(0.0, min(hist_strength, 1.0)))
        if equalize_hist.lower() == "equalize":
            eq_channels = [cv2.equalizeHist(c) for c in base_channels]
            eq_arr = cv2.merge(eq_channels)
            arr = cv2.addWeighted(arr, 1.0 - hist_strength, eq_arr, hist_strength, 0)

        elif equalize_hist.lower() == "clahe":
            clahe = cv2.createCLAHE(clipLimit=2.0 * max(hist_strength, 0.1), tileGridSize=(8,8))
            eq_channels = [clahe.apply(c) for c in base_channels]
            arr = cv2.merge(eq_channels)

        color_str = float(max(0.0, min(color_str, 1.0)))
        if color_str > 0.00:
            
            s = cv2.addWeighted(base_s, 1.0 + color_str, base_s, 0, 0)
            hsv = cv2.merge([base_h, s, base_v])
            arr = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        soften_strength = float(max(0.0, min(soften_strength, 1.0)))
        if soften_strength > 0.00:
            blur = cv2.GaussianBlur(arr, (5,5), 2)
            arr = cv2.addWeighted(arr, 1.00 - soften_strength, blur, soften_strength, 0)

        line_strength = float(max(0.0, min(line_strength, 1.0)))
        if line_strength > 0.00:
            edges = cv2.Canny(arr, 150, 250)
            edges = cv2.GaussianBlur(edges, (3,3), 0)
            
            # HEX → RGB Trans
            hex_color = line_color.lstrip('#')
            rgb_color = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))

            edges_colored = np.zeros_like(arr)
            edges_colored[edges > 0] = rgb_color
            arr = cv2.addWeighted(arr, 1.0, edges_colored, line_strength, 0)

        brightness_strength = float(max(0.0, min(brightness_strength, 2.0)))
        if brightness_strength != 1.0:
            beta = (brightness_strength - 1.0) * 255.0
            arr = cv2.convertScaleAbs(arr, alpha=1.0, beta=beta)

        contrast_strength = float(max(0.0, min(contrast_strength, 2.0)))
        if contrast_strength != 1.0:
            alpha = contrast_strength
            arr = cv2.convertScaleAbs(arr, alpha=alpha, beta=0)
            
        light_balance = float(max(0.01, min(light_balance, 2.0)))
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
# Resampler-Average
# palette_latent = vae.encode(palette_image)
# palette_latent = resize_latent_safe(palette_latent, latent_image.shape)
# palette_latent = standardize_latent(palette_latent)
# noise = inject_custom_noise_to_latent(perin_custom)
#----------------------------------------

class IRL_ImgResampler(IO.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImgResampler",
            display_name="이미지 리샘플러",
            category="이미지 리파이너/인페인팅",
            description="이미지에 하이브리드 격자 텍스처 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다.",
            inputs=[
                IO.Model.Input("model"),
                IO.Clip.Input("clip"),
                IO.Vae.Input("vae"),
                IO.Image.Input("image", optional=True),
                IO.Sigmas.Input("sigmas", optional=True),
                IO.Mask.Input("mask", optional=True),
                IO.Float.Input("denoise", default=0.05, min=0.00, max=1.00, step=0.01),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, step=1, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."), 
                IO.Combo.Input("noise_mode", options=["normal", "small_spread", "big_spread"], default="normal", tooltip="노이즈 스케일 강도"),
                IO.Int.Input("steps", default=20, min=1, max=10000, step=1, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만,"
                             "너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Float.Input("cfg", default=2.0, min=1.0, max=20.0, step=0.1),
                IO.Float.Input("neg_str", default=0.01, min=0.01, max=0.50, step=0.01),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple"),
                IO.String.Input("pos_text", multiline=True, default="masterpiece, sharp focus"),
                IO.String.Input("neg_text", multiline=True, default="blur, low quality"),
                IO.Int.Input("latent_size_x", default=512, min=8, max=2048),
                IO.Int.Input("latent_size_y", default=512, min=8, max=2048),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리"),
                IO.Combo.Input("noise_set", options=["base_only", "random_set"], default="base_only", tooltip="base_only:원본 이미지 기반 연산\n"
                               "random_set:노이즈 재처리 모드"),
                NoisePack.Input("noise_pack", optional=True, tooltip="노이즈 생성기에서 전달된 노이즈 모드"),
            ],
            outputs=[
                IO.Image.Output("image")
            ]
        )

    @classmethod
    def execute(cls, model, clip, vae, image=None, mask=None, sigmas=None, denoise=0.05, seedset=0, noise_mode="normal", 
                steps=12, cfg=2.0, neg_str=0.01, sampler_name="euler", scheduler="simple", 
                pos_text="masterpiece, sharp focus", neg_text="blur, low quality",
                latent_size_x=512, latent_size_y=512, device_set="cpu", clear_cache=False, noise_set="base_only",
                noise_pack=None) -> IO.NodeOutput:

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

                device = "cuda"
            else:
                print("[Warning] The AMD option was selected, but the ROCm (PyTorch HIP) environment is unavailable or does not support CUDA. Switching to CPU for safety.")
                device = "cpu"
        else:
            device = "cpu"

        
        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            elif device == "cpu":
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")


        base_seed = par_seed(seedset)
        if base_seed == 0: 
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))

        print(f"\n{CYAN}{BOLD}[IRL_ImgResampler]{RESET} Running Sampler with Active Seed: {YELLOW}{BOLD}{base_seed}{RESET}")

        generator = torch.Generator(device=device).manual_seed(base_seed)
        np_rng = np.random.default_rng(base_seed)

        sampler = comfy.samplers.sampler_object(sampler_name)
        if sigmas is not None:
            use_sigmas = sigmas.to(device)
        else:
            use_sigmas = get_sigmas(model, scheduler, steps, denoise, device)

        sigmas = use_sigmas[-(steps + 1):]

        sampler_noise_mask = None

        latent_image = None
        if image is not None:
            batch, h, w, c = image.shape
            latent_batch_size = batch
            target_h = max(64, (h // 64) * 64)
            target_w = max(64, (w // 64) * 64)
            if (target_h, target_w) != (h, w):
                image = resize_image(image, (target_w, target_h))
            latent_image = vae.encode(image.to(device))
            if mask is not None:
                mask_checked = mask_type_check(mask, device)
                sampler_noise_mask = resize_mask_to_latent(mask_checked, latent_image).to(device)
            denoise_val = denoise
        else:
            latent_batch_size = 1
            latent_image = torch.zeros((1, 4, latent_size_y // 8, latent_size_x // 8), device=device)
            denoise_val = 1.0
            
        if image is not None:
            if denoise > 0:
                if noise_set == "random_set":
                    noise_source = torch.randn_like(latent_image, generator=generator, device=device)
                else: 
                    # base_only
                    noise_source = latent_image.clone()

                if noise_pack is not None:
                    sampler_noise = inject_noisemode_to_latent(
                        latent_tensor=noise_source, 
                        sigmas=sigmas, 
                        noise_mode=noise_mode,
                        noisepack=noise_pack, 
                        np_rng=np_rng, 
                        device=device,
                        seed=base_seed
                    )
                else:
                    sampler_noise = inject_custom_noise_to_latent(
                        latent_tensor=noise_source, 
                        sigmas=sigmas, 
                        noise_mode=noise_mode, 
                        np_rng=np_rng, 
                        device=device
                    )
            else:
                sampler_noise = latent_image.clone()
        else:
            sampler_noise = torch.randn_like(latent_image, generator=generator, device=device)

        positive = build_conditioning(clip, pos_text, 1.0, device)

        neg_str = max(0.01, min(neg_str, 0.50))
        negative = build_conditioning(clip, neg_text, 1.0 * neg_str, device)

        latent_image = latent_image.to(device)
        sampler_noise = sampler_noise.to(device)
        extra = None
        if latent_image.ndim == 5:
            extra = {"is_flow": True}
        model_options = {"transformer_options": extra or {}}
        comfy.samplers.cast_to_load_options(model_options, device=device, dtype=latent_image.dtype)

        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED
        
        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        latent_refined = comfy.sample.sample_custom(model, sampler_noise, cfg, sampler, sigmas, positive, negative, latent_image, noise_mask=sampler_noise_mask, callback=callback, disable_pbar=disable_pbar, seed=base_seed)


        del sampler_noise, latent_image
        if sampler_noise_mask is not None: del sampler_noise_mask

        decoded = vae.decode(latent_refined)
        arr = to_numpy_image_out(decoded)

        del latent_refined, decoded

        output_tensor = to_tensor_imgoutput(Image.fromarray(arr))
        image=output_tensor.float()
        return IO.NodeOutput(image)

#----------------------------------------
# Resampler-Mixing
#----------------------------------------

class IRL_ImgResamplerMix(IO.ComfyNode):
    
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImgResamplerMix",
            display_name="이미지 리샘플러(믹스)",
            category="이미지 리파이너/인페인팅",
            description="이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다.\n"
                        "시스템 성능이 낮을 경우 색감이 하락할 수 있습니다.\n"
                        "입력 이미지의 가로·세로 크기는 8의 배수 형태를 권장합니다.(ex:512,512)\n"
                        "단일 이미지 처리용입니다. 몇몇 디퓨저 모델에는 사용할 수 없습니다.\n"
                        "절대 여러장을 동시처리하지 마세요. 샘플러가 터집니다.",
            inputs=[
                IO.Model.Input("model", tooltip="참고할 모델"),
                IO.Clip.Input("clip", tooltip="참고할 clip"),
                IO.Vae.Input("vae", tooltip="참고할 vae 객체"),
                IO.Image.Input("image", tooltip="조정할 대상 이미지"),
                IO.Sigmas.Input("sigmas", optional=True),
                IO.Mask.Input("mask", tooltip="참고할 대상 마스크", optional=True),
                IO.Float.Input("denoise", default=0.10, min=0.00, max=1.00, step=0.01,
                               tooltip="디노이즈 처리"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, step=1, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Combo.Input("noise_mode", options=["normal", "small_spread", "big_spread"], default="normal", tooltip="노이즈 스케일 강도"),
                IO.Int.Input("steps", default=20, min=1, max=10000, step=1, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만,"
                             "너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Float.Input("cfg", default=2.0, min=1.0, max=20.0, step=0.1, tooltip="CFG 스케일"),
                IO.Float.Input("neg_str", default=0.01, min=0.01, max=0.50, step=0.01, tooltip="부정 조건 강도 페널티 스케일. 낮출수록 부정 텍스트의 영향을 낮춥니다."),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler", tooltip="샘플러 방식"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple", tooltip="스케줄러 방식"),
                IO.String.Input("pos_text", multiline=True, default="illustration style, global illumination, sharp focus, vivid colors, color balanced",
                                 tooltip="긍정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.String.Input("neg_text", multiline=True, default="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)",
                                 tooltip="부정 프롬프트 텍스트. 키워드를 너무 많이 넣으시면 안됩니다.", optional=True),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="실행 장치"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="남아있는 샘플링 잔여 기록을 정리합니다.\n"
                               "이미지가 붕괴되는게 개선되지 않을 경우 사용해보는걸 권장합니다."),
                IO.Combo.Input("noise_set", options=["base_only", "random_set"], default="base_only", tooltip="base_only:원본 이미지 기반 연산\n"
                               "random_set:노이즈 재처리 모드"),
                NoisePack.Input("noise_pack", optional=True, tooltip="노이즈 생성기에서 전달된 노이즈 모드")
            ],
            outputs=[
                IO.Image.Output("image", tooltip="디테일링 결과 이미지")
            ]
        )

    @classmethod
    def execute(cls, model, clip, vae, image, sigmas=None, mask=None, denoise=0.10, seedset=0, noise_mode="normal", steps=12, cfg=2.0, neg_str=0.01, sampler_name="euler", scheduler="simple", pos_text="illustration style, global illumination, sharp focus, vivid colors, color balanced", 
                neg_text="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)", device_set="cpu", clear_cache=False,
                noise_set="base_only", noise_pack=None) -> IO.NodeOutput:

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

                device = "cuda"
            else:
                print("[Warning] The AMD option was selected, but the ROCm (PyTorch HIP) environment is unavailable or does not support CUDA. Switching to CPU for safety.")
                device = "cpu"
        else:
            device = "cpu"


        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            elif device == "cpu":
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")


        # Seed Settings
        base_seed = par_seed(seedset)
        if base_seed == 0: 
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        print(f"\n{CYAN}{BOLD}[IRL_ImgResamplerMix]{RESET} Running Sampler with Active Seed: {YELLOW}{BOLD}{base_seed}{RESET}")
        
        generator = torch.Generator(device=device).manual_seed(base_seed)
        np_rng = np.random.default_rng(base_seed)

        noise_source = None
        sampler = comfy.samplers.sampler_object(sampler_name)
        if sigmas is not None:
            use_sigmas = sigmas.to(device)
        else:
            use_sigmas = get_sigmas(model, scheduler, steps, denoise, device)

        sigmas = use_sigmas[-(steps + 1):]

        sampler_noise_mask = None
        latent_image = None
        if image is None:
            raise ValueError(f"{CYAN}{BOLD}[IRL_ImgResamplerMix]{RESET} The required image input was not provided.")
        else:
            batch, h, w, c = image.shape
            latent_batch_size = batch
            target_h = max(64, (h // 64) * 64)
            target_w = max(64, (w // 64) * 64)
            if (target_h, target_w) != (h, w):
                image = resize_image(image, (target_w, target_h))
            latent_image = vae.encode(image.to(device))
            if mask is not None:
                mask_checked = mask_type_check(mask, device)
                sampler_noise_mask = resize_mask_to_latent(mask_checked, latent_image).to(device)
            denoise_val = denoise

        if denoise > 0:
            if noise_set == "random_set":
                noise_source = torch.randn_like(latent_image, generator=generator, device=device)
            else: 
                # base_only
                noise_source = latent_image.clone()

            if noise_pack is not None:
                custom_latent_noise = inject_noisemode_to_latent(
                    latent_tensor=noise_source, 
                    sigmas=sigmas, 
                    noise_mode=noise_mode,
                    noisepack=noise_pack, 
                    np_rng=np_rng, 
                    device=device,
                    seed=base_seed
                )
            else:
                custom_latent_noise = inject_custom_noise_to_latent(
                    latent_tensor=latent_image, 
                    sigmas=sigmas, 
                    noise_mode=noise_mode, 
                    np_rng=np_rng, 
                    device=device
                )
        else:
            custom_latent_noise = latent_image.clone()

        sampler_noise = custom_latent_noise.to(device)

        extra = None
        if latent_image.ndim == 5:
            extra = {"is_flow": (latent_image.ndim == 5)}
        model_options = {"transformer_options": extra if extra is not None else {}}


        positive = build_conditioning(clip, pos_text, 1.0, device)

        neg_str = max(0.01, min(neg_str, 0.50))
        negative = build_conditioning(clip, neg_text, 1.0 * neg_str, device)



        comfy.samplers.cast_to_load_options(model_options, device=device, dtype=latent_image.dtype)

        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        latent_refined = comfy.sample.sample_custom(model, sampler_noise, cfg, sampler, use_sigmas, positive, negative, latent_image, noise_mask=sampler_noise_mask, callback=callback, disable_pbar=disable_pbar, seed=base_seed)

        decoded = vae.decode(latent_refined)
        arr = to_numpy_image_out(decoded)
        del sampler_noise, latent_image
        if sampler_noise_mask is not None:
            del sampler_noise_mask
        if noise_source is not None:
            del noise_source
        del latent_refined, decoded

        image = to_tensor_imgoutput(Image.fromarray(arr))
        return IO.NodeOutput(image)

#----------------------------------------
class IRL_InpaintAndMask_CV(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_InpaintAndMask_CV",
            display_name="CV 인페인트 및 마스크 처리",
            category="이미지 리파이너/인페인팅",
            description="OpenCV 기반 인페인팅 노드. 마스크 영역을 지정된 강도와 방식으로 정밀하게 메꿉니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.Mask.Input("mask", tooltip="인페인팅 대상 마스크", optional=True),
                IO.Combo.Input("method", options=["telea", "Navier_Stokes", "hybrid_te", "hybrid", "hybrid_NS"], default="telea", tooltip="인페인팅 알고리즘 (telea / NS / hybrid_te / hybrid / hybrid_NS)"),
                IO.Float.Input("strength", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="인페인트 블렌딩 강도"),
                IO.Combo.Input("mask_set", options=["Normal", "invert"], default="Normal", tooltip="마스크 반전/적용 세팅"),
                IO.Combo.Input("mask_mode", options=["off", "small_spread", "light_spread", "basic", "spread", "big_spread", "hard_spread", "veryhard_spread", "cutoff"], default="off", tooltip="마스크 확장 및 처리 모드"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output("image")]
        )

    @classmethod
    def execute(cls, image, mask=None, method="telea", strength=0.00, mask_set="Normal", mask_mode="off", show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            current_device = model_management.get_torch_device()
            if current_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")        

        arr_orig = ensure_image_tensor(image)
        H, W = arr_orig.shape[2:]
        arr = arr_orig[0].permute(1, 2, 0).cpu().numpy()
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        if mask is None or mask_mode=="off":
            final_out = cv2.cvtColor(arr, cv2.COLOR_BGR2RGB)
            return IO.NodeOutput(to_tensor_output(final_out))

        mask_arr = ensure_mask_tensor(mask)[0, 0].cpu().numpy()
        mask_arr = apply_mask_mode_numpy(mask_arr, mask_set, mask_mode, (H, W))
        mask_arr = mask_arr.squeeze() if mask_arr.ndim == 3 else mask_arr
        _, hard_mask = prepare_mask_for_opencv(mask_arr)

        radius = 1
        blend_alpha = min(max(0.00, float(strength)),1.00)
        if method == "Navier_Stokes":
            flag = cv2.INPAINT_NS
            inpainted = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, flag)
        elif method == "hybrid_NS":
            inpainted_telea = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
            inpainted_ns = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, cv2.INPAINT_NS)
            mix_ratio = 0.3
            inpainted = cv2.addWeighted(inpainted_telea, mix_ratio, inpainted_ns, 1.0 - mix_ratio, 0)
        elif method == "hybrid":
            inpainted_telea = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
            inpainted_ns = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, cv2.INPAINT_NS)
            mix_ratio = 0.5
            inpainted = cv2.addWeighted(inpainted_telea, mix_ratio, inpainted_ns, 1.0 - mix_ratio, 0)
        elif method == "hybrid_te":
            inpainted_telea = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, cv2.INPAINT_TELEA)
            inpainted_ns = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, cv2.INPAINT_NS)
            mix_ratio = 0.7
            inpainted = cv2.addWeighted(inpainted_telea, mix_ratio, inpainted_ns, 1.0 - mix_ratio, 0)
        else:
            flag = cv2.INPAINT_TELEA
            inpainted = cv2.inpaint(arr, hard_mask.astype(np.uint8), radius, flag)

        if mask_arr.max() <= 1.0:
            mask_float = np.clip(mask_arr.astype(np.float32), 0.0, 1.0)
        else:
            mask_float = np.clip(mask_arr.astype(np.float32) / 255.0, 0.0, 1.0)

        effective_mask = mask_float * blend_alpha 
        mask_3c = np.repeat(effective_mask[:, :, np.newaxis], 3, axis=2)
        final_output = (arr * (1.0 - mask_3c) + inpainted * mask_3c).astype(np.uint8)

        final_output = cv2.cvtColor(final_output, cv2.COLOR_BGR2RGB)
        
        if show_preview:
            result_rgb = to_tensor_output(final_output)
            return IO.NodeOutput(to_tensor_output(final_output),ui=UI.PreviewImage(result_rgb))
        return IO.NodeOutput(to_tensor_output(final_output))

#----------------------------------------

class IRL_AutoComposite_Post_CV(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_AutoComposite_Post_CV",
            display_name="CV 오토 콤포짓 및 후처리",
            category="이미지 리파이너/인페인팅",
            description="OpenCV 기반 후처리 노드. 마스크가 있으면 마스크 영역에만 후처리가 적용되며, 팔레트 이미지와 함께 입력 시 합성 후처리됩니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.Image.Input("pal_image", tooltip="합성 대상 이미지 (선택)", optional=True),
                IO.Mask.Input("mask", tooltip="영역 마스크 (선택)", optional=True),
                IO.Combo.Input("mask_mode", options=["off", "small_spread", "light_spread", "basic", "spread", "big_spread", "hard_spread", "veryhard_spread", "cutoff"], default="off", tooltip="마스크 확장 및 처리 모드"),
                IO.Float.Input("contrast_str", default=1.00, min=0.00, max=2.00, step=0.01, tooltip="대비 조절 강도"),
                IO.Float.Input("light_balance", default=1.00, min=0.01, max=2.00, step=0.01, tooltip="광원(V) 보정 강도"),
                IO.Float.Input("color_str", default=0.00, min=0.00, max=2.00, step=0.01, tooltip="채도 보정 강도"),
                IO.Float.Input("sharpen_str", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="샤픈 후처리 강도"),
                IO.Float.Input("line_str", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="라인 엣지 강조 강도"),
                IO.String.Input("line_color", default="#000000", tooltip="라인 색상 (HEX)"),
                IO.Combo.Input("line_mode", options=["basic", "thin", "normal", "bold"], default="basic", tooltip="라인 두께"),
                IO.Float.Input("blendstr", default=0.0, min=0.0, max=1.0, step=0.1, tooltip="마스크 영역 내의 이미지 우선치를 조정합니다. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Combo.Input("blendmode", options=["default", "overlay", "blend", "overwrite"], default="default", tooltip="마스크 영역 내의 합성 형태를 지정합니다. 샘플 팔레트가 없으면 작동하지 않습니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output("image")]
        )

    @classmethod
    def execute(cls, image, pal_image=None, mask=None, mask_mode="off", contrast_str=1.00, light_balance=1.00, color_str=0.00, 
                sharpen_str=0.00, line_str=0.00, line_color="#000000", line_mode="basic", blendstr=0.0, blendmode="default", show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            current_device = model_management.get_torch_device()
            if current_device.type == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")        

        arr_orig = ensure_image_tensor(image)
        H, W = arr_orig.shape[2:]
        arr = arr_orig[0].permute(1, 2, 0).cpu().numpy()# Shape: (h, w, 3)
        arr = (arr * 255).clip(0, 255).astype(np.uint8)
        proc_buf = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        mask_3c = None
        if mask is not None and mask_mode != "off":
            mask_arr = ensure_mask_tensor(mask)[0,0].cpu().numpy()# Shape: (h, w)
            mask_arr = apply_mask_mode_numpy(mask_arr, "Normal", mask_mode, (H,W))# Shape: (h, w, 1)
            if mask_arr.max() <= 1.0:
                mask_float = np.clip(mask_arr.astype(np.float32), 0.0, 1.0)
            else:
                mask_float = np.clip(mask_arr.astype(np.float32) / 255.0, 0.0, 1.0)
            mask_3c = np.repeat(mask_float, 3, axis=-1) # Shape: (h, w, 3)

        post_buf = proc_buf.copy()
        if pal_image is not None and mask_3c is not None:
            pal_orig = ensure_image_tensor(pal_image)
            pal_arr = pal_orig[0].permute(1,2,0).cpu().numpy()# Shape: (h, w, 3)
            pal_arr = (pal_arr*255).clip(0,255).astype(np.uint8)
            pal_arr = cv2.cvtColor(pal_arr, cv2.COLOR_RGB2BGR)
            pal_arr = cv2.resize(pal_arr,(W,H))

            effective_blendstr = float(max(0.0,min(blendstr,1.0)))
            current_mask_3c = mask_3c*effective_blendstr

            if blendmode=="overwrite":
                blended_pal = pal_arr
            elif blendmode=="overlay":
                img_f = proc_buf.astype(np.float32)/255.0
                pal_f = pal_arr.astype(np.float32)/255.0
                overlay_f = np.where(img_f<0.5,2.0*img_f*pal_f,1.0-2.0*(1.0-img_f)*(1.0-pal_f))
                blended_pal = (overlay_f*255.0).clip(0,255).astype(np.uint8)
            elif blendmode=="blend":
                orig_str = 1 - effective_blendstr
                blended_pal = cv2.addWeighted(proc_buf,orig_str,pal_arr,effective_blendstr,0)
            else: # default → 원본 유지
                blended_pal = proc_buf

            post_buf = (proc_buf*(1.0-current_mask_3c)+blended_pal*current_mask_3c).astype(np.uint8)

        if contrast_str!=1.0:
            post_buf = cv2.convertScaleAbs(post_buf,alpha=contrast_str,beta=0)
        hsv = cv2.cvtColor(post_buf,cv2.COLOR_BGR2HSV)
        h_c,s_c,v_c = cv2.split(hsv)
        if color_str>0.0:
            s_c = np.clip(s_c.astype(np.float32)*(1.0+color_str),0,255).astype(np.uint8)
        if light_balance!=1.0:
            v_c = np.clip(v_c.astype(np.float32)*light_balance,0,255).astype(np.uint8)
        hsv = cv2.merge([h_c,s_c,v_c])
        post_buf = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)

        if sharpen_str>0.0:
            blur_layer = cv2.GaussianBlur(post_buf,(0,0),3)
            post_buf = cv2.addWeighted(post_buf,1.0+sharpen_str,blur_layer,-sharpen_str,0)

        if line_str>0.0:
            gray_line = cv2.cvtColor(post_buf,cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(cv2.equalizeHist(gray_line),50,150)
            if line_mode=="thin":
                edges = cv2.dilate(edges,np.ones((1,1),np.uint8),iterations=1)
            elif line_mode=="normal":
                edges = cv2.dilate(edges,np.ones((2,2),np.uint8),iterations=1)
            elif line_mode=="bold":
                edges = cv2.dilate(edges,np.ones((3,3),np.uint8),iterations=2)
            hex_color = line_color.lstrip('#')
            line_rgb = tuple(int(hex_color[i:i+2],16) for i in (0,2,4)) # RGB
            line_bgr = (line_rgb[2],line_rgb[1],line_rgb[0])
            edges_colored = np.zeros_like(post_buf)
            edges_colored[edges>0] = line_bgr
            edges_colored = cv2.GaussianBlur(edges_colored,(3,3),0)
            post_buf = cv2.addWeighted(post_buf,1.0,edges_colored,line_str,0)

        if mask_3c is not None:
            final_proc = (proc_buf*(1.0-mask_3c)+post_buf*mask_3c).astype(np.uint8)
        else:
            final_proc = post_buf

        final_output = cv2.cvtColor(final_proc,cv2.COLOR_BGR2RGB)
        tensor_out = to_tensor_output(final_output)

        if show_preview:
            return IO.NodeOutput(tensor_out,ui=UI.PreviewImage(tensor_out))
        return IO.NodeOutput(tensor_out)

#----------------------------------------

class IRL_ResamplerInpaint(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ResamplerInpaint",
            display_name="리샘플러 세미오토 인페인팅",
            category="이미지 리파이너/인페인팅",
            description="모델 기반 리샘플링과 OpenCV 인페인팅 후처리를 통합한 노드.\n"
                        "텍스트 프롬프트와 마스크를 함께 활용할 수 있습니다."
                        "입력 이미지의 가로·세로 크기는 8의 배수 형태를 권장합니다.(ex:512,512)\n"
                        "단일 이미지 처리용입니다. 몇몇 디퓨저 모델에는 사용할 수 없습니다.\n"
                        "절대 여러장을 동시처리하지 마세요. 샘플러가 터집니다.",
            inputs=[
                IO.Model.Input("model", tooltip="참고할 모델"),
                IO.Clip.Input("clip", tooltip="참고할 clip"),
                IO.Vae.Input("vae", tooltip="참고할 vae 객체"),
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.Sigmas.Input("sigmas", optional=True),
                IO.Mask.Input("mask", tooltip="재처리 영역 마스크", optional=True),
                IO.Combo.Input("noise_set", options=["base_only", "random_set"], default="base_only", tooltip="base_only:원본 이미지 기반 연산\n"
                               "random_set:노이즈 재처리 모드"),
                IO.Float.Input("denoise", default=0.10, min=0.00, max=1.00, step=0.01, tooltip="디노이즈 처리"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, step=1, tooltip="노이즈 시드.0이면 랜덤 시드를 넣고, 시드넘버를 넣은 경우 고정시드로 취급됩니다."),
                IO.Combo.Input("noise_mode", options=["normal", "small_spread", "big_spread"], default="normal", tooltip="노이즈 스케일 강도"),
                IO.Int.Input("steps", default=20, min=1, max=10000, step=1, tooltip="스텝 수. 수텝수가 많아질수록 추가작업이 들어가지만,"
                             "너무 많은 경우 품질이 떨어질 수 있습니다."),
                IO.Float.Input("cfg", default=2.0, min=1.0, max=20.0, step=0.1, tooltip="CFG 스케일"),
                IO.Float.Input("neg_str", default=0.01, min=0.01, max=0.50, step=0.01,
                               tooltip="부정 조건 강도"),
                IO.Combo.Input("sampler_name", options=comfy.samplers.KSampler.SAMPLERS, default="euler",
                               tooltip="샘플러 방식"),
                IO.Combo.Input("scheduler", options=comfy.samplers.KSampler.SCHEDULERS, default="simple",
                               tooltip="스케줄러 방식"),
                IO.String.Input("pos_text", multiline=True, default="illustration style, global illumination, sharp focus, vivid colors, color balanced",
                                tooltip="긍정 프롬프트"),
                IO.String.Input("neg_text", multiline=True, default="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)",
                                tooltip="부정 프롬프트"),
                IO.Float.Input("color_sen", default=0.00, min=0.00, max=1.00, step=0.01, tooltip="색상 민감도(Detail Enhance). OpenCV 디테일 강화 효과를 부여합니다 (0이면 비활성화)."),
                IO.Float.Input("color_sig", default=10, min=1, max=50, step=1,
                               tooltip="색상 시그마(Spatial Sigma). 디테일 강화 시 반영할 공간적 반경 크기입니다.\n"
                               "값이 작아질수록 색상이 강조됩니다."),
                IO.Float.Input("color_str", default=0.00, min=0.00, max=2.00, step=0.01,
                               tooltip="채도 보정(Saturation). 이미지의 전체적인 색감 채도를 끌어올리거나 낮춥니다."),
                IO.Float.Input("contrast_str", default=1.00, min=0.00, max=2.00, step=0.01,
                               tooltip="대비 조절(Contrast). 명암의 대비를 낮추거나 늘리며 조절합니다 (1.0 기준)."),
                IO.Float.Input("light_balance", default=1.00, min=0.01, max=2.00, step=0.01,
                               tooltip="광원 보정(Light Balance). 밝기(Value) 채널을 스케일링하여 전체 밝기를 조절합니다."),
                IO.Float.Input("line_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="라인 강조 강도(Canny Edge). 외곽선 검출 후 선을 또렷하게 덧입힙니다."),
                IO.String.Input("line_color", default="#000000", tooltip="라인 색상. Canny 외곽선에 적용할 색상 코드를 입력합니다. (HEX 형식, 예: #000000)."),
                IO.Combo.Input("line_mode", options=["basic", "thin", "normal", "bold"], default="basic", tooltip="라인 굵기 모드. 외곽선의 두께를 조절합니다."),
                IO.Combo.Input("line_blur", options=["basic", "off", "overlay"], default="basic", tooltip="라인 블러 처리 방식. 엣지 라인에 블러를 먹여 과한 엣지를 줄이거나 블러를 스킵하고 강한 엣지라인을 부여합니다."),
                IO.Float.Input("sharpen_str", default=0.00, min=0.00, max=1.00, step=0.01,
                               tooltip="샤픈 후처리 강도(Sharpening). 언샤프 마스크를 통해 이미지를 선명하게 다듬습니다."),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="연산 처리를 수행할 하드웨어 디바이스 지정"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="켰을 시 캐시를 정리합니다."),
                NoisePack.Input("noise_pack", optional=True, tooltip="외부 노이즈 생성기에서 전달된 커스텀 노이즈 패키지 모드")
            ],
            outputs=[IO.Image.Output("image", tooltip="최종 결과 이미지")]
        )


    @classmethod
    def execute(cls, model, clip, vae, image, sigmas=None, mask=None, noise_set="base_only", denoise=0.10, 
                seedset=0, noise_mode="normal", steps=12, cfg=2.0, neg_str=0.01, sampler_name="euler", scheduler="simple",
                pos_text="illustration style, global illumination, sharp focus, vivid colors, color balanced",
                neg_text="text, watermark, (bad anatomy:0.3), (extra limbs:0.3), (blur:0.5), (desaturated:0.5)",
                contrast_str=1.0, color_sen=0.0, color_sig=10, color_str=0.0, light_balance=1.0, 
                line_str=0.0, line_color="#000000", line_mode="basic", line_blur="basic", sharpen_str=0.00, 
                device_set="cpu", clear_cache=False, noise_pack=None) -> IO.NodeOutput:

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

                device = "cuda"
            else:
                print("[Warning] The AMD option was selected, but the ROCm (PyTorch HIP) environment is unavailable or does not support CUDA. Switching to CPU for safety.")
                device = "cpu"
        else:
            device = "cpu"

        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            elif device == "cpu":
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        # Seed Settings
        base_seed = par_seed(seedset)
        if base_seed == 0: 
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        generator = torch.Generator(device=device).manual_seed(base_seed)
        
        print(f"\n{CYAN}{BOLD}[IRL_ResamplerInpaint]{RESET}  Processing Seed: {YELLOW}{BOLD}{base_seed}{RESET}")
        np_rng = np.random.default_rng(base_seed)

        # latent scaling & encode & set latent
        noise_source = None
        sampler = comfy.samplers.sampler_object(sampler_name)
        if sigmas is not None:
            use_sigmas = sigmas.to(device)
        else:
            use_sigmas = get_sigmas(model, scheduler, steps, denoise, device)

        sigmas = use_sigmas[-(steps + 1):]
       
        sampler_noise_mask = None
        latent_image = None
        if image is not None:
            batch, h, w, c = image.shape
            latent_batch_size = batch
            target_h = max(64, (h // 64) * 64)
            target_w = max(64, (w // 64) * 64)
            if (target_h, target_w) != (h, w):
                image = resize_image(image, (target_w, target_h))
            latent_image = vae.encode(image.to(device))
            if mask is not None:
                mask_checked = mask_type_check(mask, device)
                sampler_noise_mask = resize_mask_to_latent(mask_checked, latent_image).to(device)
            denoise_val = denoise
        else:
            raise ValueError(f"{CYAN}{BOLD}[IRL_ResamplerInpaint]{RESET}The required image input was not provided.")
            
        if image is not None:
            if denoise > 0:
                if noise_set == "random_set":
                    latent_source = torch.randn_like(latent_image, generator=generator, device=device)
                else: 
                    # base_only
                    latent_source = latent_image.clone()

                if noise_pack is not None:
                    noise_source = inject_noisemode_to_latent(
                        latent_tensor=latent_source, 
                        sigmas=sigmas, 
                        noise_mode=noise_mode,
                        noisepack=noise_pack, 
                        np_rng=np_rng, 
                        device=device,
                        seed=base_seed
                    )
                else:
                    noise_source = inject_custom_noise_to_latent(
                        latent_tensor=latent_source, 
                        sigmas=sigmas, 
                        noise_mode=noise_mode, 
                        np_rng=np_rng, 
                        device=device
                    )
            else:
                noise_source = latent_image.clone()
        else:
            noise_source = torch.randn_like(latent_image, generator=generator, device=device)
        

        sampler_noise = noise_source.to(device)

        # 8. text encoding & set conditionig
        positive = build_conditioning(clip, pos_text, 1.0, device)

        neg_str = max(0.01, min(neg_str, 0.50))
        negative = build_conditioning(clip, neg_text, 1.0 * neg_str, device)

        # 9. Sampling

        extra = None
        if latent_image.ndim == 5:
            extra = {"is_flow": (latent_image.ndim == 5)}
        model_options = {"transformer_options": extra if extra is not None else {}}

        comfy.samplers.cast_to_load_options(model_options, device=device, dtype=latent_image.dtype)


        callback = latent_preview.prepare_callback(model, steps)
        disable_pbar = not comfy.utils.PROGRESS_BAR_ENABLED

        latent_refined = comfy.sample.sample_custom(model, sampler_noise, cfg, sampler, use_sigmas, positive, negative, latent_image, noise_mask=sampler_noise_mask, callback=callback, disable_pbar=disable_pbar, seed=base_seed)

        # 10. return image & CV detailEnhancer
        decoded = vae.decode(latent_refined)
        rgb_arr = to_numpy_image_out(decoded)
        inpainted = cv2.cvtColor(rgb_arr, cv2.COLOR_RGB2BGR)


        # -------------------------------------------------------------------------
        # 🧪 [Advanced CV logic]
        # -------------------------------------------------------------------------

        color_sen = float(max(0.00, min(color_sen, 1.00)))
        color_sig = int(max(1, min(color_sig, 50)))
        if color_sen > 0.00:
            inpainted = cv2.detailEnhance(inpainted, sigma_s=color_sig, sigma_r=color_sen)

        # BGR2HSV convert
        hsv = cv2.cvtColor(inpainted, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)
        del latent_refined, decoded
        del noise_source, sampler_noise, latent_image, positive, negative
        if latent_source is not None:
            del latent_source

        # Contrast
        contrast_str = float(max(0.00, min(contrast_str, 2.00)))
        if contrast_str != 1.00:
            v_ch = cv2.convertScaleAbs(v_ch, alpha=contrast_str, beta=0)

        # Light Balance
        light_balance = float(max(0.01, min(light_balance, 2.00)))
        if light_balance != 1.00:
            v_ch = np.clip(v_ch.astype(np.float32) * light_balance, 0, 255).astype(np.uint8)

        # Sharpening
        sharpen_str = float(max(0.00, min(sharpen_str, 1.00)))
        if sharpen_str > 0.00:
            v_blur = cv2.GaussianBlur(v_ch, (0, 0), 3)
            v_ch = cv2.addWeighted(v_ch, 1.0 + sharpen_str, v_blur, -sharpen_str, 0)

        # Saturation Adjust
        color_str = float(max(0.00, min(color_str, 2.00)))
        if color_str > 0.00:
            s_ch = np.clip(s_ch.astype(np.float32) * (1.0 + color_str), 0, 255).astype(np.uint8)

        # return BGR
        hsv_combined = cv2.merge([h_ch, s_ch, v_ch])
        inpainted = cv2.cvtColor(hsv_combined, cv2.COLOR_HSV2BGR)

        # Canny Edge line enhancement
        line_str = float(max(0.00, min(line_str, 1.00)))
        if line_str > 0.00:
            gray = cv2.cvtColor(inpainted, cv2.COLOR_BGR2GRAY)
            gray_eq = cv2.equalizeHist(gray)
            edges = cv2.Canny(gray_eq, 50, 150)

            if line_mode == "thin": edges = cv2.dilate(edges, np.ones((1, 1), np.uint8), iterations=1)
            elif line_mode == "normal": edges = cv2.dilate(edges, np.ones((2, 2), np.uint8), iterations=1)
            elif line_mode == "bold": edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=2)

            hex_color = line_color.lstrip('#')
            line_rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
            line_bgr = (line_rgb[2], line_rgb[1], line_rgb[0])

            if line_blur == "overlay":
                inpainted[edges > 0] = line_bgr
                blured = self.apply_blur(inpainted, 3)  # blur_radius
                inpainted = np.where(mask > 0.5 if mask is not None else False, blured, inpainted)
            else:
                edges_colored = np.zeros_like(inpainted)
                edges_colored[edges > 0] = line_bgr
                inpainted = cv2.addWeighted(inpainted, 1.0, edges_colored, line_str, 0)

        # 11. return
        image = to_tensor_imgoutput(cv2.cvtColor(inpainted, cv2.COLOR_BGR2RGB))

        # garbage collecting
        del rgb_arr

        if mask is not None:
            del mask, sampler_noise_mask
        del inpainted, h_ch, s_ch, v_ch

        return IO.NodeOutput(image)

#----------------------------------------


class IRL_rescaler(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_rescaler",
            display_name="리스케일러",
            category="이미지 리파이너/인페인팅",
            description="다운스케일 후 리스케일을 시도해서 위화감을 줄여보려는 실험 노드.\n"
                        "입력되는 이미지의 높이/폭은 짝수여야 합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.UpscaleModel.Input("upscale_model", tooltip="참고 업스케일 모델", optional=True),
                IO.Combo.Input("tile_str", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11", "12", "13", "14", "15", "16", "17", "18", "19", "20"],
                               default="0", tooltip="타일링 처리 설정.수치가 높을수록 처리강도가 낮아지고  블러처리가 늘어납니다."),
                IO.Combo.Input("overlap_str", options=["0", "1", "2", "3", "4", "5", "6", "7", "8", "9", "10"],
                               default="0", tooltip="겹침 처리 설정.수치가 높을수록 처리강도가 낮아지고 블러처리가 늘어납니다."),
                IO.Combo.Input("rescale_mode", options=["off", "resize", "upscalemodel"], default="resize",
                               tooltip="출력 스케일 처리 방식: off=리사이즈 안함, resize=기본 보간, upscalemodel=업스케일 모델 사용(업스케일 모델 사용시는 디테일강도, 엣지필터강도는 적용되지 않습니다."),
                IO.Combo.Input("downscale_filter", options=["off", "bilinear", "pixelbox", "vectorimage"], default="off",
                                tooltip="다운스케일 보간 시 사용할 보간 방법. 보간 필터: bilinear=Bilinear, pixelbox=INTER_AREA, nearest=Vectorimage"),
                IO.Combo.Input("rescale_filter", options=["off", "nearest", "cubic", "lanczos"], default="off",
                                tooltip="스케일 보간 시 사용할 보간 방법. 모델이 있을땐 작동하지 않습니다. 보간 필터: cubic=INTER_CUBIC, lanczos=INTER_LANCZOS4, nearest=INTER_NEAREST"),
                IO.Combo.Input("detail_str", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
                               default="6", tooltip="디테일 강도 처리 설정.수치가 높을수록 처리강도가 낮아지고  블러처리가 늘어납니다. 모델이 있을땐 작동하지 않습니다."),
                IO.Combo.Input("edgeFilter_str", options=["1", "2", "3", "4", "5", "6", "7", "8", "9", "10", "11"],
                               default="6", tooltip="엣지필터 처리 설정.수치가 높을수록 처리강도가 낮아지고  블러처리가 늘어납니다. 모델이 있을땐 작동하지 않습니다."),
                IO.Combo.Input("resize_factor", options=["0", "1", "2", "3", "4"], default="0", tooltip="출력 크기 조정: 0=조정 없음, 1=2배, 2=4배, 3=6배, 4=8배"),
                IO.Combo.Input("device_set", options=["cpu", "nvidia", "amd"], default="cpu", tooltip="실행 장치"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="켰을 시 캐시를 정리합니다."),
            ],
            outputs=[IO.Image.Output("image")]
        )
    @classmethod
    def execute(cls, image, upscale_model=None, method="telea", tile_str="0", overlap_str="0", rescale_mode="off", downscale_filter="off", 
                rescale_filter="off", detail_str="6", edgeFilter_str="6", resize_factor="0", device_set="cpu", clear_cache=False) -> IO.NodeOutput:

                    
        # --- Select Device --- 

        if device_set == "cpu":
            device = "cpu"
        elif device_set == "nvidia":
            device = "cuda"
        elif device_set == "amd":
            if torch.cuda.is_available() and torch.version.hip:
                props = torch.cuda.get_device_properties(0)
                arch = getattr(props, "gcnArchName", "")
                print("AMD arch:", arch, "ROCm version:", torch.version.hip)

                device = "cuda"
            else:
                print("[Warning] The AMD option was selected, but the ROCm (PyTorch HIP) environment is unavailable or does not support CUDA. Switching to CPU for safety.")
                device = "cpu"
        else:
            device = "cpu"

        if clear_cache:
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            elif device == "cpu":
                print("CPU mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        # --- Image to numpy ---
        arr = ensure_image_tensor(image)
        H, W = arr.shape[2:]
        arr = arr[0].permute(1,2,0).cpu().numpy()
        arr = (arr * 255).clip(0,255).astype(np.uint8)

        # downscale

        small_H, small_W = max(64, H // 2), max(64, W // 2)
        if downscale_filter == "off":
            arr_small = arr.copy()

        elif downscale_filter == "bilinear":
            arr_small = cv2.resize(arr, (small_W, small_H), interpolation=cv2.INTER_LINEAR)

        elif downscale_filter == "pixelbox":
            arr_small = cv2.resize(arr, (small_W, small_H), interpolation=cv2.INTER_AREA)

        elif downscale_filter == "vectorimage":
            arr_small = run_vector_resize(arr, small_W, small_H, "pixelbox")

        # rescale factor
        tile_val = int(tile_str)
        tile_size = 1 if tile_val == 0 else tile_val * 4
        overlap_val = int(overlap_str)
        overlap_size = overlap_val * 2

        # rescale
        if rescale_mode == "upscalemodel" and upscale_model is not None:


            inpainted = run_upscale_with_progress(upscale_model, arr_small, tile=int(512 * tile_size), overlap=int(32 * overlap_size))

        elif rescale_mode == "resize":
            resize_val = int(resize_factor)+1
            if rescale_filter == "off":
                resize_val = 1
                inpainted = arr_small.copy()
            interp = cv2.INTER_CUBIC if rescale_filter=="cubic" else \
                     cv2.INTER_LANCZOS4 if rescale_filter=="lanczos" else \
                     cv2.INTER_NEAREST

            target_W = small_W * resize_val
            target_H = small_H * resize_val
            inpainted = tiled_resize(arr_small, target_W, target_H, tile_size=int(512 * tile_size), overlap_size=int(32 * overlap_size), interp=interp)

            inpainted = apply_detail_enhance(inpainted, detail_str)

            inpainted = apply_edge_filter(inpainted, edgeFilter_str)

        else:
            inpainted = arr

        del arr
        del arr_small

        inpaint=to_tensor_output(inpainted)
        del inpainted

        return IO.NodeOutput(inpaint)
    
#----------------------------------------


SAMPLING_NODE_CLASS_MAPPINGS = {
    "IRL_ColorTransfer": IRL_ColorTransfer,
    "IRL_ImgDetailer": IRL_ImgDetailer,
    "IRL_ImgResampler": IRL_ImgResampler,
    "IRL_ImgResamplerMix": IRL_ImgResamplerMix,
    "IRL_InpaintAndMask_CV": IRL_InpaintAndMask_CV,
    "IRL_AutoComposite_Post_CV": IRL_AutoComposite_Post_CV,
    "IRL_ResamplerInpaint": IRL_ResamplerInpaint,
    "IRL_rescaler": IRL_rescaler,
}

SAMPLING_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_ColorTransfer": "컬러 트랜스퍼",
    "IRL_ImgDetailer": "이미지 디테일러",
    "IRL_ImgResampler": "이미지 리샘플러",
    "IRL_ImgResamplerHook": "이미지 리샘플러(믹스)",
    "IRL_InpaintAndMask_CV": "CV 인페인트 및 마스크 처리",
    "IRL_AutoComposite_Post_CV": "CV 오토 콤포짓 및 후처리",
    "IRL_ResamplerInpaint": "리샘플러 세미오토 인페인팅",
    "IRL_rescaler": "리스케일러",
}

#----------------------------------------
