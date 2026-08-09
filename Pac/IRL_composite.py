# -------------------------------
# IR Lite — Composite Nodes
# -------------------------------

import os
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
import cv2
import torchvision
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import resize
from torchvision.transforms import InterpolationMode
from comfy_api.latest import IO, UI

# ---------------------------------------
# Header Utils
#----------------------------------------

def to_tensor_output(tensor: torch.Tensor):
    """
    Torch tensor return(add batch)
    """
    if tensor.ndim == 3:  # (H, W, C)
        tensor = tensor.unsqueeze(0)
    return tensor.float().clamp(0.0, 1.0)

def to_torch_image(image):
    """
    input-> Torch float32 [0,1] tensor
    """
    if isinstance(image, torch.Tensor):
        return image.float().clamp(0.0, 1.0)  # (1,H,W,3)
    elif isinstance(image, np.ndarray):
        arr = torch.from_numpy(image).float()
        if arr.max() > 1.0:
            arr = arr / 255.0
        return arr.unsqueeze(0) if arr.ndim == 3 else arr  # (1,H,W,3)
    elif isinstance(image, Image.Image):
        arr = torch.from_numpy(np.array(image.convert("RGB"))).float() / 255.0
        return arr.unsqueeze(0)  # (1,H,W,3)
    else:
        raise TypeError("Unsupported image type")

def torch_to_numpy(image_t: torch.Tensor) -> np.ndarray:
    """
    Torch tensor -> numpy (H,W,3) uint8
    Supports (B,C,H,W) or (B,H,W,C)
    """
    if image_t.ndim == 4:
        image_t = image_t[0]

    if image_t.shape[0] in [1,3]:  
        # (C,H,W)
        image_np = image_t.permute(1, 2, 0).cpu().numpy()
    else:
        # (H,W,C)
        image_np = image_t.cpu().numpy()

    image_np = (image_np * 255).clip(0, 255).astype(np.uint8)
    return image_np


def numpy_to_torch(image_np: np.ndarray) -> torch.Tensor:
    """
    numpy (H,W,3) or (H,W,1) or (B,H,W,3/1) uint8 -> Torch tensor (B,H,W,C) [0,1]
    """
    arr = torch.from_numpy(image_np).float() / 255.0

    if arr.ndim == 3:
        # (H,W,C) → (1,H,W,C)
        arr = arr.unsqueeze(0)
    elif arr.ndim == 4:
        # (B,H,W,C)
        pass
    else:
        raise ValueError(f"Unexpected shape for image_np: {arr.shape}")

    return arr

    
    
def gaussian_blur(tensor, kernel_size=5, sigma=2):
    # tensor: (B,1,H,W)
    k = kernel_size // 2
    x = torch.arange(-k, k+1, dtype=torch.float32)
    gauss = torch.exp(-(x**2)/(2*sigma**2))
    gauss = gauss / gauss.sum()
    kernel2d = gauss.unsqueeze(0) @ gauss.unsqueeze(1)
    kernel2d = kernel2d / kernel2d.sum()
    kernel2d = kernel2d.unsqueeze(0).unsqueeze(0)  # (1,1,k,k)

    blurred = F.conv2d(tensor, kernel2d, padding=k)
    return blurred  # (B,1,H,W)


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

def process_mask(mask, a, Mask_mode="normal"):
    m = ensure_mask_tensor(mask)  # (B,1,H,W)

    if Mask_mode == "Small_spread":
        m = F.max_pool2d(m, 3, stride=1, padding=1)
    elif Mask_mode == "big_spread":
        m = F.max_pool2d(m, 5, stride=1, padding=2)
    elif Mask_mode == "blur":
        m = gaussian_blur(m, kernel_size=5, sigma=2)
        m = m.expand_as(a)

    # (B,1,H,W) → (B,H,W,1)
    m = m.permute(0, 2, 3, 1)

    # (B,H,W,1) → (B,H,W,3)
    m = m.expand(-1, -1, -1, a.shape[-1])

    return m

def apply_blend_mode(a, b, mode="Blend", factor=0.5):
    if mode == "Blend":
        return (a * (1.0 - factor) + b * factor).clamp(0.0, 1.0)
    elif mode == "Overlay":
        return torch.where(a < 0.5, 2 * a * b, 1 - 2 * (1 - a) * (1 - b))
    elif mode == "Add":
        return (a + b).clamp(0.0, 1.0)
    elif mode == "Multiply":
        return (a * b).clamp(0.0, 1.0)
    elif mode == "Difference":
        return torch.abs(a - b)
    else:
        return a

def apply_weighted(a, blended, priority_a, priority_b, strength):
    weighted = (a * priority_a + blended * priority_b) / (priority_a + priority_b)
    return (a * (1.0 - strength) + weighted * strength).clamp(0.0, 1.0)
    
# ---------------------------------------
# Cut Layout Loader
# ---------------------------------------

BASE_DIR = os.path.dirname(__file__)
CUT_LAYOUT_DIR = os.path.join(BASE_DIR, "cut_layout")

COLOR_MAP = {
    "A": (255, 0, 0),   # R → image_a
    "B": (0, 255, 0),   # G → image_b
    "C": (0, 0, 255),   # B → image_C
    "D": (255, 255, 0),   # Y → image_D
    "E": (255, 0, 255),   # Magenta → image_E
    "F": (0, 255, 255),   # CYan → image_F
}

def load_reference_canvas(cut_type="2cut", filename="sample.png"):
    ref_path = os.path.join(CUT_LAYOUT_DIR, cut_type, filename)
    if not os.path.exists(ref_path):
        raise FileNotFoundError(f"Reference canvas not found: {ref_path}")
    img = Image.open(ref_path).convert("RGB")
    return to_torch_image(img)

def make_fixed_masks(ref_img_np, tolerance=10):
    masks = {}
    for key, color in COLOR_MAP.items():
        diff = np.abs(ref_img_np - np.array(color))
        mask = np.all(diff < tolerance, axis=-1).astype(np.float32)
        masks[key] = torch.from_numpy(mask).unsqueeze(0).unsqueeze(-1)  # (1,H,W,1)
    return masks

def hex_to_rgb(hex_color: str):
    hex_color = hex_color.lstrip('#')
    lv = len(hex_color)
    return tuple(int(hex_color[i:i+lv//3], 16) for i in range(0, lv, lv//3))

def hex_to_tensor_value(hex_color: str):
    r, g, b = hex_to_rgb(hex_color)
    return (r/255.0, g/255.0, b/255.0)

def make_color_canvas(H, W, pad_color="#FFFFFF"):
    r, g, b = hex_to_rgb(pad_color)
    color_tensor = torch.tensor([r/255.0, g/255.0, b/255.0], dtype=torch.float32)
    canvas = color_tensor.view(1,1,1,3).repeat(1,H,W,1)  # (1,H,W,3)
    return canvas

def apply_padding(image, top=0, bottom=0, left=0, right=0, pad_color="#FFFFFF"):

    print("apply image.shape:", image.shape)
    _, h, w, c = image.shape
    r, g, b = hex_to_rgb(pad_color)
    color_tensor = torch.tensor([r/255.0, g/255.0, b/255.0], dtype=torch.float32)
    canvas = color_tensor.view(1,1,1,3).expand(1,h+top+bottom,w+left+right,3).clone()
    # return
    canvas[:, top:top+h, left:left+w, :] = image
    return canvas

def resize_keep_ratio(img, target_w=None, target_h=None, resize_set="NEAREST"):

    interp_map = {
        "NEAREST": InterpolationMode.NEAREST,
        "BILINEAR": InterpolationMode.BILINEAR,
        "BICUBIC": InterpolationMode.BICUBIC,
        "LANCZOS": InterpolationMode.LANCZOS,
    }
    interpolation = interp_map.get(resize_set, InterpolationMode.NEAREST)

    _, h, w, c = img.shape
    if target_w is not None:  # vertical 모드
        scale = target_w / w
        new_h = int(h * scale)
        new_w = target_w
    elif target_h is not None:  # horizontal 모드
        scale = target_h / h
        new_w = int(w * scale)
        new_h = target_h
    else:
        new_h, new_w = h, w
    img = img.permute(0,3,1,2)
    img = TF.resize(img, (new_h, new_w), interpolation=interpolation)
    return img.permute(0,2,3,1)

def apply_image_2cut(ref_canvas, image_a, image_b, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
    
    H, W = ref_canvas.shape[1:3]
    Ha, Wa = image_a.shape[1:3]
    Hb, Wb = image_b.shape[1:3]
    print("apply ref_canvas.shape:", ref_canvas.shape)
    print("apply image_a.shape:", image_a.shape)
    print("apply image_b.shape:", image_b.shape)


    # resize
    if mode == "vertical":
        image_a = resize_keep_ratio(image_a, target_w=W-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_w=W-32, resize_set=resize_set)
        print("apply resize image_a.shape:", image_a.shape)
        print("apply resize image_b.shape:", image_b.shape)
        Ha, Wa = image_a.shape[1:3]
        Hb, Wb = image_b.shape[1:3]
    else:  # horizontal
        image_a = resize_keep_ratio(image_a, target_h=H-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_h=H-32, resize_set=resize_set)
        print("apply resize image_a.shape:", image_a.shape)
        print("apply resize image_b.shape:", image_b.shape)
        Ha, Wa = image_a.shape[1:3]
        Hb, Wb = image_b.shape[1:3]

    # padding logic
    if mode == "vertical":
        Y = H-16
        padded_a = apply_padding(image_a, top=16, bottom=max(0, Y - Hb), left=16, right=16, pad_color=pad_color)
        padded_b = apply_padding(image_b, top=max(0, Y - Ha), bottom=16, left=16, right=16, pad_color=pad_color)
        return padded_a, padded_b

    elif mode == "horizontal":
        X = W-16
        padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=max(0, X - Wb), pad_color=pad_color)
        padded_b = apply_padding(image_b, top=16, bottom=16, left=max(0, X - Wa), right=16, pad_color=pad_color)
        return padded_a, padded_b

    else:
        return ref_canvas

def composite_fixed_2cut(ref_canvas, padded_a, padded_b, masks):

    print(padded_a.min(), padded_a.max())
    print(padded_b.min(), padded_b.max())
    print("mask A shape:", masks["A"].shape)
    print("mask B shape:", masks["B"].shape)
    print("mask A unique:", masks["A"].unique())
    print("mask B unique:", masks["B"].unique())
    print("apply padded_a.shape:", padded_a.shape)
    print("apply padded_b.shape:", padded_b.shape)
    
    mask_a = masks["A"].float().repeat(1,1,1,3)  # (1,H,W,3)
    mask_b = masks["B"].float().repeat(1,1,1,3)  # (1,H,W,3)
    print("apply mask_a.shape:", mask_a.shape)
    print("apply mask_b.shape:", mask_b.shape)
    composite_a = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite = composite_a * (1 - mask_b) + padded_b * mask_b
    print("composite min/max:", composite.min().item(), composite.max().item())
    return composite

def apply_image_3cut(ref_canvas, image_a, image_b, image_c, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
    H, W = ref_canvas.shape[1:3]

    if mode == "vertical":
        image_a = resize_keep_ratio(image_a, target_w=W-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_w=W-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_w=W-32, resize_set=resize_set)
    else:
        image_a = resize_keep_ratio(image_a, target_h=H-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_h=H-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_h=H-32, resize_set=resize_set)

    if mode == "vertical":
        padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_b = apply_padding(image_b, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_c = apply_padding(image_c, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        return padded_a, padded_b, padded_c
    else:
        padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_b = apply_padding(image_b, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        padded_c = apply_padding(image_c, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
        return padded_a, padded_b, padded_c

def composite_fixed_3cut(ref_canvas, padded_a, padded_b, padded_c, masks):
    print(padded_a.min(), padded_a.max())
    print(padded_b.min(), padded_b.max())
    print(padded_c.min(), padded_c.max())
    print("mask A shape:", masks["A"].shape)
    print("mask B shape:", masks["B"].shape)
    print("mask c shape:", masks["c"].shape)
    print("mask A unique:", masks["A"].unique())
    print("mask B unique:", masks["B"].unique())
    print("mask c unique:", masks["c"].unique())
    print("apply padded_a.shape:", padded_a.shape)
    print("apply padded_b.shape:", padded_b.shape)
    print("apply padded_c.shape:", padded_c.shape)
    mask_a = masks["A"].float().repeat(1, 1, 1, 3)
    mask_b = masks["B"].float().repeat(1, 1, 1, 3)
    mask_c = masks["C"].float().repeat(1, 1, 1, 3)
    print("apply mask_a.shape:", mask_a.shape)
    print("apply mask_b.shape:", mask_b.shape)
    print("apply mask_c.shape:", mask_c.shape)
    
    composite_a = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite_b = composite_a * (1 - mask_b) + padded_b * mask_b
    composite = composite_b * (1 - mask_c) + padded_c * mask_c
    print("composite min/max:", composite.min().item(), composite.max().item())
    return composite


def apply_image_4cut(ref_canvas, image_a, image_b, image_c, image_d, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
    H, W = ref_canvas.shape[1:3]
    
    if mode == "vertical":
        image_a = resize_keep_ratio(image_a, target_w=W-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_w=W-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_w=W-32, resize_set=resize_set)
        image_d = resize_keep_ratio(image_d, target_w=W-32, resize_set=resize_set)
    else:
        image_a = resize_keep_ratio(image_a, target_h=H-32, resize_set=resize_set)
        image_b = resize_keep_ratio(image_b, target_h=H-32, resize_set=resize_set)
        image_c = resize_keep_ratio(image_c, target_h=H-32, resize_set=resize_set)
        image_d = resize_keep_ratio(image_d, target_h=H-32, resize_set=resize_set)

    padded_a = apply_padding(image_a, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    padded_b = apply_padding(image_b, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    padded_c = apply_padding(image_c, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    padded_d = apply_padding(image_d, top=16, bottom=16, left=16, right=16, pad_color=pad_color)
    
    return padded_a, padded_b, padded_c, padded_d

def composite_fixed_4cut(ref_canvas, padded_a, padded_b, padded_c, padded_d, masks):
    mask_a = masks["A"].float().repeat(1, 1, 1, 3)
    mask_b = masks["B"].float().repeat(1, 1, 1, 3)
    mask_c = masks["C"].float().repeat(1, 1, 1, 3)
    mask_d = masks["D"].float().repeat(1, 1, 1, 3)
    
    composite = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite = composite * (1 - mask_b) + padded_b * mask_b
    composite = composite * (1 - mask_c) + padded_c * mask_c
    composite = composite * (1 - mask_d) + padded_d * mask_d
    return composite

def apply_image_5cut(ref_canvas, image_a, image_b, image_c, image_d, image_e, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
    H, W = ref_canvas.shape[1:3]
    
    imgs = [image_a, image_b, image_c, image_d, image_e]
    resized = []
    for img in imgs:
        if mode == "vertical":
            resized.append(resize_keep_ratio(img, target_w=W-32, resize_set=resize_set))
        else:
            resized.append(resize_keep_ratio(img, target_h=H-32, resize_set=resize_set))

    padded = [apply_padding(img, top=16, bottom=16, left=16, right=16, pad_color=pad_color) for img in resized]
    return tuple(padded)

def composite_fixed_5cut(ref_canvas, padded_a, padded_b, padded_c, padded_d, padded_e, masks):
    mask_a = masks["A"].float().repeat(1, 1, 1, 3)
    mask_b = masks["B"].float().repeat(1, 1, 1, 3)
    mask_c = masks["C"].float().repeat(1, 1, 1, 3)
    mask_d = masks["D"].float().repeat(1, 1, 1, 3)
    mask_e = masks["E"].float().repeat(1, 1, 1, 3)
    
    composite = ref_canvas * (1 - mask_a) + padded_a * mask_a
    composite = composite * (1 - mask_b) + padded_b * mask_b
    composite = composite * (1 - mask_c) + padded_c * mask_c
    composite = composite * (1 - mask_d) + padded_d * mask_d
    composite = composite * (1 - mask_e) + padded_e * mask_e
    return composite

def apply_image_6cut(ref_canvas, image_a, image_b, image_c, image_d, image_e, image_f, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST"):
    H, W = ref_canvas.shape[1:3]
    
    imgs = [image_a, image_b, image_c, image_d, image_e, image_f]
    resized = []
    for img in imgs:
        if mode == "vertical":
            resized.append(resize_keep_ratio(img, target_w=W-32, resize_set=resize_set))
        else:
            resized.append(resize_keep_ratio(img, target_h=H-32, resize_set=resize_set))

    padded = [apply_padding(img, top=16, bottom=16, left=16, right=16, pad_color=pad_color) for img in resized]
    return tuple(padded)

def composite_fixed_6cut(ref_canvas, padded_a, padded_b, padded_c, padded_d, padded_e, padded_f, masks):
    keys = ["A", "B", "C", "D", "E", "F"]
    paddings = [padded_a, padded_b, padded_c, padded_d, padded_e, padded_f]
    
    composite = ref_canvas.clone()
    for key, pad_img in zip(keys, paddings):
        if key in masks:
            m = masks[key].float().repeat(1, 1, 1, 3)
            composite = composite * (1 - m) + pad_img * m
    return composite

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

def cv_resize_upsize(img_np, target_w, target_h, method="LANCZOS"):
    h, w = img_np.shape[:2]
    
    interp = cv2.INTER_LANCZOS4 if method == "LANCZOS" else cv2.INTER_NEAREST
    
    resized = cv2.resize(img_np, (target_w, target_h), interpolation=interp)
    
    contours = image_to_vector(img_np)
    scaled_contours = resize_vector(contours, target_w, target_h, w, h)
    
    vector_enhanced = vector_to_image(scaled_contours, target_w, target_h, base_image=resized, thickness=1)
    return vector_enhanced

def cv_resize_downsize(img_np, target_w, target_h, method="PIXELBOX"):
    h, w = img_np.shape[:2]
    
    interp = cv2.INTER_AREA if method == "PIXELBOX" else cv2.INTER_CUBIC
    resized = cv2.resize(img_np, (target_w, target_h), interpolation=interp)
    
    if target_w >= 256 and target_h >= 256:
        contours = image_to_vector(img_np)
        scaled_contours = resize_vector(contours, target_w, target_h, w, h)
        vector_enhanced = vector_to_image(scaled_contours, target_w, target_h, base_image=resized, thickness=1)
        return vector_enhanced
    else:
        return resized

def composite_dynamic_cuts(working_canvas, slot_masks, slot_images):

    for key, mask_tensor in slot_masks.items():
        img_tensor = slot_images.get(key)
        if img_tensor is None:
            continue

        mask_np = mask_tensor.squeeze().cpu().numpy()
        y_indices, x_indices = np.where(mask_np > 0.5)
        if len(y_indices) == 0 or len(x_indices) == 0:
            continue

        y_min, y_max = y_indices.min(), y_indices.max()
        x_min, x_max = x_indices.min(), x_indices.max()
        box_h = (y_max - y_min) + 1
        box_w = (x_max - x_min) + 1

        img_np = img_tensor.squeeze(0).cpu().numpy()
        if img_np.max() <= 1.0:
            img_np = (img_np * 255.0).astype(np.uint8)
        else:
            img_np = img_np.astype(np.uint8)

        orig_h, orig_w = img_np.shape[:2]

        if box_w > orig_w or box_h > orig_h:
            processed_np = cv_resize_upsize(img_np, box_w, box_h, method="LANCZOS")
        elif box_w < orig_w or box_h < orig_w:
            processed_np = cv_resize_downsize(img_np, box_w, box_h, method="PIXELBOX")
        else:
            processed_np = img_np

        processed_tensor = torch.from_numpy(processed_np).float() / 255.0
        processed_tensor = processed_tensor.unsqueeze(0).to(working_canvas.device)

        m = mask_tensor.float().repeat(1, 1, 1, 3).to(working_canvas.device)
        slot_canvas = working_canvas[:, y_min:y_max+1, x_min:x_max+1, :]
        
        blended = slot_canvas * (1.0 - m[:, y_min:y_max+1, x_min:x_max+1, :]) + processed_tensor * m[:, y_min:y_max+1, x_min:x_max+1, :]
        working_canvas[:, y_min:y_max+1, x_min:x_max+1, :] = blended

    return working_canvas
# -------------------------------

class IRL_Imagecomposite(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Imagecomposite",
            display_name="이미지 합성(통합)",
            description="두 이미지를 선택한 방식에 맞게 처리합니다.",
            inputs=[
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Float.Input("factor", default=0.5, min=0.0, max=1.0, step=0.01,
                               tooltip="Blend 모드 처리 비율. 수치를 손대지 않을경우 무시됩니다."),
                IO.Float.Input("strength", default=1.0, min=0.0, max=1.0, step=0.01,
                               tooltip="블렌드 효과 강도"),
                IO.Float.Input("priority_a", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 A의 우선도"),
                IO.Float.Input("priority_b", default=1.0, min=0.0, max=3.0, step=0.1,
                               tooltip="이미지 B의 우선도"),
                IO.Mask.Input("mask", tooltip="합성에 사용할 마스크 이미지", optional=True),
                IO.Combo.Input("blend_mode", options=["Blend", "Overlay", "Add", "Multiply", "Difference"], default="Blend", tooltip="합성 방식"),
                IO.Combo.Input("Mask_mode", options=["normal", "Small_spread", "big_spread", "blur"], default="normal", tooltip="마스크 적용 모드"),
                IO.Float.Input("saturation", default=0.00, min=-1.00, max=1.00, step=0.01,
                               tooltip="마스크 영역 채도 조정"),
                IO.Float.Input("out_satur", default=0.00, min=-1.00, max=1.00, step=0.01,
                               tooltip="마스크 반전 영역 채도 조정")
            ],
            outputs=[IO.Image.Output("image", tooltip="두 이미지의 합성 결과")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, image_a, image_b, factor=0.5, strength=1.0, priority_a=1.0, priority_b=1.0, mask=None, blend_mode="Blend", Mask_mode="normal", saturation=0.00, out_satur=0.00) -> IO.NodeOutput:
        factor = max(0.00, min(1.00, factor))
        strength = max(0.00, min(1.00, strength))
        priority_a = max(0.00, min(3.00, priority_a))
        priority_b = max(0.00, min(3.00, priority_b))
        saturation = max(-1.00, min(1.00, saturation))
        a = to_torch_image(image_a)
        b = to_torch_image(image_b)

        if mask is not None:
            m = process_mask(mask, a, Mask_mode)
        else:
            m = torch.ones_like(a)

        blended = apply_blend_mode(a, b, blend_mode, factor)
        blended = blended * m + a * (1 - m)
        
        result = apply_weighted(a, blended, priority_a, priority_b, strength)

        if saturation != 0.0:
            blended_np = torch_to_numpy(blended)
            hsv = cv2.cvtColor(blended_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            s = s * (1.0 + saturation)
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
            blended_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            blended_sat_t = numpy_to_torch(blended_sat)

            result = result * (1 - m) + blended_sat_t * m

        if out_satur != 0.0:
            blended_np = torch_to_numpy(blended)
            hsv = cv2.cvtColor(blended_np, cv2.COLOR_RGB2HSV).astype(np.float32)
            h, s, v = cv2.split(hsv)
            s = s * (1.0 + out_satur)
            s = np.clip(s, 0, 255)
            hsv = cv2.merge([h.astype(np.uint8), s.astype(np.uint8), v.astype(np.uint8)])
            blended_sat = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

            blended_sat_t = numpy_to_torch(blended_sat)

            result = result * m + blended_sat_t * (1 - m)

        return IO.NodeOutput(result)

# -------------------------------

class IRL_Imagecutcomposite(IO.ComfyNode):
    
    CUT_TYPE = "2cut"
    BASE_DIR = os.path.dirname(__file__)
    CUT_LAYOUT_DIR = os.path.join(BASE_DIR, "cut_layout")

    layout_dir = os.path.join(CUT_LAYOUT_DIR, CUT_TYPE)
    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(layout_dir):
        for f in os.listdir(layout_dir):
            full_path = os.path.join(layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Imagecutcomposite",
            display_name="이미지 컷 레이아웃(2컷)",
            description="이미지를 선택한 컷 레이아웃에 맞게 배치합니다.",
            inputs=[
                IO.Combo.Input("cut_layout", options=cls.files, default=cls.files[0] if cls.files else "no ref_canvas", tooltip="합성에 사용할 레퍼런스 이미지"),
                IO.Combo.Input("mode", options=["vertical", "horizontal"], default="vertical", tooltip="레이아웃 적용 모드"),
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.String.Input("pad_color", default="#FFFFFF", tooltip="삽입 이미지 패딩 처리 색상"),
                IO.Combo.Input("resize_set", options=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], default="NEAREST", tooltip="리사이징 세팅. 이미지가 컷보다 작으면 켜지지 않습니다."),
            ],
            outputs=[IO.Image.Output("image", tooltip="이미지 컷 배치 적용 결과물")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, cut_layout, image_a, image_b, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST") -> IO.NodeOutput:

        ref_canvas = load_reference_canvas(cls.CUT_TYPE, cut_layout)

        if mode == "horizontal":
            # (vertical → horizontal)
            ref_canvas = torch.rot90(ref_canvas, k=1, dims=(1,2))
        else:
           # (default = vertical)
            pass

        working_canvas = ref_canvas.clone()
        
        canvas_np = ref_canvas.squeeze(0).cpu().numpy()
        if canvas_np.max() <= 1.0:
            canvas_np = (canvas_np * 255.0).astype(np.uint8)
        else:
            canvas_np = canvas_np.astype(np.uint8)

        slot_masks = make_fixed_masks(canvas_np, tolerance=10)

        slot_images = {
            "A": image_a,
            "B": image_b
        }

        working_canvas = composite_dynamic_cuts(working_canvas, slot_masks, slot_images)
        return IO.NodeOutput(working_canvas)


# -------------------------------

class IRL_Image3cutcomposite(IO.ComfyNode):
    
    CUT_TYPE = "3cut"
    BASE_DIR = os.path.dirname(__file__)
    CUT_LAYOUT_DIR = os.path.join(BASE_DIR, "cut_layout")

    layout_dir = os.path.join(CUT_LAYOUT_DIR, CUT_TYPE)
    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(layout_dir):
        for f in os.listdir(layout_dir):
            full_path = os.path.join(layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Image3cutcomposite",
            display_name="이미지 컷 레이아웃(3컷)",
            description="이미지를 선택한 컷 레이아웃에 맞게 배치합니다.",
            inputs=[
                IO.Combo.Input("cut_layout", options=cls.files, default=cls.files[0] if cls.files else "no ref_canvas", tooltip="합성에 사용할 레퍼런스 이미지"),
                IO.Combo.Input("mode", options=["vertical", "horizontal"], default="vertical", tooltip="레이아웃 적용 모드"),
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Image.Input("image_c", tooltip="세 번째 입력 이미지"),
                IO.String.Input("pad_color", default="#FFFFFF", tooltip="삽입 이미지 패딩 처리 색상"),
                IO.Combo.Input("resize_set", options=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], default="NEAREST", tooltip="리사이징 세팅. 이미지가 컷보다 작으면 켜지지 않습니다."),
            ],
            outputs=[IO.Image.Output("image", tooltip="이미지 컷 배치 적용 결과물")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, cut_layout, image_a, image_b, image_c, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST") -> IO.NodeOutput:

        ref_canvas = load_reference_canvas(cls.CUT_TYPE, cut_layout)

        if mode == "horizontal":
            # (vertical → horizontal)
            ref_canvas = torch.rot90(ref_canvas, k=1, dims=(1,2))
        else:
           # (default = vertical)
            pass

        working_canvas = ref_canvas.clone()
        
        canvas_np = ref_canvas.squeeze(0).cpu().numpy()
        if canvas_np.max() <= 1.0:
            canvas_np = (canvas_np * 255.0).astype(np.uint8)
        else:
            canvas_np = canvas_np.astype(np.uint8)

        slot_masks = make_fixed_masks(canvas_np, tolerance=10)

        slot_images = {
            "A": image_a,
            "B": image_b,
            "C": image_c
        }

        working_canvas = composite_dynamic_cuts(working_canvas, slot_masks, slot_images)
        return IO.NodeOutput(working_canvas)

# -------------------------------

class IRL_Image4cutcomposite(IO.ComfyNode):
    
    CUT_TYPE = "4cut"
    BASE_DIR = os.path.dirname(__file__)
    CUT_LAYOUT_DIR = os.path.join(BASE_DIR, "cut_layout")

    layout_dir = os.path.join(CUT_LAYOUT_DIR, CUT_TYPE)
    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(layout_dir):
        for f in os.listdir(layout_dir):
            full_path = os.path.join(layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Image4cutcomposite",
            display_name="이미지 컷 레이아웃(4컷)",
            description="이미지를 선택한 컷 레이아웃에 맞게 배치합니다.",
            inputs=[
                IO.Combo.Input("cut_layout", options=cls.files, default=cls.files[0] if cls.files else "no ref_canvas", tooltip="합성에 사용할 레퍼런스 이미지"),
                IO.Combo.Input("mode", options=["vertical", "horizontal"], default="vertical", tooltip="레이아웃 적용 모드"),
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Image.Input("image_c", tooltip="세 번째 입력 이미지"),
                IO.Image.Input("image_d", tooltip="네 번째 입력 이미지"),
                IO.String.Input("pad_color", default="#FFFFFF", tooltip="삽입 이미지 패딩 처리 색상"),
                IO.Combo.Input("resize_set", options=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], default="NEAREST", tooltip="리사이징 세팅. 이미지가 컷보다 작으면 켜지지 않습니다."),
            ],
            outputs=[IO.Image.Output("image", tooltip="이미지 컷 배치 적용 결과물")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, cut_layout, image_a, image_b, image_c, image_d, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST") -> IO.NodeOutput:

        ref_canvas = load_reference_canvas(cls.CUT_TYPE, cut_layout)

        if mode == "horizontal":
            # (vertical → horizontal)
            ref_canvas = torch.rot90(ref_canvas, k=1, dims=(1,2))
        else:
           # (default = vertical)
            pass

        working_canvas = ref_canvas.clone()
        
        canvas_np = ref_canvas.squeeze(0).cpu().numpy()
        if canvas_np.max() <= 1.0:
            canvas_np = (canvas_np * 255.0).astype(np.uint8)
        else:
            canvas_np = canvas_np.astype(np.uint8)

        slot_masks = make_fixed_masks(canvas_np, tolerance=10)

        slot_images = {
            "A": image_a,
            "B": image_b,
            "C": image_c,
            "D": image_d
        }

        working_canvas = composite_dynamic_cuts(working_canvas, slot_masks, slot_images)
        return IO.NodeOutput(working_canvas)

# -------------------------------

class IRL_Image5cutcomposite(IO.ComfyNode):
    
    CUT_TYPE = "5cut"
    BASE_DIR = os.path.dirname(__file__)
    CUT_LAYOUT_DIR = os.path.join(BASE_DIR, "cut_layout")

    layout_dir = os.path.join(CUT_LAYOUT_DIR, CUT_TYPE)
    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(layout_dir):
        for f in os.listdir(layout_dir):
            full_path = os.path.join(layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Image5cutcomposite",
            display_name="이미지 컷 레이아웃(5컷)",
            description="이미지를 선택한 컷 레이아웃에 맞게 배치합니다.",
            inputs=[
                IO.Combo.Input("cut_layout", options=cls.files, default=cls.files[0] if cls.files else "no ref_canvas", tooltip="합성에 사용할 레퍼런스 이미지"),
                IO.Combo.Input("mode", options=["vertical", "horizontal"], default="vertical", tooltip="레이아웃 적용 모드"),
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Image.Input("image_c", tooltip="세 번째 입력 이미지"),
                IO.Image.Input("image_d", tooltip="네 번째 입력 이미지"),
                IO.Image.Input("image_e", tooltip="다섯 번째 입력 이미지"),
                IO.String.Input("pad_color", default="#FFFFFF", tooltip="삽입 이미지 패딩 처리 색상"),
                IO.Combo.Input("resize_set", options=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], default="NEAREST", tooltip="리사이징 세팅. 이미지가 컷보다 작으면 켜지지 않습니다."),
            ],
            outputs=[IO.Image.Output("image", tooltip="이미지 컷 배치 적용 결과물")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, cut_layout, image_a, image_b, image_c, image_d, image_e, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST") -> IO.NodeOutput:

        ref_canvas = load_reference_canvas(cls.CUT_TYPE, cut_layout)

        if mode == "horizontal":
            # (vertical → horizontal)
            ref_canvas = torch.rot90(ref_canvas, k=1, dims=(1,2))
        else:
           # (default = vertical)
            pass

        working_canvas = ref_canvas.clone()
        
        canvas_np = ref_canvas.squeeze(0).cpu().numpy()
        if canvas_np.max() <= 1.0:
            canvas_np = (canvas_np * 255.0).astype(np.uint8)
        else:
            canvas_np = canvas_np.astype(np.uint8)

        slot_masks = make_fixed_masks(canvas_np, tolerance=10)

        slot_images = {
            "A": image_a,
            "B": image_b,
            "C": image_c,
            "D": image_d,
            "E": image_e
        }

        working_canvas = composite_dynamic_cuts(working_canvas, slot_masks, slot_images)
        return IO.NodeOutput(working_canvas)


# -------------------------------

class IRL_Image6cutcomposite(IO.ComfyNode):
    
    CUT_TYPE = "6cut"
    BASE_DIR = os.path.dirname(__file__)
    CUT_LAYOUT_DIR = os.path.join(BASE_DIR, "cut_layout")

    layout_dir = os.path.join(CUT_LAYOUT_DIR, CUT_TYPE)
    files = []
    VALID_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
    if os.path.exists(layout_dir):
        for f in os.listdir(layout_dir):
            full_path = os.path.join(layout_dir, f)
            if os.path.isfile(full_path) and f.lower().endswith(VALID_EXTENSIONS):
                files.append(f)

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_Image6cutcomposite",
            display_name="이미지 컷 레이아웃(6컷)",
            description="이미지를 선택한 컷 레이아웃에 맞게 배치합니다.",
            inputs=[
                IO.Combo.Input("cut_layout", options=cls.files, default=cls.files[0] if cls.files else "no ref_canvas", tooltip="합성에 사용할 레퍼런스 이미지"),
                IO.Combo.Input("mode", options=["vertical", "horizontal"], default="vertical", tooltip="레이아웃 적용 모드"),
                IO.Image.Input("image_a", tooltip="첫 번째 입력 이미지"),
                IO.Image.Input("image_b", tooltip="두 번째 입력 이미지"),
                IO.Image.Input("image_c", tooltip="세 번째 입력 이미지"),
                IO.Image.Input("image_d", tooltip="네 번째 입력 이미지"),
                IO.Image.Input("image_e", tooltip="다섯 번째 입력 이미지"),
                IO.Image.Input("image_f", tooltip="여섯 번째 입력 이미지"),
                IO.String.Input("pad_color", default="#FFFFFF", tooltip="삽입 이미지 패딩 처리 색상"),
                IO.Combo.Input("resize_set", options=["NEAREST", "BILINEAR", "BICUBIC", "LANCZOS"], default="NEAREST", tooltip="리사이징 세팅. 이미지가 컷보다 작으면 켜지지 않습니다."),
            ],
            outputs=[IO.Image.Output("image", tooltip="이미지 컷 배치 적용 결과물")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, cut_layout, image_a, image_b, image_c, image_d, image_e, image_f, mode="vertical", pad_color="#FFFFFF", resize_set="NEAREST") -> IO.NodeOutput:

        ref_canvas = load_reference_canvas(cls.CUT_TYPE, cut_layout)

        if mode == "horizontal":
            # (vertical → horizontal)
            ref_canvas = torch.rot90(ref_canvas, k=1, dims=(1,2))
        else:
           # (default = vertical)
            pass

        working_canvas = ref_canvas.clone()
        
        canvas_np = ref_canvas.squeeze(0).cpu().numpy()
        if canvas_np.max() <= 1.0:
            canvas_np = (canvas_np * 255.0).astype(np.uint8)
        else:
            canvas_np = canvas_np.astype(np.uint8)

        slot_masks = make_fixed_masks(canvas_np, tolerance=10)

        slot_images = {
            "A": image_a,
            "B": image_b,
            "C": image_c,
            "D": image_d,
            "E": image_e,
            "F": image_f,
        }

        working_canvas = composite_dynamic_cuts(working_canvas, slot_masks, slot_images)

        return IO.NodeOutput(working_canvas)

# -------------------------------

class IRL_ImagecutPreper(IO.ComfyNode):

    COLOR_Keys = {
    "A": (255, 0, 0),   # R → image_a
    "B": (0, 255, 0),   # G → image_b
    "C": (0, 0, 255),   # B → image_C
    "D": (255, 255, 0),   # Y → image_D
    "E": (255, 0, 255),   # Magenta → image_E
    "F": (0, 255, 255),   # CYan → image_F
}
    @classmethod
    def ensure_image_tensor(cls, arr):
        if not isinstance(arr, torch.Tensor):
            arr = torch.from_numpy(np.array(arr)).float()

        if arr.dim() == 2:
            arr = arr.unsqueeze(0).unsqueeze(-1)

        elif arr.dim() == 3:
            arr = arr.unsqueeze(0)

        elif arr.dim() == 4:
            pass
            
        else:
            raise ValueError(f"Unsupported image shape: {arr.shape}")

        if arr.shape[1] > 4 and arr.shape[3] <= 4:
            # (B, H, W, C) -> (B, C, H, W)
            arr = arr.permute(0, 3, 1, 2)
        elif arr.shape[1] > 4 and arr.shape[2] <= 4:
            # (B, C, H, W)
            arr = arr.permute(0, 2, 1, 3)
            
        return arr.float()

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImagecutPreper",
            display_name="이미지 컷 프레퍼",
            description="간이 컷 레이아웃 슬롯을 만듭니다. 이 노드로 만든 컷 레이아웃은 커스텀 컷 노드에선 바로 쓸 수 있습니다.",
            inputs=[
                IO.Combo.Input("cut_layout", options=["2cut", "3cut", "4cut", "5cut", "6cut"], default="2cut", tooltip="준비할 컷의 개수"),
                IO.Int.Input("mask1_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 1의 X축 길이"),
                IO.Int.Input("mask1_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 1의 Y축 길이"),
                IO.Int.Input("left1_x", default=1, min=0, max=2048, step=1, tooltip="마스크 1을 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top1_y", default=1, min=0, max=2048, step=1, tooltip="마스크 1을 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("mask2_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 2의 X축 길이"),
                IO.Int.Input("mask2_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 2의 Y축 길이"),
                IO.Int.Input("left2_x", default=1, min=0, max=2048, step=1, tooltip="마스크 2을 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top2_y", default=1, min=0, max=2048, step=1, tooltip="마스크 2을 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("mask3_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 3의 X축 길이"),
                IO.Int.Input("mask3_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 3의 Y축 길이"),
                IO.Int.Input("left3_x", default=1, min=0, max=2048, step=1, tooltip="마스크 3을 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top3_y", default=1, min=0, max=2048, step=1, tooltip="마스크 3을 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("mask4_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 4의 X축 길이"),
                IO.Int.Input("mask4_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 4의 Y축 길이"),
                IO.Int.Input("left4_x", default=1, min=0, max=2048, step=1, tooltip="마스크 4을 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top4_y", default=1, min=0, max=2048, step=1, tooltip="마스크 4을 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("mask5_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 5의 X축 길이"),
                IO.Int.Input("mask5_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 5의 Y축 길이"),
                IO.Int.Input("left5_x", default=1, min=0, max=2048, step=1, tooltip="마스크 5을 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top5_y", default=1, min=0, max=2048, step=1, tooltip="마스크 5을 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("mask6_x", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 6의 X축 길이"),
                IO.Int.Input("mask6_y", default=8, min=1, max=2048, step=1, tooltip="생성할 마스크 6의 Y축 길이"),
                IO.Int.Input("left6_x", default=1, min=0, max=2048, step=1, tooltip="마스크 6을 좌측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("top6_y", default=1, min=0, max=2048, step=1, tooltip="마스크 6을 상부측으로 어느지점에 배치할지의 지정. 축의 생성길이가 캔버스보다 작아질 경우 여백은 보정됩니다."),
                IO.Int.Input("line_size", default=1, min=1, max=5, step=1, tooltip="컷 라인 두께 지정"),
                IO.String.Input("line_color", default="#000000", tooltip="컷 라인 색상. #xxxxxx 형태의 헥스값을 받습니다."),
                IO.Int.Input("canvas_x", default=544, min=1, max=4096, step=1, tooltip="생성할 캔버스의 X축 길이"),
                IO.Int.Input("canvas_y", default=1072, min=1, max=4096, step=1, tooltip="생성할 캔버스의 Y축 길이"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="이미지 위에 마스크 영역을 오버레이하여 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[IO.Image.Output("image", tooltip="이미지 컷 배치 적용 결과물")],
            category="이미지 리파이너/합성"
        )

    @classmethod
    def execute(cls, cut_layout, mask1_x, mask1_y, left1_x, top1_y, mask2_x, mask2_y, left2_x, top2_y, mask3_x, mask3_y, left3_x, top3_y, 
                mask4_x, mask4_y, left4_x, top4_y, mask5_x, mask5_y, left5_x, top5_y, mask6_x, mask6_y, left6_x, top6_y,
                line_size, line_color="#000000", canvas_x=544, canvas_y=1072, preview_mode=False) -> IO.NodeOutput:

        # 1. base canvas (B, C, H, W)
        canvas = torch.ones((1, 3, canvas_y, canvas_x), dtype=torch.float32)

        # 2. set cut type
        cut_count_map = {"2cut": 2, "3cut": 3, "4cut": 4, "5cut": 5, "6cut": 6}
        active_cuts = cut_count_map.get(cut_layout, 2)

        cuts = [
            (left1_x, top1_y, mask1_x, mask1_y),
            (left2_x, top2_y, mask2_x, mask2_y),
            (left3_x, top3_y, mask3_x, mask3_y),
            (left4_x, top4_y, mask4_x, mask4_y),
            (left5_x, top5_y, mask5_x, mask5_y),
            (left6_x, top6_y, mask6_x, mask6_y),
        ]

        color_keys = ["A", "B", "C", "D", "E", "F"]

        hex_color = line_color.lstrip('#')
        rgb_line = tuple(int(hex_color[i:i+2], 16) / 255.0 for i in (0, 2, 4))
        line_rgb = torch.tensor(rgb_line, dtype=torch.float32).view(1, 3, 1, 1)

        for i in range(active_cuts):
            lx, ty, mx, my = cuts[i]
            if mx <= 0 or my <= 0:
                continue
            
            x1 = max(0, min(lx, canvas_x))
            y1 = max(0, min(ty, canvas_y))
            x2 = max(0, min(lx + mx, canvas_x))
            y2 = max(0, min(ty + my, canvas_y))

            c_key = color_keys[i % len(color_keys)]
            slot_rgb = [val / 255.0 for val in cls.COLOR_Keys[c_key]]
            c_val = torch.tensor(slot_rgb, dtype=torch.float32).view(1, 3, 1, 1)
            canvas[:, :, y1:y2, x1:x2] = c_val

            ls = line_size
            if ls > 0:
                canvas[:, :, y1:y1+ls, x1:x2] = line_rgb
                canvas[:, :, y2-ls:y2, x1:x2] = line_rgb
                canvas[:, :, y1:y2, x1:x1+ls] = line_rgb
                canvas[:, :, y1:y2, x2-ls:x2] = line_rgb

        output = canvas.permute(0, 2, 3, 1)

        if preview_mode:

            return IO.NodeOutput(output,ui=UI.PreviewImage(output))
        return IO.NodeOutput(output)

# -------------------------------
class IRL_ImagecutCompositeCustom(IO.ComfyNode):

    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImagecutCompositeCustom",
            display_name="이미지 컷 컴포짓 커스텀",
            description="프레퍼 노드가 만든 컬러 레이아웃을 기반으로 이미지를 각 컷 슬롯에 합성합니다.",
            inputs=[
                IO.Image.Input("canvas_image", tooltip="IRL_ImagecutPreper 노드에서 출력된 레이아웃 캔버스"),
                IO.Image.Input("image_a", tooltip="A 슬롯 이미지"),
                IO.Image.Input("image_b", tooltip="B 슬롯 이미지", optional=True),
                IO.Image.Input("image_c", tooltip="C 슬롯 이미지", optional=True),
                IO.Image.Input("image_d", tooltip="D 슬롯 이미지", optional=True),
                IO.Image.Input("image_e", tooltip="E 슬롯 이미지", optional=True),
                IO.Image.Input("image_f", tooltip="F 슬롯 이미지", optional=True),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="최종 합성된 컷 이미지"),
            ],
            category="이미지 리파이너/합성"
        )
    @classmethod
    def execute(cls, canvas_image, image_a, image_b=None, image_c=None, 
                image_d=None, image_e=None, image_f=None) -> IO.NodeOutput:
        working_canvas = canvas_image.clone()

        canvas_np = canvas_image.squeeze(0).cpu().numpy()
        if canvas_np.max() <= 1.0:
            canvas_np = (canvas_np * 255.0).astype(np.uint8)
        else:
            canvas_np = canvas_np.astype(np.uint8)

        slot_masks = make_fixed_masks(canvas_np, tolerance=10)

        slot_images = {
            "A": image_a,
            "B": image_b,
            "C": image_c,
            "D": image_d,
            "E": image_e,
            "F": image_f,
        }

        working_canvas = composite_dynamic_cuts(working_canvas, slot_masks, slot_images)

        return IO.NodeOutput(working_canvas)

# -------------------------------
 
COMPOSITE_NODE_CLASS_MAPPINGS = {
    "IRL_Imagecomposite": IRL_Imagecomposite,
    "IRL_Imagecutcomposite": IRL_Imagecutcomposite,
    "IRL_Image3cutcomposite": IRL_Image3cutcomposite,
    "IRL_Image4cutcomposite": IRL_Image4cutcomposite,
    "IRL_Image5cutcomposite": IRL_Image5cutcomposite,
    "IRL_Image6cutcomposite": IRL_Image6cutcomposite,
    "IRL_ImagecutPreper": IRL_ImagecutPreper,
    "IRL_ImagecutCompositeCustom": IRL_ImagecutCompositeCustom,
}

COMPOSITE_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_Imagecomposite": "이미지 합성(통합)",
    "IRL_Imagecutcomposite": "이미지 컷 레이아웃(2컷)",
    "IRL_Image3cutcomposite": "이미지 컷 레이아웃(3컷)",
    "IRL_Image4cutcomposite": "이미지 컷 레이아웃(4컷)",
    "IRL_Image5cutcomposite": "이미지 컷 레이아웃(5컷)",
    "IRL_Image6cutcomposite": "이미지 컷 레이아웃(6컷)",
    "IRL_ImagecutPreper": "이미지 컷 프레퍼",
    "IRL_ImagecutCompositeCustom": "이미지 컷 컴포짓 커스텀",
}

# -------------------------------
