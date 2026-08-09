# -------------------------------
# IR Lite — Noise Nodes
# -------------------------------

import numpy as np
from PIL import Image
import torch
import cv2
import hashlib
import math
import random
import gc

from comfy_api.latest import IO, UI
from comfy_api.latest._io_public import ComfyTypeIO, comfytype, Custom

# ---------------------------------------
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

def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]#add batch


    return torch.from_numpy(arr)

def to_numpy_image(image):
    if isinstance(image, torch.Tensor):
        arr = image[0].cpu().numpy()
        if arr.max() <= 1.0:
            arr = (arr * 255).clip(0, 255).astype(np.uint8)
        else:
            arr = arr.astype(np.uint8)
        return arr
    elif isinstance(image, Image.Image):
        return np.array(image.convert("RGB"))
    elif isinstance(image, np.ndarray):
        return image.astype(np.uint8)
    else:
        raise TypeError("Unsupported image type")
#
# Noise logics
#--------------------------------
def GaussianNoise(width, height, scale, seed, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    noise_map = rng.normal(0, sigma, size=(height, width))
    return noise_map.astype(np.float32)

def PerlinNoise(width, height, scale, seed, sigma, rng=None):
    #parameter settings
    octaves = 4
    persistence = 0.5

    if rng is None:
        rng = np.random.default_rng(seed)
    
    if scale <= 0:
        scale = 1.0

    perm = np.arange(256, dtype=int)
    rng.shuffle(perm)
    perm = np.tile(perm, 4)

    def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
    def lerp(a, b, t): return a + t * (b - a)

    def grad_vectorized(hash_map, x, y):
        h = hash_map & 3
        u = np.where(h < 2, x, y)
        v = np.where(h < 2, y, x)
        res_u = np.where((h & 1) == 0, u, -u)
        res_v = np.where((h & 2) == 0, v, -v)
        return res_u + res_v

    def perlin_vectorized(grid_x, grid_y):
        xi = grid_x.astype(int) & 255
        yi = grid_y.astype(int) & 255
        
        xf = grid_x - np.floor(grid_x)
        yf = grid_y - np.floor(grid_y)
        
        u = fade(xf)
        v = fade(yf)

        p_x = perm[xi]
        p_x_next = perm[(xi + 1) & 255]

        aa = perm[(p_x + yi) % 256]
        ab = perm[(p_x + yi + 1) % 256]
        ba = perm[(p_x_next + yi) % 256]
        bb = perm[(p_x_next + yi + 1) % 256]

        x1 = lerp(grad_vectorized(aa, xf, yf), grad_vectorized(ba, xf - 1, yf), u)
        x2 = lerp(grad_vectorized(ab, xf, yf - 1), grad_vectorized(bb, xf - 1, yf - 1), u)
        return (lerp(x1, x2, v) + 1.0) / 2.0

    X = np.arange(width, dtype=np.float32) / scale
    Y = np.arange(height, dtype=np.float32) / scale
    base_grid_x, base_grid_y = np.meshgrid(X, Y)

    total = np.zeros((height, width), dtype=np.float32)
    frequency = 1.0
    amplitude = 1.0
    max_value = 0.0

    for _ in range(octaves):
        total += perlin_vectorized(base_grid_x * frequency, base_grid_y * frequency) * amplitude
        max_value += amplitude
        amplitude *= persistence
        frequency *= 2.0

    noise_map = total / max_value
    noise_map = noise_map - np.mean(noise_map)
    noise_std = np.std(noise_map)
    if noise_std > 1e-8:
        noise_map = noise_map / noise_std

    return noise_map.astype(np.float32)

def WhiteNoise(width, height, scale, seed, sigma, rng=None):
    #parameter settings
    intensity=1.0
    if rng is None:
        rng = np.random.default_rng(seed)
    noise_map = rng.uniform(-intensity, intensity, size=(height, width))
    return noise_map.astype(np.float32)

def SaltPepperNoise(width, height, scale, seed, sigma, rng=None):
    #parameter settings
    amount = 0.02
    if rng is None:
        local_rng = np.random.default_rng(seed)
    else:
        local_rng = rng

    noise_map = np.zeros((height, width), dtype=np.float32)
    num_salt = int(amount * height * width * 0.5)    
    num_pepper = int(amount * height * width * 0.5)

    def get_randint(low, high, size):
        if hasattr(local_rng, "integers"):
            return local_rng.integers(low, high, size)
        else:
            return local_rng.randint(low, high, size)
    coords_s = [local_rng.integers(0, height, num_salt), local_rng.integers(0, width, num_salt)]
    noise_map[tuple(coords_s)] = 1.0

    coords_p = [local_rng.integers(0, height, num_pepper), local_rng.integers(0, width, num_pepper)]
    noise_map[tuple(coords_p)] = -1.0
    return noise_map.astype(np.float32)

def RandomColor(width, height, scale, seed, sigma, rng=None):
    if rng is None:
        rng = np.random.default_rng(seed)
    noise_map = rng.normal(0, 1.0, (height, width))
    return noise_map.astype(np.float32)

def generate_hybrid_texture_noise_2d(width, height, scale, seed, sigma, rng=None):

    if rng is None:
        rng = np.random.default_rng(seed) 

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


def generate_hybrid_texture_noise_3d(width, height, depth, scale, seed, sigma, rng=None):
    #parameter settings
    scale = 8

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


# -------------------------------
# Gaussian Noise
# -------------------------------

class IRL_AddGaussianNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_AddGaussianNoise",
            display_name="가우시안 노이즈 추가",
            description="시드 통제가 가능한 정밀 가우시안 노이즈를 이미지에 추가합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="노이즈를 추가할 이미지"),
                IO.Float.Input("sigma", default=10.0, min=0.0, max=255.0, step=0.1,
                               tooltip="가우시안 노이즈의 표준편차 (0~255 범위 권장)"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, tooltip="0이면 랜덤, 숫자를 지정하면 고정된 노이즈 패턴 생성"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="노이즈가 추가된 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, image, sigma, seedset=0, show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        arr = to_numpy_image(image).astype(np.float32)

        parsed_seed = par_seed(seedset)
        if parsed_seed == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        else:
            base_seed = parsed_seed

        rng = np.random.default_rng(base_seed)

        noise = rng.normal(0, sigma, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(noisy)

        result_rgb = to_tensor_output(pil_img)  
        if show_preview:
            return IO.NodeOutput(result_rgb,ui=UI.PreviewImage(result_rgb))
        return IO.NodeOutput(result_rgb)


# -------------------------------
# Salt & Pepper Noise 
# -------------------------------

class IRL_SaltPepperNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_SaltPepperNoise",
            display_name="소금 & 후추 노이즈",
            description="컬러 오염 없이 정확한 흑백 소금&후추 노이즈를 고정 시드로 추가합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="노이즈를 추가할 이미지"),
                IO.Float.Input("amount", default=0.010, min=0.000, max=1.000, step=0.001,
                               tooltip="노이즈의 비율 (0~1)"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, tooltip="0이면 랜덤, 숫자를 지정하면 고정된 노이즈 패턴 생성"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="노이즈가 추가된 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, image, amount, seedset=0, show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        arr = to_numpy_image(image)
        h, w, c = arr.shape
        noisy = arr.copy()

        parsed_seed = par_seed(seedset)
        if parsed_seed == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        else:
            base_seed = parsed_seed

        rng = np.random.default_rng(base_seed)

        num_pixels = h * w
        num_salt = int(np.ceil(amount * num_pixels * 0.5))
        num_pepper = int(np.ceil(amount * num_pixels * 0.5))

        if num_salt > 0:
            salt_y = rng.integers(0, h, num_salt)
            salt_x = rng.integers(0, w, num_salt)
            noisy[salt_y, salt_x, :] = 255

        if num_pepper > 0:
            pepper_y = rng.integers(0, h, num_pepper)
            pepper_x = rng.integers(0, w, num_pepper)
            noisy[pepper_y, pepper_x, :] = 0

        pil_img = Image.fromarray(noisy)
        result_rgb = to_tensor_output(pil_img)  
        if show_preview:
            return IO.NodeOutput(result_rgb,ui=UI.PreviewImage(result_rgb))
        return IO.NodeOutput(result_rgb)


# -------------------------------
# Perlin Noise 
# -------------------------------

class IRL_PerlinNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_PerlinNoise",
            display_name="퍼린 노이즈",
            description="인덱스 크래시 없이, 격자 배열 연산으로 프랙탈 퍼린 노이즈 패턴을 초고속 생성합니다.",
            inputs=[
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
                IO.Float.Input("scale", default=32.0, min=4.0, max=128.0, step=1.0, tooltip="노이즈 스케일 (패턴 크기 조절)"),
                IO.Int.Input("octaves", default=4, min=1, max=8, tooltip="옥타브 수 (패턴 레이어 수)"),
                IO.Float.Input("persistence", default=0.5, min=0.1, max=1.0, step=0.1, tooltip="옥타브별 세기 감소율"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, tooltip="0이면 랜덤, 숫자를 지정하면 고정된 노이즈 패턴 생성"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="생성된 퍼린 노이즈 이미지 텐서"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, width, height, scale, octaves, persistence, seedset=0, show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        if scale <= 0: 
            scale = 1.0

        parsed_seed = par_seed(seedset)
        if parsed_seed == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        else:
            base_seed = parsed_seed

        rng = np.random.default_rng(base_seed)

        perm = np.arange(256, dtype=int)
        rng.shuffle(perm)
        perm = np.tile(perm, 4)

        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        def lerp(a, b, t): return a + t * (b - a)

        def grad_vectorized(hash_map, x, y):
            h = hash_map & 3
            u = np.where(h < 2, x, y)
            v = np.where(h < 2, y, x)
            res_u = np.where((h & 1) == 0, u, -u)
            res_v = np.where((h & 2) == 0, v, -v)
            return res_u + res_v

        def perlin_vectorized(grid_x, grid_y):
            xi = grid_x.astype(int) & 255
            yi = grid_y.astype(int) & 255
            
            xf = grid_x - np.floor(grid_x)
            yf = grid_y - np.floor(grid_y)
            
            u = fade(xf)
            v = fade(yf)

            p_x = perm[xi]
            p_x_next = perm[(xi + 1) & 255]

            aa = perm[(p_x + yi) % 256]
            ab = perm[(p_x + yi + 1) % 256]
            ba = perm[(p_x_next + yi) % 256]
            bb = perm[(p_x_next + yi + 1) % 256]

            x1 = lerp(grad_vectorized(aa, xf, yf), grad_vectorized(ba, xf - 1, yf), u)
            x2 = lerp(grad_vectorized(ab, xf, yf - 1), grad_vectorized(bb, xf - 1, yf - 1), u)
            return (lerp(x1, x2, v) + 1.0) / 2.0

        X = np.arange(width, dtype=np.float32) / scale
        Y = np.arange(height, dtype=np.float32) / scale
        base_grid_x, base_grid_y = np.meshgrid(X, Y)

        total = np.zeros((height, width), dtype=np.float32)
        frequency = 1.0
        amplitude = 1.0
        max_value = 0.0

        for _ in range(octaves):
            total += perlin_vectorized(base_grid_x * frequency, base_grid_y * frequency) * amplitude
            max_value += amplitude
            amplitude *= persistence
            frequency *= 2.0

        arr = total / max_value

        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        arr_rgb = np.stack([arr] * 3, axis=-1)

        out_tensor = torch.from_numpy(arr_rgb).float() / 255.0
        out_tensor = out_tensor.unsqueeze(0) # [1, H, W, C]
        if show_preview:
            result_rgb = to_tensor_output(out_tensor)  
            return IO.NodeOutput(out_tensor,ui=UI.PreviewImage(result_rgb))
        return IO.NodeOutput(out_tensor)


# -------------------------------
# Random Color 
# -------------------------------

class IRL_RandomColor(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_RandomColor",
            display_name="랜덤 컬러 이미지",
            description="시드 통제가 가능한 무작위 컬러 화이트 노이즈 이미지를 초고속으로 생성합니다.",
            inputs=[
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, tooltip="0이면 랜덤, 숫자를 지정하면 고정된 노이즈 패턴 생성"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="랜덤 색상으로 채워진 이미지 텐서"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, width, height, seedset=0, show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        parsed_seed = par_seed(seedset)
        if parsed_seed == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        else:
            base_seed = parsed_seed

        rng = np.random.default_rng(base_seed)

        arr = rng.integers(0, 256, (height, width, 3), dtype=np.uint8)

        out_tensor = torch.from_numpy(arr).float() / 255.0
        out_tensor = out_tensor.unsqueeze(0)  # [1, H, W, C] 
        if show_preview:
            result_rgb = to_tensor_output(out_tensor) 
            return IO.NodeOutput(out_tensor,ui=UI.PreviewImage(out_tensor))
        return IO.NodeOutput(out_tensor)



# -------------------------------
# White Noise 
# -------------------------------
class IRL_WhiteNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_WhiteNoise",
            display_name="화이트 노이즈",
            description="아날로그 질감을 가진 커스텀 노이즈 패턴 이미지를 초고속으로 생성합니다.",
            inputs=[
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
                IO.Float.Input("scale", default=8.0, min=1.0, max=64.0, step=1.0, tooltip="노이즈 스케일 (패턴 크기 조절)"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, tooltip="0이면 랜덤, 숫자를 지정하면 고정된 노이즈 패턴 생성"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="생성된 노이즈 이미지 텐서"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, width, height, scale, seedset=0, show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        if scale <= 0: 
            scale = 1.0

        parsed_seed = par_seed(seedset)
        if parsed_seed == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        else:
            base_seed = parsed_seed

        rng = np.random.default_rng(base_seed)

        perm = np.arange(256, dtype=int)
        rng.shuffle(perm)
        perm = np.tile(perm, 4)

        X = np.arange(width, dtype=np.float32) / scale
        Y = np.arange(height, dtype=np.float32) / scale
        grid_x, grid_y = np.meshgrid(X, Y)

        xi = grid_x.astype(int) & 255
        yi = grid_y.astype(int) & 255
        
        xf = grid_x - np.floor(grid_x)
        yf = grid_y - np.floor(grid_y)

        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        def lerp(a, b, t): return a + t * (b - a)
        
        u = fade(xf)
        v = fade(yf)

        p_x = perm[xi]
        p_x_next = perm[(xi + 1) & 255] 

        aa = perm[(p_x + yi) % 256]
        ab = perm[(p_x + yi + 1) % 256]
        ba = perm[(p_x_next + yi) % 256]
        bb = perm[(p_x_next + yi + 1) % 256]

        def grad_np(h, x, y):
            match_0 = (h & 3) == 0
            match_1 = (h & 3) == 1
            u_val = np.where(match_0 | match_1, x, y)
            v_val = np.where(match_0 | match_1, y, x)
            
            res_u = np.where((h & 1) == 0, u_val, -u_val)
            res_v = np.where((h & 2) == 0, v_val, -v_val)
            return res_u + res_v

        x1 = lerp(grad_np(aa, xf, yf), grad_np(ba, xf - 1, yf), u)
        x2 = lerp(grad_np(ab, xf, yf - 1), grad_np(bb, xf - 1, yf - 1), u)
        arr = (lerp(x1, x2, v) + 1.0) / 2.0

        arr = np.clip(arr * 255, 0, 255).astype(np.uint8)
        arr_rgb = np.stack([arr] * 3, axis=-1)

        out_tensor = torch.from_numpy(arr_rgb).float() / 255.0
        out_tensor = out_tensor.unsqueeze(0) # [1, H, W, C]

 
        if show_preview:
            result_rgb = to_tensor_output(out_tensor) 
            return IO.NodeOutput(out_tensor,ui=UI.PreviewImage(out_tensor))
        return IO.NodeOutput(out_tensor)


# -------------------------------
# RGBColor (independent generator)
# -------------------------------

class IRL_RGBColor(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_RGBColor",
            display_name="컬러 이미지",
            description="메모리 낭비 없이, 지정된 RGB 색상으로 채워진 단색 배경 캔버스를 초고속 생성합니다.",
            inputs=[
                IO.Int.Input("str_r", default=0, min=0, max=255, tooltip="적색 영역"),
                IO.Int.Input("str_g", default=0, min=0, max=255, tooltip="녹색 영역"),
                IO.Int.Input("str_b", default=0, min=0, max=255, tooltip="청색 영역"),
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="단일 색상으로 채워진 이미지 텐서"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, str_r, str_g, str_b, width, height, show_preview=False, clear_cache=False) -> IO.NodeOutput:

        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        r_val = max(0, min(255, str_r)) / 255.0
        g_val = max(0, min(255, str_g)) / 255.0
        b_val = max(0, min(255, str_b)) / 255.0
        width = max(16, min(2048, width))
        height = max(16, min(2048, height))

        base_color_tensor = torch.tensor([r_val, g_val, b_val], dtype=torch.float32) # [3]
        

        out_tensor = base_color_tensor.view(1, 3, 1, 1).expand(1, 3, height, width) # [1, C, H, W]
        if show_preview:
            result_rgb = to_tensor_output(out_tensor) 
            return IO.NodeOutput(out_tensor.contiguous(),ui=UI.PreviewImage(out_tensor))
        return IO.NodeOutput(out_tensor.contiguous())

# -------------------------------

class IRL_NoiseCreator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_NoiseCreator",
            display_name="노이즈 생성기",
            description="시드 통제가 가능한 정밀 노이즈를 준비합니다.",
            inputs=[
                IO.Combo.Input("noise_mode", options=["GaussianNoise", "SaltPepperNoise", "PerlinNoise", "RandomColor", "WhiteNoise", "generate_hybrid_texture_noise"], default="GaussianNoise"),
                IO.Int.Input("seedset", default=0, min=0, max=2**31 - 1, tooltip="0이면 랜덤, 숫자를 지정하면 고정된 노이즈 패턴 생성"),
                IO.Boolean.Input("noise_switch", default=False, tooltip="샘플러가 받을수 있는 정보로서 시드와 노이즈 정보를 전달합니다."),
                IO.Boolean.Input("show_preview", default=False, tooltip="프리뷰 표시 여부"),
                IO.Boolean.Input("clear_cache", default=False, tooltip="노드 시작시 캐시 정리")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                NoisePack.Output("noise_pack", tooltip="노이즈 정보"),
                IO.Int.Output("seedset", tooltip="노이즈 시드 코드")
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, noise_mode="GaussianNoise", seedset=0, noise_switch=False, show_preview=False, clear_cache=False) -> IO.NodeOutput:
        if clear_cache:
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
                    print("GPU cache initialization complete.")
                except Exception as e:
                    print("GPU cache initialization failed:", e)
            else:
                print("CPU/Non-CUDA mode: Skip GPU cache initialization")

            gc.collect()
            print("CPU cache initialization complete")

        #parameter settings
        scale = 8
        sigma=10
        height, width = 512, 512
        base_color_tensor = torch.tensor([255, 255, 255], dtype=torch.float32) # [3]
        image = base_color_tensor.view(1, 3, 1, 1).expand(1, 3, height, width)

        arr = to_numpy_image(image).astype(np.float32)

        parsed_seed = par_seed(seedset)
        if parsed_seed == 0:
            base_seed = int(np.random.default_rng().integers(1, 2**31 - 1))
        else:
            base_seed = parsed_seed
        print(f"\n{CYAN}{BOLD}[IRL_NoiseCreator]{RESET} Active Seed: {YELLOW}{BOLD}{base_seed}{RESET}")

        rng = np.random.default_rng(base_seed)

        if noise_mode == "GaussianNoise":
            texture_np = GaussianNoise(width, height, scale, base_seed, sigma, rng=rng)

        if noise_mode == "SaltPepperNoise":
            texture_np = SaltPepperNoise(width, height, scale, base_seed, sigma, rng=rng)

        if noise_mode == "PerlinNoise":
            texture_np = PerlinNoise(width, height, scale, base_seed, sigma, rng=rng)

        if noise_mode == "RandomColor":
            texture_np = RandomColor(width, height, scale, base_seed, sigma, rng=rng)

        if noise_mode == "WhiteNoise":
            texture_np = WhiteNoise(width, height, scale, base_seed, sigma, rng=rng)

        if noise_mode == "generate_hybrid_texture_noise":
            texture_np = generate_hybrid_texture_noise_2d(width, height, scale, base_seed, sigma, rng=rng)

        noisepack = str(noise_mode)

        if show_preview:
            norm_map = texture_np - texture_np.min()
            map_max = norm_map.max()
            if map_max > 1e-8:
                norm_map = norm_map / map_max
            preview_arr = np.clip(norm_map * 255.0, 0, 255).astype(np.uint8)
            preview_rgb = np.stack([preview_arr] * 3, axis=-1)
            
            pil_img = Image.fromarray(preview_rgb)
            result_rgb = to_tensor_output(pil_img)

            return IO.NodeOutput(noisepack, base_seed, ui=UI.PreviewImage(result_rgb))
        return IO.NodeOutput(noisepack, base_seed)

# -------------------------------

NOISE_NODE_CLASS_MAPPINGS = {
    "IRL_AddGaussianNoise": IRL_AddGaussianNoise,
    "IRL_SaltPepperNoise": IRL_SaltPepperNoise,
    "IRL_PerlinNoise": IRL_PerlinNoise,
    "IRL_RandomColor": IRL_RandomColor,
    "IRL_WhiteNoise": IRL_WhiteNoise,
    "IRL_RGBColor": IRL_RGBColor,
    "IRL_NoiseCreator": IRL_NoiseCreator
}

NOISE_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_AddGaussianNoise": "가우시안 노이즈 추가",
    "IRL_SaltPepperNoise": "소금 & 후추 노이즈",
    "IRL_PerlinNoise": "퍼린 노이즈",
    "IRL_RandomColor": "랜덤 컬러 이미지",
    "IRL_WhiteNoise": "화이트 노이즈",
    "IRL_RGBColor": "컬러 이미지",
    "IRL_NoiseCreator": "노이즈 생성기"
}
# -------------------------------
