# -------------------------------
# IR Lite — Noise Nodes
# (LOCALE-based multilingual description support included)
# -------------------------------
import numpy as np
from PIL import Image
import torch
import cv2


# ComfyUI 최신 API
from comfy_api.latest import IO, UI


# -------------------------------

def to_tensor_output(canvas: Image.Image):
    arr = np.array(canvas).astype(np.float32) / 255.0
    arr = arr[None, ...]  # batch 차원 추가


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




# -------------------------------
# Gaussian Noise
# -------------------------------
class IRL_AddGaussianNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_AddGaussianNoise",
            display_name="가우시안 노이즈 추가",
            description="이미지에 가우시안 노이즈를 추가합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="노이즈를 추가할 이미지"),
                IO.Float.Input("sigma", default=0.050, min=0.000, max=200.000, step=0.001,
                               tooltip="가우시안 노이즈의 표준편차"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="노이즈가 추가된 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, image, sigma) -> IO.NodeOutput:
        arr = to_numpy_image(image).astype(np.float32)

        noise = np.random.normal(0, sigma * 5, arr.shape)
        noisy = np.clip(arr + noise, 0, 255).astype(np.uint8)

        pil_img = Image.fromarray(noisy)
        return IO.NodeOutput(to_tensor_output(pil_img))

# -------------------------------
# Salt & Pepper Noise
# -------------------------------
class IRL_SaltPepperNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_SaltPepperNoise",
            display_name="소금 & 후추 노이즈",
            description="이미지에 소금&후추 노이즈를 추가합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="노이즈를 추가할 이미지"),
                IO.Float.Input("amount", default=0.001, min=0.000, max=1.000, step=0.001,
                               tooltip="노이즈의 비율 (0~1)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="노이즈가 추가된 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, image, amount) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        noisy = arr.copy()

        num_salt = int(np.ceil(amount * arr.size * 0.5))
        coords = [np.random.randint(0, i, num_salt) for i in arr.shape]
        noisy[tuple(coords)] = 255

        num_pepper = int(np.ceil(amount * arr.size * 0.5))
        coords = [np.random.randint(0, i, num_pepper) for i in arr.shape]
        noisy[tuple(coords)] = 0

        pil_img = Image.fromarray(noisy)
        return IO.NodeOutput(to_tensor_output(pil_img))

# -------------------------------
# Perlin Noise (independent generator)
# -------------------------------

class IRL_PerlinNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_PerlinNoise",
            display_name="퍼린 노이즈",
            description="퍼린 노이즈 패턴 이미지를 생성합니다.",
            inputs=[
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
                IO.Float.Input("scale", default=32.0, min=4.0, max=128.0, step=1.0,
                               tooltip="노이즈 스케일 (패턴 크기 조절)"),
                IO.Int.Input("octaves", default=4, min=1, max=8, tooltip="옥타브 수 (패턴 레이어 수)"),
                IO.Float.Input("persistence", default=0.5, min=0.1, max=1.0, step=0.1,
                               tooltip="옥타브별 세기 감소율"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="생성된 퍼린 노이즈 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, width, height, scale, octaves, persistence) -> IO.NodeOutput:
        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        def lerp(a, b, t): return a + t * (b - a)
        def grad(hash, x, y):
            h = hash & 3
            u = x if h < 2 else y
            v = y if h < 2 else x
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

        # 고정된 permutation 테이블 (랜덤 셔플 제거)
        perm = np.arange(256, dtype=int)
        perm = np.tile(perm, 2)

        def perlin(x, y):
            xi = int(x) & 255
            yi = int(y) & 255
            xf = x - int(x)
            yf = y - int(y)
            u = fade(xf)
            v = fade(yf)

            aa = perm[perm[xi] + yi]
            ab = perm[perm[xi] + yi + 1]
            ba = perm[perm[xi + 1] + yi]
            bb = perm[perm[xi + 1] + yi + 1]

            x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
            x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)
            return (lerp(x1, x2, v) + 1) / 2

        def fractal_perlin(x, y, octaves, persistence):
            total = 0
            frequency = 1
            amplitude = 1
            max_value = 0
            for _ in range(octaves):
                total += perlin(x * frequency, y * frequency) * amplitude
                max_value += amplitude
                amplitude *= persistence
                frequency *= 2
            return total / max_value

        arr = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                arr[y, x] = fractal_perlin(x / scale, y / scale, octaves, persistence)

        arr = (arr * 255).astype(np.uint8)
        arr_rgb = np.stack([arr] * 3, axis=-1)

        pil_img = Image.fromarray(arr_rgb)
        return IO.NodeOutput(to_tensor_output(pil_img))

# -------------------------------
# Random Color (independent generator)
# -------------------------------
class IRL_RandomColor(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_RandomColor",
            display_name="랜덤 컬러 이미지",
            description="무작위 색상으로 채워진 이미지를 생성합니다.",
            inputs=[
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="랜덤 색상으로 채워진 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, width, height) -> IO.NodeOutput:
        arr = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
        pil_img = Image.fromarray(arr, mode="RGB")
        return IO.NodeOutput(to_tensor_output(pil_img))
        
# -------------------------------
# White Noise (independent generator)
# -------------------------------
class IRL_WhiteNoise(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_WhiteNoise",
            display_name="화이트 노이즈",
            description="화이트 노이즈 패턴 이미지를 생성합니다.",
            inputs=[
                IO.Int.Input("width", default=256, min=16, max=2048, tooltip="출력 이미지의 가로 크기"),
                IO.Int.Input("height", default=256, min=16, max=2048, tooltip="출력 이미지의 세로 크기"),
                IO.Float.Input("scale", default=8.0, min=1.0, max=64.0, step=1.0,
                               tooltip="노이즈 스케일 (패턴 크기 조절)"),
            ],
            outputs=[
                IO.Image.Output("image", tooltip="생성된 화이트 노이즈 이미지"),
            ],
            category="이미지 리파이너/노이즈"
        )

    @classmethod
    def execute(cls, width, height, scale) -> IO.NodeOutput:
        def fade(t): return t * t * t * (t * (t * 6 - 15) + 10)
        def lerp(a, b, t): return a + t * (b - a)
        def grad(hash, x, y):
            h = hash & 3
            u = x if h < 2 else y
            v = y if h < 2 else x
            return (u if (h & 1) == 0 else -u) + (v if (h & 2) == 0 else -v)

        perm = np.arange(256, dtype=int)
        np.random.shuffle(perm)
        perm = np.tile(perm, 2)

        def perlin(x, y):
            xi = int(x) & 255
            yi = int(y) & 255
            xf = x - int(x)
            yf = y - int(y)
            u = fade(xf)
            v = fade(yf)

            aa = perm[perm[xi] + yi]
            ab = perm[perm[xi] + yi + 1]
            ba = perm[perm[xi + 1] + yi]
            bb = perm[perm[xi + 1] + yi + 1]

            x1 = lerp(grad(aa, xf, yf), grad(ba, xf - 1, yf), u)
            x2 = lerp(grad(ab, xf, yf - 1), grad(bb, xf - 1, yf - 1), u)
            return (lerp(x1, x2, v) + 1) / 2

        arr = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                arr[y, x] = perlin(x / scale, y / scale)

        arr = (arr * 255).astype(np.uint8)
        arr_rgb = np.stack([arr] * 3, axis=-1)

        pil_img = Image.fromarray(arr_rgb)
        return IO.NodeOutput(to_tensor_output(pil_img))
        

# -------------------------------

NOISE_NODE_CLASS_MAPPINGS = {
    "IRL_AddGaussianNoise": IRL_AddGaussianNoise,
    "IRL_SaltPepperNoise": IRL_SaltPepperNoise,
    "IRL_PerlinNoise": IRL_PerlinNoise,
    "IRL_RandomColor": IRL_RandomColor,
    "IRL_WhiteNoise": IRL_WhiteNoise,
}

NOISE_NODE_DISPLAY_NAME_MAPPINGS = {
    "IRL_AddGaussianNoise": "가우시안 노이즈 추가",
    "IRL_SaltPepperNoise": "소금 & 후추 노이즈",
    "IRL_PerlinNoise": "퍼린 노이즈",
    "IRL_RandomColor": "랜덤 컬러 이미지",
    "IRL_WhiteNoise": "화이트 노이즈",
}
# -------------------------------