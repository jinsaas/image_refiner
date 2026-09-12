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
import re
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
            description="이미지 픽셀값의 평균과 표준편차를 계산해 노드 위젯에 표시합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="분석할 이미지"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="텍스트 픽셀 크기 (8~48)")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        mean_value, std_value = float(gray.mean()), float(gray.std())
        stats_text = f"MEAN: {mean_value:.4f}\nSTD: {std_value:.4f}"

        # Send the value to the UI channel so that the frontend widget updates immediately
        return IO.NodeOutput(ui={"stats": [stats_text], "font_size": [font_size]})

# -------------------------------

class IRL_ImageMinMax(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageMinMax",
            display_name="이미지 최소 & 최대",
            description="이미지 픽셀값의 최소값과 최대값을 계산합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="분석할 이미지"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="텍스트 픽셀 크기 (8~48)")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=14) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        min_value, max_value = float(gray.min()), float(gray.max())

        stats_text = f"ImagePixelsMIN: {min_value:.4f}\nImagePixelsMAX: {max_value:.4f}"

        # Send the value to the UI channel so that the frontend widget updates immediately
        return IO.NodeOutput( ui={"stats": [stats_text], "font_size": [font_size]})


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
                IO.Float.Input("edge_scale", default=1.0, min=0.1, max=5.0, step=0.1, tooltip="엣지 강도 스케일"),
                IO.Float.Input("threshold", default=50.0, min=0.0, max=255.0, step=1.0, tooltip="이진화 임계값"),
                IO.Boolean.Input("preview_mode", default=False, tooltip="생성 결과를 노드에서 미리보기")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("image", tooltip="엣지 맵 이미지"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, edge_scale=1.0, threshold=50.0, preview_mode=False) -> IO.NodeOutput:
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
        
        edge = np.clip(edge, 0, 255)
        # Binary Thresholding
        edge = np.where(edge > threshold, 255, 0).astype(np.uint8)
        canvas = Image.fromarray(edge)

        output = to_tensor_output(canvas)
        if preview_mode:
            return IO.NodeOutput(output,ui=UI.PreviewImage(output))
        return IO.NodeOutput(output)

# -------------------------------

class IRL_CustomDepthlikeMapGenerator(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_CustomDepthlikeMapGenerator",
            display_name="커스텀 유사 뎁스 맵 생성기",
            description="엣지 맵과 거리 변환을 활용해 입체적인 간이 뎁스 맵을 생성합니다.\n"
                        "워터셰드 옵션을 켜면, 뎁스 맵을 근거로 서로 붙어있는 영역을 분리한 마스크도 함께 출력합니다.\n"
                        "엣지 필을 통해 구멍이 뚫린 엣지를 보강할 수 있으며, 엣지 필 도트포인트의 크기는 '각 변의 최소크기*0.01'입니다.",
            inputs=[
                IO.Image.Input("image", tooltip="대상 이미지"),
                IO.Float.Input("threshold", default=50.0, min=0.0, max=255.0, step=1.0, tooltip="엣지 이진화 임계값"),
                IO.Boolean.Input("blur_enabled", default=False, tooltip="블러 처리 여부(고주파 잔선을 지우고 굵은 구조만 남깁니다)"),
                IO.Int.Input("morph_kernel", default=3, min=1, max=11, step=2, tooltip="끊어진 선을 이어주는 닫기(Closing) 연산 커널 크기"),
                IO.String.Input("edge_fill_coords", default="", multiline=True, tooltip="벽을 세울 좌표들. 예: (205, 260), (150, 300)"),
                IO.Boolean.Input("invert_target", default=False, tooltip="꺼짐(기본): 영역 내부가 기준 → 중심이 밝고 경계로 갈수록 어두워짐(일반적인 뎁스 형태).\n"
                                                                    "켜짐: 엣지선이 기준 → 선 자체를 강조하는 형태로 반전됩니다."),
                IO.Boolean.Input("enable_watershed", default=False, tooltip="뎁스 맵의 국소 최댓값을 각 영역의 중심 씨앗으로 삼아, 붙어있는 영역들을 자동으로 나눕니다."),
                IO.Boolean.Input("edge_fill", default=False, tooltip="켜짐: 지정한 좌표(seed_x, y)를 기준으로 도트 찍기 동작.\n꺼짐: 기본 엣지/실루엣 정보 그대로 사용"),
                IO.Int.Input("min_marker_distance", default=40, min=1, max=200, step=1, tooltip="영역 중심으로 인식할 최소 간격 (enable_watershed 켰을 때만 작동)"),
                IO.Boolean.Input("preview_guide", default=True, tooltip="True면 가이드 프리뷰 출력, False면 일반 뎁스 출력"),
                IO.Float.Input("overwrite_gray", default=0.0, min=0.0, max=0.5, step=0.1, tooltip="0.1 이상인 경우 엣지정보에 그레이스케일을 포함, 0.0이면 일반 엣지로 출력"),
                IO.Float.Input("clahe_clip_limit", default=1.5, min=0.0, max=8.0, step=0.5,
                    tooltip="지역 대비 보정 강도. 낮을수록 원본에 가깝고, 높을수록 어두운 영역의 미세한 단차까지 강하게 끌어올립니다."),
                IO.Int.Input("clahe_tile_size", default=16, min=4, max=64, step=4,
                    tooltip="대비 보정을 계산하는 타일 크기(정사각형 한 변). 클수록 부드럽고 넓은 영역 기준으로 보정, 작을수록 국소적이고 거칠게 보정됩니다."),
                IO.Combo.Input("depth_style", options=["distance", "gaussian_blur", "blended"], default="distance",
                    tooltip="depth_map(출력 이미지)의 스타일을 결정합니다.\n"
                            "distance: 거리 변환 기반 (부위별 깊이 차등 뚜렷, 다소 거칠 수 있음)\n"
                            "gaussian_blur: 실루엣을 부드럽게 블러 (매끈하지만 부위 간 깊이 차이는 사라짐)\n"
                            "blended: 두 결과를 절반씩 섞음")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[
                IO.Image.Output("depth_map", tooltip="생성된 유사 뎁스 맵"),
                IO.Mask.Output("region_mask", tooltip="워터셰드로 분리된 영역 마스크 (꺼져있으면 빈 마스크)"),
            ],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, threshold=50.0, blur_enabled=False, edge_fill_coords="", morph_kernel=3, invert_target=False, enable_watershed=False, edge_fill=False, 
                min_marker_distance=40, preview_guide=False, overwrite_gray=0.0, clahe_clip_limit=0.0, clahe_tile_size=16, depth_style="distance") -> IO.NodeOutput:
        arr = to_numpy_image(image)

        # 1. RGB to GRAY & Heavy Gaussian Blur
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        gray = gray.astype(np.uint8)
        H, W = gray.shape[:2]
        
        safe_max_distance = max(1, min(H, W) // 3)
        min_marker_distance = min(min_marker_distance, safe_max_distance)

        if blur_enabled:
            # A minimal, light stabilization blur that doesn’t compromise detail (sigma=1.0)
            gray_blurred = cv2.GaussianBlur(gray, (0, 0), sigmaX=1.0)
        else:
            gray_blurred = gray


        clahe_clip_limit = min(8.0,max(clahe_clip_limit,0.0))
        clahe_tile_size = min(64,max(clahe_tile_size,4))
        if clahe_clip_limit > 0.0:
            clahe = cv2.createCLAHE(clipLimit=clahe_clip_limit, tileGridSize=(clahe_tile_size,clahe_tile_size))
            gray_eq = clahe.apply(gray_blurred)
        else:
            gray_eq = gray_blurred
        # 2. Sobel Edge Extraction (Edges = 255, Inside = 0)
        sobel_x = cv2.Sobel(gray_eq, cv2.CV_32F, 1, 0)
        sobel_y = cv2.Sobel(gray_eq, cv2.CV_32F, 0, 1)
        edge = np.sqrt(sobel_x**2 + sobel_y**2)
        edge_p99 = np.percentile(edge, 99.0)
        edge = np.clip(edge / (edge_p99 + 1e-6) * 255, 0, 255).astype(np.uint8)

        overwrite_gray = min(0.5,max(overwrite_gray,0.0))
        if overwrite_gray > 0.0:
            # 2-1. Overwrite/Blend Edge onto Grayscale
            # Overwrite the Sobel edge onto the grayscale image to firmly imprint the boundary wall
            gray_overlaid = cv2.addWeighted(edge, 1.0, gray, overwrite_gray, 0)

            _, edge_binary = cv2.threshold(gray_overlaid, threshold, 255, cv2.THRESH_BINARY)

        else:
            _, edge_binary = cv2.threshold(edge, threshold, 255, cv2.THRESH_BINARY)

        # 3. Morphology Closing
        if morph_kernel > 1:
            mk_size = morph_kernel if morph_kernel % 2 == 1 else morph_kernel + 1
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (mk_size, mk_size))
            thick_edges = cv2.morphologyEx(edge_binary, cv2.MORPH_CLOSE, kernel)
        else:
            thick_edges = edge_binary

        # --- Coordinate Processing & Guide Canvas ---
        guide_canvas = cv2.cvtColor(thick_edges, cv2.COLOR_GRAY2RGB)
        
        for i in range(10):
            x_pos = int(W * i / 9)
            y_pos = int(H * i / 9)
            
            cv2.line(guide_canvas, (x_pos, 0), (x_pos, H), (40, 80, 40), 1)
            cv2.line(guide_canvas, (0, y_pos), (W, y_pos), (40, 80, 40), 1)
            
            if i == 0:
                cv2.putText(guide_canvas, f"X{i}:{x_pos}", (x_pos + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                cv2.putText(guide_canvas, f"Y{i}:{y_pos}", (2, y_pos + 12), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
            elif i == 9:
                cv2.putText(guide_canvas, f"X{i}:{x_pos}", (max(0, x_pos - 55), 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
                cv2.putText(guide_canvas, f"Y{i}:{y_pos}", (2, max(12, y_pos - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 100, 100), 1)
            else:
                cv2.putText(guide_canvas, f"{i}:{x_pos}", (x_pos + 2, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 0), 1)
                cv2.putText(guide_canvas, f"{i}:{y_pos}", (2, y_pos - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (200, 200, 0), 1)

        valid_coords = []
        if edge_fill_coords.strip():
            invalid_chars = re.sub(r'[\d,\(\)\s]', '', edge_fill_coords)
            if invalid_chars:
                print(f"[IRL_CustomDepthlikeMapGenerator] Warning: Contains unallowed key ('{invalid_chars}') and is therefore filtered.")
            
            raw_pairs = re.findall(r'\(?\s*(\d+)\s*,\s*(\d+)\s*\)?', edge_fill_coords)
            for p in raw_pairs:
                try:
                    x, y = int(p[0]), int(p[1])
                    if 0 <= x < W and 0 <= y < H:
                        valid_coords.append((x, y))
                except ValueError:
                    pass

        # --- Solid Silhouette Generation ---

        if edge_fill:
            for vx, vy in valid_coords:
                cv2.circle(thick_edges, (vx, vy), radius=max(3, min(H, W) // 100), color=255, thickness=-1)

        if edge_fill:
            for idx, (vx, vy) in enumerate(valid_coords):
                cv2.drawMarker(guide_canvas, (vx, vy), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=15, thickness=2)
                cv2.putText(guide_canvas, f"W{idx+1}({vx},{vy})", (vx + 5, vy - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1)

        padded_edges = cv2.copyMakeBorder(thick_edges, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
        padded_H, padded_W = padded_edges.shape[:2]
        
        contours, _ = cv2.findContours(padded_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        base_silhouette_padded = np.zeros((padded_H, padded_W), dtype=np.uint8)
        
        if contours:
            cv2.drawContours(base_silhouette_padded, contours, -1, 255, thickness=-1)

        base_silhouette = base_silhouette_padded[1:1+H, 1:1+W]

        processed_silhouette = base_silhouette.copy()

        # 4 Create Depth Map
        # 4-1. Distance Transform
        def compute_distance_map():
            if invert_target:
                dist_input = cv2.GaussianBlur(cv2.bitwise_not(processed_silhouette), (5, 5), sigmaX=2.0)
            else:
                dist_input = cv2.GaussianBlur(processed_silhouette, (5, 5), sigmaX=2.0)
            dt = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
            if dt.max() > 0:
                norm_dt = dt / (dt.max() + 1e-6)
                return (np.power(norm_dt, 1.2) * 255).astype(np.uint8)
            return gray.copy()

        # 4-2. Shading Volume
        def compute_shading_map():
            masked_gray = cv2.bitwise_and(gray_eq, gray_eq, mask=processed_silhouette)
            if invert_target:
                masked_gray = 255 - masked_gray
            smoothed = cv2.GaussianBlur(masked_gray, (15, 15), sigmaX=5.0)
            if processed_silhouette.sum() > 0:
                valid_vals = smoothed[processed_silhouette > 0]
                v_min, v_max = valid_vals.min(), valid_vals.max()
                if v_max > v_min:
                    smoothed = np.clip((smoothed - v_min) / (v_max - v_min) * 255, 0, 255).astype(np.uint8)
            return np.where(processed_silhouette > 0, smoothed, 0).astype(np.uint8)

        # 4-3. Gaussian Blur
        def compute_blur_map():
            blurred = cv2.GaussianBlur(processed_silhouette, (31, 31), sigmaX=10.0)
            if invert_target:
                blurred = 255 - blurred
            return blurred.astype(np.uint8)

        # Create a depth map according to style
        if depth_style == "distance":
            depth_map = compute_distance_map()
        elif depth_style == "shading_volume":
            depth_map = compute_shading_map()
        elif depth_style == "gaussian_blur":
            depth_map = compute_blur_map()
        elif depth_style == "blended":
            d_map = compute_distance_map()
            s_map = compute_shading_map()
            depth_map = cv2.addWeighted(d_map, 0.5, s_map, 0.5, 0)
        else:
            depth_map = compute_distance_map()

        if invert_target:
            dist_input = cv2.GaussianBlur(cv2.bitwise_not(processed_silhouette), (5, 5), sigmaX=2.0)
        else:
            dist_input = cv2.GaussianBlur(processed_silhouette, (5, 5), sigmaX=2.0)

        dist_transform = cv2.distanceTransform(dist_input, cv2.DIST_L2, 5)
        
        if dist_transform.max() > 0:
            norm_dist = dist_transform / (dist_transform.max() + 1e-6)
            depth_map = (np.power(norm_dist, 1.2) * 255).astype(np.uint8)
        else:
            depth_map = gray.copy()
        
        depth_map = depth_map.astype(np.uint8)

        # 5. Watershed Region Mask
        region_mask_out = processed_silhouette.copy()
        if enable_watershed:
            processed_silhouette = processed_silhouette.astype(np.uint8)
            seed_threshold = int(threshold * 0.8)
            _, strong_peaks = cv2.threshold(depth_map, seed_threshold, 255, cv2.THRESH_BINARY)
            strong_peaks = cv2.bitwise_and(strong_peaks, processed_silhouette)
            
            dynamic_marker_dist = max(3, min_marker_distance // 2)
            kernel_marker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (dynamic_marker_dist, dynamic_marker_dist))
            strong_peaks = cv2.erode(strong_peaks, kernel_marker)

            num_labels, markers = cv2.connectedComponents(strong_peaks)
            if num_labels > 1:
                markers_int = np.zeros((H, W), dtype=np.int32)
                for i in range(1, num_labels):
                    markers_int[markers == i] = i + 1
                
                # The real background outside the silhouette is explicitly number 1 (background marker)
                markers_int[processed_silhouette == 0] = 1

                arr_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR) if arr.ndim == 3 else cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
                cv2.watershed(arr_bgr, markers_int)

                # Extract only the object area (>1) in white (255), and set the border (-1) and background (1) to 0
                region_mask_out = np.where(markers_int > 1, 255, 0).astype(np.uint8)
            else:
                region_mask_out = processed_silhouette.copy()
        else:
            region_mask_out = processed_silhouette.copy()

        # 6. Pillow image convert to 'Simple depth-like map'
        canvas = Image.fromarray(depth_map).convert("RGB")
        guide_out_img = Image.fromarray(guide_canvas)
        mask_out = torch.from_numpy(region_mask_out.astype(np.float32) / 255.0).unsqueeze(0)

        tensor_depth = to_tensor_output(canvas)
        tensor_guide = to_tensor_output(guide_out_img)

        if preview_guide:
            return IO.NodeOutput(tensor_depth, mask_out, ui=UI.PreviewImage(tensor_guide))
        else:
            return IO.NodeOutput(tensor_depth, mask_out)

# -------------------------------

class IRL_ImageBrightnessContrast(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_ImageBrightnessContrast",
            display_name="밝기 & 대비",
            description="이미지의 밝기와 대비를 계산합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="분석할 이미지"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="텍스트 픽셀 크기 (8~48)")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=14) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        brightness_value, contrast_value = float(gray.mean()), float(gray.std())

        stats_text = f"Brightness: {brightness_value:.4f}\nContrast: {contrast_value:.4f}"

        # Send the value to the UI channel so that the frontend widget updates immediately
        return IO.NodeOutput(ui={"stats": [stats_text], "font_size": [font_size]})



# -------------------------------
class IRL_CannyEdgeStats(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_CannyEdgeStats",
            display_name="캐니 에지 통계",
            description="캐니 에지 검출을 통해 에지 밀도와 평균을 계산합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="케니 엣지를 검출해 분석할 이미지"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="텍스트 픽셀 크기 (8~48)")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_size=14) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = (0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]).astype(np.uint8) if arr.ndim == 3 else arr.astype(np.uint8)

        # Canny edge detection
        edges = cv2.Canny(gray, 100, 200).astype(np.float32)
        edge_density = float((edges > 0).mean())
        edge_mean = float(edges.mean() / 255.0)

        stats_text = f"EDGE_DENSITY: {edge_density:.4f}\nEDGE_MEAN: {edge_mean:.4f}"

        # Send the value to the UI channel so that the frontend widget updates immediately
        return IO.NodeOutput(ui={"stats": [stats_text], "font_size": [font_size]})


# -------------------------------

class IRL_DepthStats(IO.ComfyNode):
    @classmethod
    def define_schema(cls):
        return IO.Schema(
            node_id="IRL_DepthStats",
            display_name="깊이 통계",
            description="이미지의 깊이 평균과 표준편차를 계산합니다.\n"
                        "font 폴더 내의 폰트를 자동으로 탐색하여 적용합니다.",
            inputs=[
                IO.Image.Input("image", tooltip="뎁스 평균값과 표준편차를 분석할 이미지"),
                IO.Int.Input("font_size", default=14, min=8, max=48, tooltip="텍스트 픽셀 크기 (8~48)")
            ],
            hidden=[IO.Hidden.prompt, IO.Hidden.extra_pnginfo],
            is_output_node=True,
            outputs=[],
            category="이미지 리파이너/분석"
        )

    @classmethod
    def execute(cls, image, font_name="arial", font_size=40) -> IO.NodeOutput:
        arr = to_numpy_image(image)
        gray = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2] if arr.ndim == 3 else arr
        arr = gray.astype(np.float32) / 255.0
        depth_mean, depth_std = float(arr.mean()), float(arr.std())

        stats_text = f"DEPTH_MEAN: {depth_mean:.4f}\nDEPTH_STD: {depth_std:.4f}"

        # Send the value to the UI channel so that the frontend widget updates immediately
        return IO.NodeOutput(ui={"stats": [stats_text], "font_size": [font_size]})


# -------------------------------

ANALYSIS_NODE_CLASS_MAPPINGS = {
    "IRL_RGBSplit": IRL_RGBSplit,
    "IRL_HistogramPlot": IRL_HistogramPlot,
    "IRL_ImageMeanStd": IRL_ImageMeanStd,
    "IRL_ImageMinMax": IRL_ImageMinMax,
    "IRL_ImageEdgeMap": IRL_ImageEdgeMap,
    "IRL_CustomDepthlikeMapGenerator": IRL_CustomDepthlikeMapGenerator,
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
    "IRL_CustomDepthlikeMapGenerator": "커스텀 유사 뎁스 맵 생성기",
    "IRL_ImageBrightnessContrast": "밝기 & 대비",
    "IRL_CannyEdgeStats": "캐니 에지 통계",
    "IRL_DepthStats": "깊이 통계",
}

# -------------------------------
