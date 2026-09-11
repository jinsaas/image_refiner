

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
        
