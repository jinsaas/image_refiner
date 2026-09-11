
#### Image Refiner Lite — ComfyUI용 독립 이미지 보정·마스크·텍스트 유틸리티 노드팩####
Image Refiner Lite는 독립 이미지 처리 플러그인입니다.

Stable Diffusion 모델, latent기반 기능을 전부 제거하고 ComfyUI Standalone 환경에서

100% 안정적으로 동작하는 순수 이미지·마스크·텍스트 처리 노드만 제공합니다.

####Image Refiner Lite는 다음 원칙을 기반으로 설계되었습니다####

• 환경 안정성 최우선

• latent/WebUI 잔재 완전 제거

• 순수 이미지 처리 기반

• 결정성 100% (Portable 환경에서도 동일 결과)

• 유지보수 가능한 구조

• 불필요한 의존성 제거

####외부 라이선스 관련####

신규 노드에 comfyui의 샘플러 베이스가 참고된 노드가 있습니다.
ComfyUI(AGPL-3.0)의 텍스트 프롬프트, 샘플러 및 임베딩 노드를 연결하기 위한 로직들을 
기반으로 수정된 버전입니다.
# 원본 프로젝트: https://github.com/comfyanonymous/ComfyUI
comfyui의 라이선스는 외부 라이선스로서 exterlal licenses에 들어가 있습니다.


###requirements.txt가 비어 있는 이유###
ComfyUI는 확장팩 로딩 시 requirements.txt를 자동 설치합니다.
그러나 pip나 git을 통한 원격설치 방식은 환경 손상 위험이 매우 높기 때문에,
Image Refiner Lite판에서는 requirements.txt를 빈 파일로 유지합니다.

대신, 다음의 코드로 최소 의존성 설치법을 남깁니다.

python -m pip install --no-deps simpleeval

python -m pip install --no-deps gitpython

python -m pip install --no-deps pilgram

python -m pip install --no-deps matplotlib


위 패키지들은 라이트판에서 사용하는 안전한 최소 의존성입니다.
설치는 선택 사항이며, 대부분의 기능은 기본 상태에서도 작동합니다.
-
인텔 내장 그래픽 관련 로직은 작동하게 하기 위해선 numpy 버전을 내려야 합니다. 그래서 로직 파기했습니다.
-
#[설치 방법]#



Installation:

1. ZIP 다운로드: 저장소를 .zip 파일로 다운로드합니다.

2. custom_nodes 폴더에 배치: 압축을 풀고 ComfyUI/custom_nodes 폴더에 넣습니다.

3. ComfyUI 재시작: 재시작하면 노드가 로드됩니다.


#[Node Usage Manual]#
텍스트 위젯을 추가해서 노드 자체에서 font파일을 쓸 일이 사라졌기 때문에, 윈도우 폰트가 포함되었던 폴더는 지웠습니다.

[Adjustments Nodes]

IRL_RGBLevels
#색감 강조효과 추가
- node_id: IRL_RGBLevels

- display_name:RGB 레벨

- category: 이미지 리파이너/이미지조정

- 역할: RGB 채널별 레벨 조정 + 색감 강조

- Inputs: image, R Levels, G Levels, B Levels

- Outputs: image



IRL_BlackWhiteLevels

- node_id: IRL_BlackWhiteLevels

- display_name:블랙 & 화이트 레벨

- category: 이미지 리파이너/이미지조정

- 역할: 전체 블랙/화이트 포인트 조정

- Inputs: image, black_point, white_point

- Outputs: image



IRL_LevelsAdjustment 


- node_id: IRL_LevelsAdjustment

- display_name:레벨 조정

- category: 이미지 리파이너/이미지조정

- 역할: 입력/출력 레벨 및 감마 조정

- Inputs: image, in_brightness, gamma, out_brightness

- Outputs: image



IRL_GradientMap

#컬러 코드 직접입력을 고정, 슬라이더로 강도 조절

- node_id: IRL_GradientMap

- display_name:그라디언트 맵

- category: 이미지 리파이너/이미지조정

- 역할: 제시된 색상팔레트들의 조합 및 기존이미지와의 오버레이 처리로 다양한 사용법을 추구합니다.

- Inputs: image, 7color palette, color_str, base_suf, blend_mode, gradient옵션 제공

- Outputs: image


IRL_ShadowsHighlights

- node_id: IRL_ShadowsHighlights

- display_name:그림자 & 하이라이트

- category: 이미지 리파이너/이미지조정

- 역할: 그림자/하이라이트 디테일 복원

- Inputs: image, shadow_amount, highlight_amount

- Outputs: image


[Sampling Nodes]

IRL_ColorTransfer
- node_id: IRL_ColorTransfer

- display_name: 컬러 트랜스퍼

- category: 이미지 리파이너/이미지조정

- 역할: 참조 이미지의 색상 기준으로 원본 이미지 색상값을 교정합니다.

- Inputs: image, palette_image

- Outputs: image

IRL_ImgDetailer
- node_id: IRL_ImgDetailer

- display_name: 이미지 디테일러

- category: 이미지 리파이너/이미지조정

- 역할: 샤프닝, 히스토그램 평활화, 라인 강조 등으로 품질 향상

- Inputs: image, sharpening, histogram_equalize, smoothing, line_strength, line_color(hex), color_transfer

- Outputs: image

IRL_ImgResampler
#변경점. 노이즈 팩 정보를 받아 노이즈 옵션이 가능하게 교체
- node_id: IRL_ImgResampler

- display_name: 이미지 리샘플러

- category: 이미지 리파이너/인페인팅

- 역할: 노이즈 추가 및 디노이즈 재처리, 인코드/디코드 포함

- Inputs: model, clip, vae, image, noise_option, sampler_option

- Outputs: image

IRL_ImgResamplerMix
#변경점. 노이즈 팩 정보를 받아 노이즈 옵션이 가능하게 교체
- node_id: IRL_ImgResamplerMix

- display_name: 이미지 리샘플러(믹스)

- category: 이미지 리파이너/인페인팅

- 역할: 여러 샘플러 옵션을 혼합해 노이즈 추가 및 디노이즈 재처리

- Inputs: model, clip, vae, image, noise_option, sampler_option

- Outputs: image

#변경점. Autoinpaint 노드를 다음 두 노드로 쪼갰습니다.
IRL_InpaintAndMask_CV
- node_id: IRL_InpaintAndMask_CV

- display_name: CV 인페인트 및 마스크 처리

- category: 이미지 리파이너/인페인팅

- 역할: OpenCV 기반 인페인팅, 마스크 영역을 정밀하게 메꿈

- Inputs: image, mask, method, strength, mask_set, mask_mode, show_preview, clear_cache

- Outputs: image

IRL_AutoComposite_Post_CV
- node_id: IRL_AutoComposite_Post_CV

- display_name: CV 오토 콤포짓 및 후처리

- category: 이미지 리파이너/인페인팅

- 역할: OpenCV 기반 후처리, 팔레트 이미지와 합성 가능

- Inputs: image, pal_image, mask, mask_mode, contrast_str, light_balance, color_str, sharpen_str, line_str, line_color, line_mode, blendstr, blendmode, show_preview, clear_cache

- Outputs: image

IRL_ResamplerInpaint
#변경점. 노이즈 팩 정보를 받아 노이즈 옵션이 가능하게 교체
- node_id: IRL_ResamplerInpaint

- display_name: 리샘플러 세미오토 인페인팅

- category: 이미지 리파이너/인페인팅

- 역할: 마스크 기반 인페인팅, 노이즈 및 색상 강조 포함

- Inputs: image, mask, palette_image, noise_setting, color_strength, saturation, line_strength, sharpening, light_balance, contrast

- Outputs: image

IRL_rescaler
- node_id: IRL_rescaler

- display_name: 리스케일러

- category: 이미지 리파이너/인페인팅

- 역할: 다운스케일 후 리스케일을 시도해서 품질 향상을 시도합니다. 256픽셀 이하로 다운스케일을 할 경우에는 큰 효과가 없습니다.

- Inputs: image, 재처리로직, 업스케일러모델 인풋(옵션), 타일링 업스케일 보간(모델 사용시에는 모델위주로 처리), 엣지보간(모델 사용시에는 모델위주로 처리)

- Outputs: image

[Filter Nodes]

IRL_GaussianBlur
- node_id: IRL_GaussianBlur

- display_name: 가우시안 블러

- category: 이미지 리파이너/필터

- 역할: 커널 크기와 시그마 값을 사용하여 이미지에 가우시안 블러 적용

- Inputs: image, kernel_size, sigma

- Outputs: image

IRL_MedianBlur
- node_id: IRL_MedianBlur

- display_name: 미디언 블러

- category: 이미지 리파이너/필터

- 역할: 커널 크기를 사용하여 이미지에 미디언 블러 적용

- Inputs: image, kernel_size

- Outputs: image

IRL_BilateralFilter
- node_id: IRL_BilateralFilter

- display_name: 양방향 필터

- category: 이미지 리파이너/필터

- 역할: 지름과 시그마 값을 사용하여 이미지에 양방향 필터 적용

- Inputs: image, diameter, sigma_color, sigma_space

- Outputs: image

IRL_Sharpen
- node_id: IRL_Sharpen

- display_name: 샤픈

- category: 이미지 리파이너/필터

- 역할: 언샤프 마스크 방식으로 이미지를 선명하게 조정

- Inputs: image, amount

- Outputs: image

IRL_HighPass
- node_id: IRL_HighPass

- display_name: 하이패스 필터

- category: 이미지 리파이너/필터

- 역할: 에지와 세부 사항을 강조하기 위해 하이패스 필터 적용

- Inputs: image, radius

- Outputs: image



[Transform Nodes]



IRL_Resize

- node_id: IRL_Resize

- display_name:리사이즈

- category: 이미지 리파이너/변형

- 역할: 이미지 리사이즈

- Inputs: image, width, height

- Outputs: image


#신규노드
IRL_VecterResize
- node_id: IRL_VecterResize

- display_name: 벡터 리사이즈

- category: 이미지 리파이너/변형

- 역할: 벡터 기반 리사이즈, 컨투어 추출 및 색상 보정 포함

- Inputs: image, width, height, sample_image, method, lineType, contour_threshold, contour_blur

- Outputs: image

#신규노드
IRL_Resize_Upsize_only
- node_id: IRL_Resize_Upsize_only

- display_name: 리사이즈(업사이즈 전용)

- category: 이미지 리파이너/변형

- 역할: 업사이즈 전용 리사이즈

- Inputs: image, width, height, method

- Outputs: image

#신규노드
IRL_Resize_downsize_only
- node_id: IRL_Resize_downsize_only

- display_name: 리사이즈(다운사이즈 전용)

- category: 이미지 리파이너/변형

- 역할: 다운사이즈 전용 리사이즈

- Inputs: image, width, height, method

- Outputs: image


IRL_Rotate
#변경사항: 프리뷰 추가, 패딩 컬러 옵션 추가

- node_id: IRL_Rotate

- display_name:회전

- category: 이미지 리파이너/변형

- 역할: 이미지 회전

- Inputs: image, pad_color, angle

- Outputs: image



IRL_Flip

- node_id: IRL_Flip

- display_name:플립

- category: 이미지 리파이너/변형

- 역할: 이미지 뒤집기 (가로/세로)

- Inputs: image
- Inputs: mode[horizontal,vertical] 수평/수직 중 선택하여 뒤집기
- Outputs: image



IRL_Crop
#변경사항: IRL_Crop은 가이딩 크롭으로 변경되었습니다. 가이드 라인이 표시된 프리뷰를 참고 가능하게 변경되었습니다.

- node_id: IRL_Crop

- display_name:크롭

- category: 이미지 리파이너/변형

- 역할: 지정 영역 크롭

- Inputs: image, x, y, width, height

- Outputs: image



IRL_CropMargins
- node_id: IRL_CropMargins
#변경사항: 잘릴 영역을 프리뷰로 볼 수 있게 변경되었습니다.

- display_name:크롭 마진

- category: 이미지 리파이너/변형

- 역할: 영역 크롭(직접선택)

- Inputs: image, L, R, T, B

- Outputs: image



IRL_PerspectiveWarp

- node_id: IRL_PerspectiveWarp

- display_name:퍼스펙티브 왜곡

- category: 이미지 리파이너/변형

- 역할: 4점 기반 투시 왜곡

- Inputs: image, src_points, dst_points

- Outputs: image

#신규노드 
IRL_MaskColorFill
- node_id: IRL_MaskColorFill

- display_name: 마스크 컬러 채우기

- category: 이미지 리파이너/변형

- 역할: 마스크 영역을 지정 색상으로 채우기

- Inputs: image, mask, fill_color, blend_mode

- Outputs: image


#신규노드 
IRL_GridGuidancePerspectiveWarp
- node_id: IRL_GridGuidancePerspectiveWarp

- display_name: 그리드가이던스 퍼스펙티브 왜곡

- category: 이미지 리파이너/변형

- 역할: 9분할 그리드 기반 퍼스펙티브 왜곡, 프리뷰 가이드라인 표시

- Inputs: image, dst_points(비율), padding_color, method, preview_mode, guide_color

- Outputs: image

#신규노드 
IRL_DragGridGuidancePerspectiveWarp
- node_id: IRL_DragGridGuidancePerspectiveWarp

- display_name: 그리드가이던스 퍼스펙티브 왜곡(드래그모드)

- category: 이미지 리파이너/변형

- 역할: 드래그 모드로 제어점을 직접 움직여 퍼스펙티브 왜곡

- Inputs: image, dst_points(비율), padding_color, method, preview_mode, guide_color

- Outputs: image

#신규노드 
IRL_GridSplineWarp
- node_id: IRL_GridSplineWarp

- display_name: 그리드 스플라인 왜곡

- category: 이미지 리파이너/변형

- 역할: 3x3 그리드 기반 스플라인 왜곡, 동적 엣지 옵션 지원

- Inputs: image, P1~P8, dynamic_edges, padding_color, method, preview_mode, guide_color

- Outputs: image

#신규노드 
IRL_DragGridSplineWarp
- node_id: IRL_DragGridSplineWarp

- display_name: 그리드 스플라인 왜곡(드래그모드)

- category: 이미지 리파이너/변형

- 역할: 드래그 모드로 제어점을 직접 움직여 스플라인 왜곡

- Inputs: image, P1~P8, dynamic_edges, padding_color, method, preview_mode, guide_color

- Outputs: image


[Composite Nodes]

IRL_Imagecomposite

- node_id: IRL_Imagecomposite

- display_name:이미지 합성(통합)

- category: 이미지 리파이너/합성

- 역할: 두 이미지를 선택한 방식에 맞게 처리합니다.

- Inputs: image_a, image_b, factor, priority, saturation

- Inputs(option):Mask

- Inputs:blend_mode : 합성 스타일을 정합니다.

- Inputs(option):Mask_mode 마스크를 스프레이 스타일로 처리할지의 여부를 정합니다.

- Outputs: image

#신규노드
IRL_SequentialLayerComposite
- node_id: IRL_SequentialLayerComposite

- display_name: 순차 레이어 콤포짓 (3슬롯)

- category: 이미지 리파이너/합성

- 역할: 베이스 캔버스 위에 마스크가 지정된 레이어들을 순차적으로 합성

- Inputs: base_canvas, layer_1, mask_1, layer_2, mask_2, strength_1, strength_2, mask_mode

- Outputs: image


#신규노드
IRL_Image2cutcomposite
- node_id: IRL_Image2cutcomposite

- display_name: 이미지 컷 레이아웃(2컷)

- category: 이미지 리파이너/합성

- 역할: 이미지를 선택한 컷 레이아웃에 맞게 배치합니다.

- Inputs: cut_layout, mode, image_a, image_b, pad_color, resize_set

- Outputs: image

#신규노드
IRL_Image3cutcomposite
- node_id: IRL_Image3cutcomposite

- display_name: 이미지 컷 레이아웃(3컷)

- category: 이미지 리파이너/합성

- 역할: 이미지를 선택한 컷 레이아웃에 맞게 배치합니다.

- Inputs: cut_layout, mode, image_a, image_b, image_c, pad_color, resize_set

- Outputs: image

#신규노드
IRL_Image4cutcomposite
- node_id: IRL_Image4cutcomposite

- display_name: 이미지 컷 레이아웃(4컷)

- category: 이미지 리파이너/합성

- 역할: 이미지를 선택한 컷 레이아웃에 맞게 배치합니다.

- Inputs: cut_layout, mode, image_a, image_b, image_c, image_d, pad_color, resize_set

- Outputs: image

#신규노드
IRL_Image5cutcomposite
- node_id: IRL_Image5cutcomposite

- display_name: 이미지 컷 레이아웃(5컷)

- category: 이미지 리파이너/합성

- 역할: 이미지를 선택한 컷 레이아웃에 맞게 배치합니다.

- Inputs: cut_layout, mode, image_a, image_b, image_c, image_d, image_e, pad_color, resize_set

- Outputs: image

#신규노드
IRL_Image6cutcomposite
- node_id: IRL_Image6cutcomposite  

- display_name: 이미지 컷 레이아웃(6컷)  

- category: 이미지 리파이너/합성  

- 역할: 이미지를 선택한 컷 레이아웃에 맞게 배치합니다.  

- Inputs: cut_layout, mode, image_a, image_b, image_c, image_d, image_e, image_f, pad_color, resize_set 
 
- Outputs: image

#신규노드
IRL_ImagecutPreper
- node_id: IRL_ImagecutPreper  

- display_name: 이미지 컷 프레퍼  

- category: 이미지 리파이너/합성  

- 역할: 간이 컷 레이아웃 슬롯을 만듭니다.  

- Inputs: cut_layout, mask1_x, mask1_y, left1_x, top1_y, mask2_x, mask2_y, left2_x, top2_y, mask3_x, 
          mask3_y, left3_x, top3_y, mask4_x, mask4_y, left4_x, top4_y, mask5_x, mask5_y, left5_x, top5_y, 
          mask6_x, mask6_y, left6_x, top6_y, line_size, line_color, canvas_x, canvas_y, preview_mode  

- Outputs: image  

#신규노드

IRL_ImagecutCompositeCustom
- node_id: IRL_ImagecutCompositeCustom  

- display_name: 이미지 컷 컴포짓 커스텀  

- category: 이미지 리파이너/합성  

- 역할: 프레퍼 노드가 만든 컬러 레이아웃을 기반으로 이미지를 각 컷 슬롯에 합성합니다.  

- Inputs: canvas_image, image_a, image_b, image_c, image_d, image_e, image_f  

- Outputs: image


[Noise / Generation Nodes]
#모든 노드 프리뷰 내장, 노이즈팩으로서 동봉 패키지의 리샘플러에 연결해 사용 가능

IRL_AddGaussianNoise
- node_id: IRL_AddGaussianNoise

- display_name: 가우시안 노이즈 추가

- category: 이미지 리파이너/노이즈

- 역할: 시드 통제가 가능한 정밀 가우시안 노이즈를 이미지에 추가

- Inputs: image, sigma, seedset, show_preview, clear_cache

- Outputs: image

IRL_SaltPepperNoise
- node_id: IRL_SaltPepperNoise

- display_name: 소금 & 후추 노이즈

- category: 이미지 리파이너/노이즈

- 역할: 컬러 오염 없는 흑백 Salt & Pepper 노이즈 추가

- Inputs: image, amount, seedset, show_preview, clear_cache

- Outputs: image

IRL_PerlinNoise
- node_id: IRL_PerlinNoise

- display_name: 퍼린 노이즈

- category: 이미지 리파이너/노이즈

- 역할: 프랙탈 퍼린 노이즈 패턴 생성

- Inputs: width, height, scale, octaves, persistence, seedset, show_preview, clear_cache

- Outputs: image

IRL_RandomColor
- node_id: IRL_RandomColor

- display_name: 랜덤 컬러 이미지

- category: 이미지 리파이너/노이즈

- 역할: 시드 통제가 가능한 무작위 컬러 화이트 노이즈 이미지 생성

- Inputs: width, height, seedset, show_preview, clear_cache

- Outputs: image

IRL_WhiteNoise
- node_id: IRL_WhiteNoise

- display_name: 화이트 노이즈

- category: 이미지 리파이너/노이즈

- 역할: 아날로그 질감을 가진 커스텀 화이트 노이즈 패턴 생성

- Inputs: width, height, scale, seedset, show_preview, clear_cache

- Outputs: image

IRL_RGBColor
- node_id: IRL_RGBColor

- display_name: 컬러 이미지

- category: 이미지 리파이너/노이즈

- 역할: 지정된 RGB 색상으로 채워진 단색 배경 캔버스 생성

- Inputs: str_r, str_g, str_b

- Outputs: image

#신규노드
IRL_NoiseCreator
- node_id: IRL_NoiseCreator

- display_name: 노이즈 생성기

- category: 이미지 리파이너/노이즈

- 역할: 시드 통제가 가능한 정밀 노이즈를 준비하고, 리샘플러(샘플러)에 전달할 노이즈 팩과 시드 코드를 생성합니다.

- Inputs: noise_mode, seedset, noise_switch, show_preview, clear_cache

- Outputs: noise_pack, seedset


[Analysis Nodes]

#글자크기조정가능.

#이제 Comfyui 내장 폰트로 그냥 씁니다. 복사도 가능하고, 폰트크기를 조절도 됩니다.

IRL_RGBSplit

- node_id: IRL_RGBSplit

- display_name=이미지 3채널 색상 분리

- category: 이미지 리파이너/분석

- 역할: RGB 레이어 분할 출력

- Inputs: image

- Outputs: histogram


IRL_HistogramPlot

- node_id: IRL_HistogramPlot

- display_name:이미지 히스토그램 그래프

- category: 이미지 리파이너/분석

- 역할: RGB 히스토그램을 계산해 그래프로 출력

- Inputs: image

- Outputs: histogram


IRL_ImageMeanStd
#개선. 위젯타입 자체실행 노드로 변경되었습니다.
- node_id: IRL_ImageMeanStd

- display_name:이미지 평균 & 표준편차

- category: 이미지 리파이너/분석

- 역할: 픽셀 평균값/표준편차 계산

- Inputs: image, font_size

- Outputs: 자체출력 



IRL_ImageMinMax
#개선. 위젯타입 자체실행 노드로 변경되었습니다.
- node_id: IRL_ImageMinMax

- display_name:이미지 최소 & 최대

- category: 이미지 리파이너/분석

- 역할: 최소/최대 픽셀값 계산

- Inputs: image, font_size

- Outputs: 자체출력



IRL_ImageEdgeMap

- node_id: IRL_ImageEdgeMap

- display_name:이미지 에지 맵

- category: 이미지 리파이너/분석

- 역할: Sobel 기반 에지 맵 생성

- Inputs: image, edge_scale

- Outputs: image


#신규노드
IRL_CustomDepthlikeMapGenerator 
- node_id: IRL_CustomDepthlikeMapGenerator

- display_name: 커스텀 유사 뎁스 맵 생성기 
 
- category: 이미지 리파이너/분석  

- 역할: 엣지 맵과 거리 변환을 활용해 입체적인 간이 뎁스 맵과 워터셰드 영역 분리 마스크를 생성합니다.
  
- Inputs: image, threshold, blur_enabled, morph_kernel, edge_fill_coords, invert_target, enable_watershed, 
          edge_fill, min_marker_distance, preview_guide, overwrite_gray, clahe_clip_limit, clahe_tile_size, depth_style
		  
- Outputs: depth_map, region_mask



IRL_ImageBrightnessContrast
#개선. 위젯타입 자체실행 노드로 변경되었습니다.
- node_id: IRL_ImageBrightnessContrast

- display_name:밝기 & 대비

- category: 이미지 리파이너/분석

- 역할: 밝기/대비 통계 계산

- Inputs: image, font_size

- Outputs: 자체출력



IRL_CannyEdgeStats
#개선. 위젯타입 자체실행 노드로 변경되었습니다.
- node_id: IRL_CannyEdgeStats

- display_name:캐니 에지 통계

- category: 이미지 리파이너/분석

- 역할: 캐니 엣지를 기반으로 에지 밀도 및 평균 에지 강도를 계산하고 시각적 통계 이미지를 생성합니다.

- Inputs: image, font_size

- Outputs: 자체출력
	
	
	
IRL_DepthStats
#개선. 위젯타입 자체실행 노드로 변경되었습니다.
- node_id: IRL_DepthStats

- display_name:깊이 통계

- category: 이미지 리파이너/분석

- 역할: 깊이맵의 평균 및 표준편차를 계산하고 시각적 통계 이미지를 생성합니다.

- Inputs: image, font_size
	
- Outputs: 자체출력





