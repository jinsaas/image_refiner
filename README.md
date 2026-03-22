
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

[Adjustments Nodes]

IRL_RGBLevels

- node_id: IRL_RGBLevels

- display_name:RGB 레벨

- category: IRL_Adjustments

- 역할: RGB 채널별 레벨 조정

- Inputs: image, R Levels, G Levels, B Levels

- Outputs: image



IRL_BlackWhiteLevels

- node_id: IRL_BlackWhiteLevels

- display_name:블랙 & 화이트 레벨

- category: IRL_Adjustments

- 역할: 전체 블랙/화이트 포인트 조정

- Inputs: image, black_point, white_point

- Outputs: image



IRL_LevelsAdjustment 


- node_id: IRL_LevelsAdjustment

- display_name:레벨 조정

- category: IRL_Adjustments

- 역할: 입력/출력 레벨 및 감마 조정

- Inputs: image, in_brightness, gamma, out_brightness

- Outputs: image



IRL_GradientMap

#컬러 코드 직접입력을 고정, 슬라이더로 강도 조절

- node_id: IRL_GradientMap

- display_name:그라디언트 맵

- category: IRL_Adjustments

- 역할: 제시된 색상팔레트들의 조합 및 기존이미지와의 오버레이 처리로 다양한 사용법을 추구합니다.

- Inputs: image, 7color palette, color_str, base_suf, blend_mode, gradient옵션 제공

- Outputs: image



IRL_ShadowsHighlights

- node_id: IRL_ShadowsHighlights

- display_name:그림자 & 하이라이트

- category: IRL_Adjustments

- 역할: 그림자/하이라이트 디테일 복원

- Inputs: image, shadow_amount, highlight_amount

- Outputs: image


#패치 노드#
디폴트 값 안정화
IRL_ImgResampler

- node_id: IRL_ImgResampler

- display_name:이미지 리샘플러

- category: IRL_Adjustments

- 역할: 이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다. 이미지 인코드, 디코드도 전부 처리합니다.

- Inputs: model, clip, vae, image, noise option, sampler 옵션

- Outputs: image

#신규 노드#

디폴트 값 안정화
IRL_ImgResampler

- node_id: IRL_ImgResampler

- display_name:이미지 리샘플러

- category: IRL_Adjustments

- 역할: 이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다. 이미지 인코드, 디코드도 전부 처리합니다.

- Inputs: model, clip, vae, image, noise option, sampler 옵션

- Outputs: image

#신규 노드#
기능 분할
IRL_ImgResamplerMix

- node_id: IRL_ImgResamplerMix

- display_name:이미지 리샘플러(믹스)

- category: IRL_Adjustments

- 역할: 이미지에 노이즈를 추가하고 디노이즈 재처리를 통해 품질 향상을 시도합니다. 이미지 인코드, 디코드도 전부 처리합니다.

- Inputs: model, clip, vae, image, noise option, sampler 옵션

- Outputs: image

#신규 노드#
로라 로드 처리 가능
IRL_ImgResamplerAnd

- node_id: IRL_ImgResamplerAnd

- display_name:이미지 리샘플러(로라로딩)

- category: IRL_Adjustments

- 역할: 로라를 읽고 이미지에 노이즈를 추가한 뒤 디노이즈 재처리를 통해 품질 향상을 시도합니다. 이미지 인코드, 디코드도 전부 처리합니다.

- Inputs: model, clip, vae, image, noise option, sampler 옵션

- Outputs: image

#노드 기능 추가#
IRL_ImgDetailer

- node_id: IRL_ImgDetailer

- display_name: 이미지 디테일러

- category: IRL_Adjustments

- 역할: 이미지 재처리를 통해 품질 향상을 시도합니다.

- Inputs: image, 샤프닝, 히스토그램 평활화, 유연화, 라인강조 슬라이더, 라인 색 헥스입력슬롯 추가

- Outputs: image

#신규 노드#

IRL_NoiseCleaner

- node_id: IRL_NoiseCleaner

- display_name: 노이즈 제거기

- category: IRL_Adjustments

- 역할: 이미지 재처리를 통해 품질 향상을 시도합니다.

- Inputs: image, 마스크, 노이즈세팅, 색상강조 슬라이더, 라인강조 슬라이더 

- Outputs: image

[Filter Nodes]

IRL_GaussianBlur
- node_id: IRL_GaussianBlur
- display_name: 가우시안 블러
- category: IRL_Filter
- 역할: 커널 크기와 시그마 값을 사용하여 이미지에 가우시안 블러 적용
- Inputs: image, kernel_size, sigma
- Outputs: image

IRL_MedianBlur
- node_id: IRL_MedianBlur
- display_name: 미디언 블러
- category: IRL_Filter
- 역할: 커널 크기를 사용하여 이미지에 미디언 블러 적용
- Inputs: image, kernel_size
- Outputs: image

IRL_BilateralFilter
- node_id: IRL_BilateralFilter
- display_name: 양방향 필터
- category: IRL_Filter
- 역할: 지름과 시그마 값을 사용하여 이미지에 양방향 필터 적용
- Inputs: image, diameter, sigma_color, sigma_space
- Outputs: image

IRL_Sharpen
- node_id: IRL_Sharpen
- display_name: 샤픈
- category: IRL_Filter
- 역할: 언샤프 마스크 방식으로 이미지를 선명하게 조정
- Inputs: image, amount
- Outputs: image

IRL_HighPass
- node_id: IRL_HighPass
- display_name: 하이패스 필터
- category: IRL_Filter
- 역할: 에지와 세부 사항을 강조하기 위해 하이패스 필터 적용
- Inputs: image, radius
- Outputs: image



[Transform Nodes]



IRL_Resize

- node_id: IRL_Resize

- display_name:리사이즈

- category: IRL_Transform

- 역할: 이미지 리사이즈

- Inputs: image, width, height

- Outputs: image



IRL_Rotate

- node_id: IRL_Rotate

- display_name:회전

- category: IRL_Transform

- 역할: 이미지 회전

- Inputs: image, angle

- Outputs: image



IRL_Flip
#변경사항: int선택이던걸 Combo 스타일로 변경

- node_id: IRL_Flip

- display_name:플립

- category: IRL_Transform

- 역할: 이미지 뒤집기 (가로/세로)

- Inputs: image
- Inputs: mode[horizontal,vertical] 수평/수직 중 선택하여 뒤집기
- Outputs: image



IRL_Crop

- node_id: IRL_Crop

- display_name:크롭

- category: IRL_Transform

- 역할: 지정 영역 크롭

- Inputs: image, x, y, width, height

- Outputs: image



IRL_CropMerge

- node_id: IRL_CropMerge

- display_name:크롭 마진

- category: IRL_Transform

- 역할: 영역 크롭(직접선택)

- Inputs: image, L, R, T, B

- Outputs: image



IRL_PerspectiveWarp

- node_id: IRL_PerspectiveWarp

- display_name:퍼스펙티브 왜곡

- category: IRL_Transform

- 역할: 4점 기반 투시 왜곡

- Inputs: image, src_points, dst_points

- Outputs: image



[Composite Nodes]
#마스크를 연결하지 않으면 이미지 위 혹은 밑에 겹침이미지가 나오는 식으로 처리.

#마스크 연결기능 추가 및 마스크 적용처리방식 추가했습니다.

#마스크는 옵션연결이라 직접 연결안하면 관련옵션은 작동되지 않습니다.

IRL_ImageBlend

- node_id: IRL_ImageBlend

- display_name:이미지 블렌드

- category: IRL_Composite

- 역할: 두 이미지 선형 블렌딩

- Inputs: image_a, image_b, factor

- Inputs(option):Mask

- Inputs(option):Mask_mode 마스크를 스프레이 스타일로 처리할지의 여부를 정합니다.

- Outputs: image



IRL_ImageOverlay

- node_id: IRL_ImageOverlay

- display_name:이미지 오버레이

- category: IRL_Composite

- 역할: 오버레이 블렌딩

- Inputs: image_a, image_b

- Inputs(option):Mask

- Inputs(option):Mask_mode 마스크를 스프레이 스타일로 처리할지의 여부를 정합니다.

- Outputs: image



IRL_ImageAdd

- node_id: IRL_ImageAdd

- display_name:이미지 더하기

- category: IRL_Composite

- 역할: 두 이미지 더하기

- Inputs: image_a, image_b

- Inputs(option):Mask

- Inputs(option):Mask_mode 마스크를 스프레이 스타일로 처리할지의 여부를 정합니다.

- Outputs: image



IRL_ImageMultiply

- node_id: IRL_ImageMultiply

- display_name:이미지 곱하기

- category: IRL_Composite

- 역할: 두 이미지 곱하기

- Inputs: image_a, image_b

- Inputs(option):Mask

- Inputs(option):Mask_mode 마스크를 스프레이 스타일로 처리할지의 여부를 정합니다.

- Outputs: image



IRL_ImageDifference

- node_id: IRL_ImageDifference

- display_name:이미지 차이

- category: IRL_Composite

- 역할: 두 이미지의 절대 차이 계산

- Inputs: image_a, image_b

- Inputs(option):Mask

- Inputs(option):Mask_mode 마스크를 스프레이 스타일로 처리할지의 여부를 정합니다.

- Outputs: image



[Noise / Generation Nodes]



IRL_AddGaussianNoise

- node_id: IRL_AddGaussianNoise

- display_name:가우시안 노이즈 추가

- category: IRL_Noise

- 역할: 가우시안 노이즈 추가

- Inputs: image, sigma

- Outputs: image



IRL_SaltPepperNoise

- node_id: IRL_SaltPepperNoise

- display_name:소금 & 후추 노이즈

- category: IRL_Noise

- 역할: Salt \& Pepper 노이즈 추가

- Inputs: image, amount

- Outputs: image



IRL_PerlinNoise

- node_id: IRL_PerlinNoise

- display_name:퍼린 노이즈

- category: IRL_Noise

- 역할: 퍼린 노이즈 이미지 생성

- Inputs: width, height, scale

- Outputs: image



IRL_RandomColor

- node_id: IRL_RandomColor

- display_name:랜덤 컬러 이미지

- category: IRL_Noise

- 역할: 무작위 단색 이미지 생성

- Inputs: width, height

- Outputs: image


IRL_WhiteNoise

- node_id: IRL_WhiteNoise

- display_name:화이트 노이즈

- category: IRL_Noise

- 역할: 화이트 노이즈 이미지 생성

- Inputs: width, height, scale

- Outputs: image


[Analysis Nodes]

#글자크기조정가능.

#윈도우 기본 폰트지만, 만약 폰트가 없을 경우 노드에 포함되어있는 폰트 설치.

#엣지 맵 추출은 강도 세팅 추가.

IRL_RGBSplit

- node_id: IRL_RGBSplit

- display_name=이미지 3채널 색상 분리

- category: IRL_Analysis

- 역할: RGB 레이어 분할 출력

- Inputs: image

- Outputs: histogram


IRL_ImageHistogram Graph

- node_id: IRL_ImageHistogramGraph

- display_name:이미지 히스토그램 그래프

- category: IRL_Analysis

- 역할: RGB 히스토그램을 계산해 그래프로 출력

- Inputs: image

- Outputs: histogram


IRL_ImageMeanStd

- node_id: IRL_ImageMeanStd

- display_name:이미지 평균 & 표준편차

- category: IRL_Analysis

- 역할: 픽셀 평균값/표준편차 계산

- Inputs: image, font_size

- Outputs: stats — 평균 & 표준편차 픽셀값 계산 + 텍스트 기반 시각화 이미지 



IRL_ImageMinMax

- node_id: IRL_ImageMinMax

- display_name:이미지 최소 & 최대

- category: IRL_Analysis

- 역할: 최소/최대 픽셀값 계산

- Inputs: image, font_size

- Outputs: minmax — 최소/최대 픽셀값 계산 + 텍스트 기반 시각화 이미지



IRL_ImageEdgeMap

- node_id: IRL_ImageEdgeMap

- display_name:이미지 에지 맵

- category: IRL_Analysis

- 역할: Sobel 기반 에지 맵 생성

- Inputs: image, edge_scale

- Outputs: image



IRL_ImageBrightnessContrast

- node_id: IRL_ImageBrightnessContrast

- display_name:밝기 & 대비

- category: IRL_Analysis

- 역할: 밝기/대비 통계 계산

- Inputs: image, font_size

- Outputs: stats — 밝기 & 대비 확인 + 텍스트 기반 시각화 이미지



IRL_CannyEdgeStats

- node_id: IRL_CannyEdgeStats

- display_name:캐니 에지 통계

- category: IRL_Analysis

- 역할: 캐니 엣지를 기반으로 에지 밀도 및 평균 에지 강도를 계산하고 시각적 통계 이미지를 생성합니다.

- Inputs: image, font_size

- Outputs: stats — 에지 밀도 및 평균 에지 강도 + 텍스트 기반 시각화 이미지
	
	
	
IRL_DepthStats

- node_id: IRL_DepthStats

- display_name:깊이 통계

- category: IRL_Analysis

- 역할: 깊이맵의 평균 및 표준편차를 계산하고 시각적 통계 이미지를 생성합니다.

- Inputs: image, font_size
	
- Outputs: stats — 깊이 평균 및 깊이 표준편차 + 텍스트 기반 시각화 이미지





