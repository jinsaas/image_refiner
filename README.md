
#### Image Refiner Lite — ComfyUI용 독립 이미지 보정·마스크·텍스트 유틸리티 노드팩####
Image Refiner Lite는 기존 WAS Node Suite의 구조나 코드에 의존하지 않는,

완전히 새롭게 작성된 독립 이미지 처리 플러그인입니다.

Stable Diffusion 모델, latent기반 기능을 전부 제거하고 ComfyUI Standalone 환경에서

100% 안정적으로 동작하는 순수 이미지·마스크·텍스트 처리 노드만 제공합니다.



####Image Refiner Lite는 다음 원칙을 기반으로 설계되었습니다####

• 환경 안정성 최우선

• latent/WebUI 잔재 완전 제거

• 순수 이미지 처리 기반

• 결정성 100% (Portable 환경에서도 동일 결과)

• 유지보수 가능한 구조

• 불필요한 의존성 제거



이 플러그인은 WAS의 후속작이 아니며, WAS의 코드나 구조를 재사용하지 않습니다.



####WAS 대비 제거된 노드 (환경 안정성 및 의존성 문제로 인해 비활성화됨)####



다음 노드들은 외부 모델, 대규모 패키지, 또는 시스템 환경 변경을 요구하기 때문에

Image Refiner Lite에서는 제공하지 않습니다.

• Rembg Remove Background (모델 의존)

• Segment Anything (SAM) (모델 의존)

• CLIP Interrogator

• Face Restore (모델 의존)

• Depth Map

• Image Captioning (모델 의존)

• Image Classification (모델 의존)

• Semantic Segmentation (모델 의존)

• Video Extract / Assemble (모델 의존)

• Audio Load / Waveform (모델 의존)

• Image Load / Save (ComfyUI 기본 노드와 기능 중복)

• Image to Latent Mask — ComfyUI의 latent 구조와 충돌할 가능성이 있어 제외했습니다.

• Image to Noise — ComfyUI 샘플러와 다른 방식으로 noise를 처리하여 과부하가 발생할 수 있어 제외했습니다.

• LatentCompositeMasked — ComfyUI의 latent 합성 방식과 달라 샘플러 동작에 영향을 줄 수 있어 제외했습니다.

• WAS 마스크 노드 전체 (ComfyUI 기본 노드 기능)

• WAS Text Progress 계열 노드 전체 (ComfyUI 기본 텍스트 노드와 기능 중복)

• 수학 및 논리 관련 노드는 대부분 ComfyUI 기본 기능과 중복되기 때문에, Image Refiner Lite에서는 제공하지 않습니다.

• 기타 git 기반 모델 로더 노드들

이 기능이 필요하다면 WAS Suite를 설치해 사용하시면 됩니다.



###제거된 의존성 패키지 + 제거 사유###

다음 패키지들은 설치 시 torch / torchvision / numpy / pillow
같은 핵심 패키지를 강제로 업데이트하거나,
ComfyUI Standalone 환경을 손상시킬 위험이 있어 제외했습니다.
rembg  : numpy, pillow, onnxruntime 및 146건의 대규모 업데이트를 진행하며 ComfyUI Standalone 환경을 박살내는 가장 큰 원인입니다.
timm     : torchvision 버전 충돌 및 torch 재설치 위험(설치시 torchvision및 torch의 재설치를 유도하며, 개별로 설치하더라도 처리 중 충돌을 유발할 가능성이 있습니다.)
fairscale : torch ABI 충돌을 유발할 가능성이 있습니다.
numba  : 포터블에서는 Numpy 환경을 강제로 바꿉니다.
cmake   : 관련 빌드 설정 과정에서 파이선 환경을 아작낼 위험이 있습니다.
ffmpy / ffmpeg    : 외부 바이너리 의존성으로 충돌 가능성
git+패키지 3~4종 : 대량설치를 유발하기에 임베드 파이선의 버전 데이터 고정이 힘들어지는 원인 중 하나입니다.

*라이트판은 환경 안정성 최우선을 목표로 하므로 위 패키지들은 모두 제거되었습니다.



###requirements.txt가 비어 있는 이유###
ComfyUI는 확장팩 로딩 시 requirements.txt를 자동 설치합니다.
그러나 pip나 git을 통한 원격설치 방식은 환경 손상 위험이 매우 높기 때문에,
Image Refiner Lite판에서는 requirements.txt를 빈 파일로 유지합니다.

대신, 다음의 코드로 최소 의존성 설치법을 남깁니다.
pip install --no-deps simpleeval
pip install --no-deps gitpython
pip install --no-deps pilgram
pip install --no-deps matplotlib

위 패키지들은 라이트판에서 사용하는 안전한 최소 의존성입니다.
설치는 선택 사항이며, 대부분의 기능은 기본 상태에서도 작동합니다.



#[설치 방법]#



Installation:

1. ZIP 다운로드: 저장소를 .zip 파일로 다운로드합니다.

2. custom_nodes 폴더에 배치: 압축을 풀고 ComfyUI/custom_nodes 폴더에 넣습니다.

3. ComfyUI 재시작: 재시작하면 노드가 로드됩니다.


#[Node Usage Manual]#

#변경사항
최신 버전에선 기동은 해도 의미가 별로 없어서, Utility 노드는 폐기했습니다. 

[Adjustments Nodes]

IRL_RGBLevels
#변경사항: 슬라이더 통합
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
#변경사항: 슬라이더 통합

- node_id: IRL_LevelsAdjustment

- display_name:레벨 조정

- category: IRL_Adjustments

- 역할: 입력/출력 레벨 및 감마 조정

- Inputs: image, in_brightness, gamma, out_brightness

- Outputs: image



IRL_GradientMap
#변경사항: 픽셀 코드 직접입력식을 교체.
#컬러 코드 직접입력을 고정, 슬라이더로 강도 조절하도록 변경

- node_id: IRL_GradientMap

- display_name:그라디언트 맵

- category: IRL_Adjustments

- 역할: 명도값을 두 색상의 그라디언트로 매핑

- Inputs: image, color_dark, color_light

- Outputs: image



IRL_ShadowsHighlights

- node_id: IRL_ShadowsHighlights

- display_name:그림자 & 하이라이트

- category: IRL_Adjustments

- 역할: 그림자/하이라이트 디테일 복원

- Inputs: image, shadow_amount, highlight_amount

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
#설명창 오류 수정
- node_id: IRL_WhiteNoise

- display_name:화이트 노이즈

- category: IRL_Noise

- 역할: 화이트 노이즈 이미지 생성

- Inputs: width, height, scale

- Outputs: image


[Analysis Nodes]

#바이큐빅으로 보간하던걸 폰트로딩으로 변경, 글자크기조정가능.

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



