# Metric3D - 2D BEV / 3D Voxel 맵 생성기 (Map Generators) 구현 리뷰 및 가이드

이 문서는 본 프로젝트의 Depth Estimation 파이프라인(`do_test.py`)에 새롭게 추가된 2D BEV (Bird's-Eye View) 맵 및 3D Voxel 맵 생성 기능에 대한 상세한 코드 리뷰와 작동 방식을 설명합니다. 모든 핵심 로직은 `mono/utils/map_generators.py`에 구현되어 있으며, `do_test.py`의 추론 루프 뒷단에서 호출되어 결과를 출력합니다.

---

## 1. 개요 (Overview)

단일 이미지 깊이 추정(Monocular Depth Estimation) 결과로 얻어진 3D Point Cloud(`pcd_np`)를 활용하여 로봇 주행 및 3D 공간 인식을 돕고자 다음 4가지 핵심 지도를 자동 생성합니다:

1. **Geometric Costmap (기하학적 비용 맵, 2D BEV)**
2. **Pure Terrain Map (순수 지형 맵, 2D BEV)**
3. **Drivable Region (주행 가능 영역, 2D BEV)**
4. **Occupancy Voxel Map (점유 보셀 맵, 3D)**

---

## 2. 주요 스크립트 모듈 설명

### 2.1. `mono/utils/map_generators.py` (신규 핵심 모듈)

이 파일은 `(N, 3)` 형태의 3D 포인트 클라우드 배열을 입력으로 받아 공간 데이터를 그리드/복셀로 정리하고 가공하는 수학적 알고리즘을 담고 있습니다.

- **`generate_geometric_costmap(pcd, res=0.05)`**
  - **원리:** 3D 포인트 클라우드를 Top-down 시점인 X-Z 평면으로 투영하여 2D Grid 형태로 샘플링합니다(기본 해상도 셀당 5cm).
  - **비용 계산 로직:**
    - 하나의 셀 격자 내에 들어온 3D 점들의 **Y축(높이) 평균**을 산출하여 지면의 기본 높이로 삼습니다.
    - 해당 셀 내 점들의 **높이 분산(Variance)**을 계산하여 지면의 요철이나 거칠기를 판단합니다.
    - 주변 셀의 평균 높이와의 차이를 살펴 **기울기(Slope)**를 계산합니다.
    - 거칠기와 기울기가 시스템이 설정한 임계값(Threshold)을 넘는 구간은 주행이 힘들거나 치명적인 장애물(벽, 급경사)로 판단하여 로봇이 기피하도록 Cost를 최댓값(255)으로 설정합니다.

- **`generate_pure_terrain_map(pcd, res=0.05, distance_threshold=0.15)`**
  - **원리:** 주변의 장애물, 사람, 구조물 등을 걷어내고 순수한 바닥(지형)의 높낮이만을 평활화하여 보여주는 Height Map입니다.
  - **알고리즘:** 무작위 샘플링을 통한 평면 적합 모델 알고리즘인 **RANSAC**(Random Sample Consensus)을 사용합니다.
  - 주어진 원본 데이터에서 RANSAC으로 가장 유력한 메인 바닥 평면(Ground plane)을 추정해 냅니다.
  - 이 평면 방정식과 각 점 간의 수직 거리를 구해 `distance_threshold` (15cm) 이상 떨어져 솟아있는 점들은 '장애물'로 간주하여 필터링(제거)합니다. 
  - 남은 순수한 바닥의 점들만 2D Grid로 다시 올려 지형 지도로 생성합니다.

- **`generate_drivable_region(costmap, terrain_mask, map_info_2d, seed_x=0.0, seed_z=0.0)`**
  - **원리:** 카메라(또는 로봇)의 현재 발밑 좌표계를 시작점으로 하여 물리적으로 안전하고, 연속적으로 뻗어나갈 수 있는 통행 가능 영역을 추출합니다.
  - **알고리즘:** **Flood Fill (플러드 필)** 알고리즘을 사용합니다.
    - 시작점(Seed, 기본 `X=0, Z=0`)에서 출발하여 상하좌우 인접 셀들을 탐색합니다.
    - Costmap 상에서 장애물이 아니고(Cost가 낮고), Terrain 상에서 데이터가 끊기지 않은 곳들로 지속적으로 '색칠하며' 나아갑니다.
    - 영역이 뻗어나가다 장애물을 마주치면 해당 방향의 확장을 정지합니다. 최종 결과 뭉치가 주행 가능 구역(`Drivable Mask`)이 됩니다.

- **`generate_occupancy_voxel_map(pcd, voxel_size=0.1)`**
  - **원리:** 3D 세계를 일정한 크기(`10cm`)의 블록(Voxel) 형태의 정방형 격자 구조로 단순 구조화합니다.
  - 포인트 클라우드의 `X, Y, Z` 실수 좌표들을 해상도로 나누어 정수 인덱스(`idx_x, idx_y, idx_z`)로 변환합니다. 점 데이터가 존재하는 3D 공간 인덱스를 1(Occupied)로 표시하여 빈공간과 찬 공간을 3D 배열 형태로 분리합니다.

- **`save_voxel_as_ply` & `save_pcd_and_voxel_as_ply`**
  - Voxel Map 내부 데이터를 실제로 시각화 프로그램에서 보기 좋게 3D 모델(`.ply` 형식)로 그려내는 유틸리티입니다.
  - 시인성을 높이기 위해 하나의 복셀을 단순히 센터에 점 한 개만 찍지 않고, 복셀 크기에 맞춘 **8개의 모서리 꼭짓점 모델**로 그려 넣어 커다란 블록 형태로 랜더링시킵니다.
  - 또한 포인트 클라우드의 `Y축(높이)` 등에 기반해 파란색(낮음)~빨간색(높음)으로 이어지는 색상 그라디언트를 입혀 직관성을 더합니다.

### 2.2. `mono/utils/do_test.py` (파이프라인 통합 파일)

이 파일은 이미지 한 장 단위로 수행되는 `postprocess_per_image` 함수 내에 추가 로직이 반영되었습니다. 원본 모델의 Depth 생성 직후 위 `map_generators.py`의 함수연쇄를 실행합니다.

- **2D 맵 이미지 가공 (Coloring & Scaling)**
  - 출력된 배열 결과물들에 직접 OpenCV(cv2)의 컬러맵을 적용합니다. 
    - Costmap은 `COLORMAP_JET` (Low=파랑, High=빨강)
    - Terrain Map도 `COLORMAP_JET` (낮은 지대=파랑, 높은 지대=빨강)으로 통일되어 있습니다.
  - 기존 `5cm / pixel` 해상도로 20m 공간을 담으면 400x400 크기의 작은 썸네일 이미지가 나오기 때문에 시인성이 떨어졌습니다. 이를 `cv2.resize`로 **4배(x4)** 스케일 업(Scale-up)하여 훨씬 크고 선명한 화질(1600x1600 가량)로 저장합니다.
  - **추가 오버레이(Grid & Marker):**
    - `draw_1m_grid`라는 내부 헬퍼 함수를 통해 배경에 **하얀색의 1m x 1m 격자(모눈) 선**을 그려 넣었습니다.
    - 카메라의 시작 원점은 항상 `X=0, Z=0` (화면의 하단 정중앙 파트)이므로 이 지점에 **강조를 위한 붉은색 원형 점(Red Dot)**을 그려주어 방향감을 잃지 않도록 보강했습니다.

- **최종 출력 관리**
  - `pcd` 폴더 내 프레임별 폴더에 `.png` 형식으로 3장의 2D BEV 맵(`_costmap.png`, `_terrain.png`, `_drivable.png`)을 저장합니다.
  - Voxel 정보 원본은 프로그래밍용 배열 형식인 `_voxel.npy`로 기록되며, 동시에 뷰어를 위해 `.ply` 시각화 파일(`_voxel.ply`, `_combined_voxel.ply`)이 저장됩니다.

---

## 3. 맵 결과물 요약본 (Outputs Summary)
실행 시 아래와 같은 종류의 파일이 생성됩니다.

1. `..._costmap.png`: 위험지역과 평지를 Jet 색상 톤으로 한눈에 파악. (Grayscale 기능은 요청에 의해 롤백됨)
2. `..._terrain.png`: 장애물이 부딫힐 위험이없는 순수 등고선(높낮이) 파악용.
3. `..._drivable.png`: 녹색 마스킹 덩어리로 Flood-fill 시뮬레이션으로 찾아낸 장애물이 없는 연속 주행 구역.
4. `..._voxel.npy`: 파이썬/C++용 점유된 voxel boolean 배열.
5. `..._voxel.ply`: 높이 색상이 적용된 Occupancy Voxel의 3D 표면 파일.
6. `..._combined_voxel.ply`: 원본 사진 색상 Point Cloud + 빨강/파랑 Voxel을 겹쳐서 보는 디버그용 3D 파일.
