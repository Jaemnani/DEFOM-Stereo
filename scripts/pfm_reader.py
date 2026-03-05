import numpy as np
import re
import cv2

def read_pfm(file_path):
    """
    PFM (Portable FloatMap) 파일을 읽어 NumPy 배열과 스케일 팩터를 반환합니다.
    """
    with open(file_path, "rb") as f:
        # 1. 헤더(Header) 읽기
        
        # PFM 형식 식별자: "PF" (컬러) 또는 "Pf" (흑백)
        color = f.readline().decode('utf-8').rstrip()
        if color == 'PF':
            channels = 3
        elif color == 'Pf':
            channels = 1
        else:
            raise Exception('Not a PFM file.')

        # 2. 이미지 크기 읽기: 너비(W) 높이(H)
        dimensions_line = f.readline().decode('utf-8').rstrip()
        dimensions = re.split(r'\s+', dimensions_line)
        width = int(dimensions[0])
        height = int(dimensions[1])

        # 3. 스케일 팩터(Scale Factor) 및 엔디안(Endianness) 읽기
        # 양수 스케일: 빅 엔디안, 음수 스케일: 리틀 엔디안
        scale = float(f.readline().decode('utf-8').rstrip())
        endianness = '<' if scale < 0 else '>' # '<'는 리틀, '>'는 빅 엔디안

        # 스케일 값의 부호만 확인했으므로, 실제 스케일 값은 절대값으로 사용
        scale = abs(scale)

        # 4. 데이터(Data) 읽기: 부동 소수점 (Float)
        
        # 데이터는 (W * H * Channels) 크기의 32비트 부동 소수점(float32) 배열
        data = np.fromfile(f, endianness + 'f')
        
        # 데이터 모양을 (H, W, Channels) 또는 (H, W)로 변환
        shape = (height, width, channels) if channels == 3 else (height, width)
        image = np.reshape(data, shape)
        
        # PFM은 보통 아래에서 위로(bottom-up) 저장되므로, 상하 반전 (Vertical Flip) 필요
        image = np.flipud(image)
        
        return image, scale

# 예시 사용법:
# path = "/storage/jeremy/sceneflow/FlyingThings3D/disparity/TRAIN/A/0000/left/0006.pfm"
# path = "/storage/jeremy/sceneflow/Monkaa/disparity/a_rain_of_stones_x2/left/0000.pfm"
path = "/storage/train_datasets/depth_estimation/sceneflow/Driving/disparity/15mm_focallength/scene_backwards/fast/left/0001.pfm"

pfm_image, scale_factor = read_pfm(path)

print(f"이미지 형태: {pfm_image.shape}")
print(f"스케일 팩터: {scale_factor}")

# 5. OpenCV 형식으로 변환 (선택 사항)
# PFM 이미지는 보통 깊이(depth) 맵이므로, 필요에 따라 정규화하거나 시각화해야 함.
# 예를 들어, 흑백(단일 채널) PFM을 8비트 이미지로 변환하여 저장하거나 표시:
show_image = False
if show_image:
    if pfm_image.ndim == 2:
        # 0과 255 사이로 정규화
        norm_image = (pfm_image - pfm_image.min()) / (pfm_image.max() - pfm_image.min()) * 255
        norm_image = norm_image.astype(np.uint8)
        
        # cv2.imshow('PFM Depth Map', norm_image)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite('test.jpg', norm_image)
        cv2.imwrite('output_depth.png', norm_image)

# 6. OpenCV로 처리할 준비가 된 NumPy 배열 반환
cv2_compatible_image = pfm_image

print("")