import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
from tqdm import tqdm
from pathlib import Path
from core.defom_stereo import DEFOMStereo
from utils.utils import InputPadder
from PIL import Image
from matplotlib import pyplot as plt
from time import time
import cv2
import os
from natsort import natsorted

DEVICE = 'cuda'

# ---------------------------------------------------------
# [Mouse Callback Function]
# --visualize 옵션 사용 시 작동하여 픽셀의 Depth 값을 출력
# ---------------------------------------------------------
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        depth_map = param
        # 이미지 범위 내 클릭인지 확인
        if 0 <= x < depth_map.shape[1] and 0 <= y < depth_map.shape[0]:
            depth_value = depth_map[y, x]
            print(f"Depth at ({x}, {y}): {depth_value:.3f} meters")

def load_image(imfile):
    """
    이미지를 로드하고 타겟 사이즈로 리사이징합니다.
    - 입력: 이미지 경로
    - 출력: 텐서 형태의 이미지, 원본 해상도 (h, w)
    """
    img = cv2.imread(imfile)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if img is None:
        raise FileNotFoundError(f"Image file {imfile} not found.")
    
    h_ori, w_ori = img.shape[:2]
    h_in, w_in = h_ori, w_ori

    # 함수 객체에 target_size 속성이 설정되어 있다면 해당 크기로 리사이징
    try:
        if hasattr(load_image, 'target_size') and load_image.target_size is not None:
            h_in, w_in = load_image.target_size
    except AttributeError:
        pass

    if (h_ori, w_ori) != (h_in, w_in):
        # 다운샘플링 시 AREA, 업샘플링 시 LINEAR 보간법 사용
        if h_ori * w_ori < h_in * w_in:
            img_resized = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_LINEAR)
        else:
            img_resized = cv2.resize(img, (w_in, h_in), interpolation=cv2.INTER_AREA)
    else:
        img_resized = img

    img = torch.from_numpy(img_resized).permute(2, 0, 1).float()
    return img[None].to(DEVICE), (h_ori, w_ori)

def preprocess(imfile1, imfile2):
    """
    좌/우 이미지를 로드하고 모델 입력에 맞게 패딩(Padding)을 수행합니다.
    """
    image1, (h_ori1, w_ori1) = load_image(imfile1)
    image2, (h_ori2, w_ori2) = load_image(imfile2)

    # 모델 구조상 입력 크기가 32의 배수여야 하므로 패딩 적용
    padder = InputPadder(image1.shape, divis_by=32)
    image1_padded, image2_padded = padder.pad(image1, image2)

    return image1_padded, image2_padded, (h_ori1, w_ori1), (h_ori2, w_ori2), padder

def inference(model, image1, image2, iters=32, scale_iters=1):
    """
    모델 추론을 수행하여 Disparity를 예측합니다.
    """
    with torch.no_grad():
        disp = model(image1, image2, iters=iters, scale_iters=scale_iters, test_mode=True)
    return disp

def scale_to_original_size(disp_unpadded, h_ori, w_ori):
    """
    예측된 Disparity 맵을 원본 해상도로 복원하고 값 스케일을 조정합니다.
    """
    in_h, in_w = disp_unpadded.shape
    disp_up = cv2.resize(disp_unpadded, (w_ori, h_ori), interpolation=cv2.INTER_LINEAR)
    
    # 해상도가 변하면 Disparity 값(픽셀 거리)도 비율에 맞춰 스케일링해야 함
    scale = float(w_ori) / float(in_w) if in_w > 0 else 1.0
    disp_rescaled = disp_up * scale
    return disp_rescaled

def normalize_disp(disp_rescaled):
    """
    시각화를 위해 Disparity 값을 0~1 사이로 정규화합니다.
    이상치(Outlier) 제거를 위해 하위 0.1%, 상위 99% 값을 기준으로 클리핑합니다.
    """
    vals = disp_rescaled[np.isfinite(disp_rescaled)]
    if vals.size == 0:
        vmin, vmax = 0.0, 1.0
    else:
        vmin, vmax = np.percentile(vals, [0.001, 99.0])
        if vmin == vmax:
            vmax = vmin + 1e-6
    norm = np.clip((disp_rescaled - vmin) / (vmax - vmin), 0.0, 1.0)  
    return norm

def visualize_from_normalized_disp(disp_normalized):
    """
    정규화된 Disparity 맵에 JET 컬러맵을 적용하여 시각화 이미지를 생성합니다.
    """
    img_vis = (255.0 * disp_normalized).astype(np.uint8)
    img_vis_colored = cv2.applyColorMap(img_vis, cv2.COLORMAP_JET)
    return img_vis_colored

def postprocess(disp, h_ori, w_ori, padder):
    """
    추론 결과를 후처리합니다: 패딩 제거 -> 원본 크기 복원 -> 정규화 -> 컬러맵 적용
    """
    # 1. 패딩 제거
    disp_unpadded = padder.unpad(disp)
    disp_result = disp_unpadded.cpu().numpy().squeeze()
    
    # 2. 원본 크기로 리사이징 및 값 스케일링
    disp_rescaled = scale_to_original_size(disp_result, h_ori, w_ori)
    
    # 3. 시각화용 정규화 및 컬러맵 적용
    disp_normalized = normalize_disp(disp_rescaled)
    img_vis_colored = visualize_from_normalized_disp(disp_normalized)
    
    return disp_result, disp_rescaled, img_vis_colored

def scaled_disp_to_depth(disp_rescaled, baseline_m, f_mm, pixel_size_mm):
    """
    Disparity를 실제 거리(Depth, 미터 단위)로 변환합니다.
    공식: Z = (f * b) / d
    """
    f_px = f_mm / pixel_size_mm  # 초점 거리를 픽셀 단위로 변환
    depth_m = f_px * baseline_m / (disp_rescaled + 1e-6)  # 미터 단위 변환, 0으로 나누기 방지
    return depth_m

def depth_to_point_cloud(depth_map, intrinsics):
    """
    Metric Depth map을 3D Point Cloud로 변환 (+60FPS를 위한 NumPy 벡터화 최적화)
    
    :param depth_map: 2D numpy array (H, W), 각 픽셀은 미터(m) 단위의 깊이 값
    :param intrinsics: 카메라 내부 파라미터 딕셔너리 {'fx': float, 'fy': float, 'cx': float, 'cy': float}
    :return: 3D Point Cloud array, shape (N, 3)
    """
    H, W = depth_map.shape
    
    # 1. 픽셀 좌표계 생성 (u: x축, v: y축)
    # 60FPS를 위해 미리 생성된 meshgrid를 캐싱하는 것도 좋은 최적화 방법입니다.
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    
    # 2. 유효한 깊이 값 필터링 (0 또는 무한대 값 제외)
    # 깊이가 존재하는 부분만 계산하여 연산량 감소
    valid_depth_mask = (depth_map > 0.0) & (depth_map < float('inf'))
    
    z = depth_map[valid_depth_mask]
    u_valid = u[valid_depth_mask]
    v_valid = v[valid_depth_mask]
    
    # 3. 카메라 내부 파라미터를 이용한 Reprojection
    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']
    
    x = (u_valid - cx) * z / fx
    y = (v_valid - cy) * z / fy
    
    # 4. Point Cloud 결합 (N, 3) 형태
    point_cloud = np.stack((x, y, z), axis=-1)
    
    return point_cloud

def save_point_cloud_ply(point_cloud, filename, colors=None):
    """
    Save 3D point cloud to PLY format.
    :param point_cloud: (N, 3) numpy array
    :param filename: output ply path
    :param colors: (N, 3) numpy array (uint8), optional
    """
    with open(filename, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {point_cloud.shape[0]}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        if colors is not None:
            f.write("property uchar red\n")
            f.write("property uchar green\n")
            f.write("property uchar blue\n")
        f.write("end_header\n")
        
        if colors is not None:
            pts_colors = np.hstack((point_cloud, colors))
            np.savetxt(f, pts_colors, fmt='%.5f %.5f %.5f %d %d %d')
        else:
            np.savetxt(f, point_cloud, fmt='%.5f %.5f %.5f')

def demo(args):
    # 1. 모델 로드 및 초기화
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location=DEVICE)
    if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()

    # ---------------------------------------------------------
    # [입력 해상도 설정]
    # Argument '--input_size'를 통해 동적으로 설정
    # 입력이 없으면 원본 해상도(None) 사용
    # ---------------------------------------------------------
    # Reference (History):
    # load_image.target_size = (400, 640)
    # load_image.target_size = (320, 512)
    # load_image.target_size = (200, 320)  <- Recent Default
    if args.input_size is not None:
        load_image.target_size = tuple(args.input_size) # (Height, Width)
        target_size_str = f"{args.input_size[1]}x{args.input_size[0]}" # WxH 표기
        print(f" Resizing input to: {load_image.target_size} (H, W)")
    else:
        load_image.target_size = None
        target_size_str = "original_resolution"
        print(f" Using original resolution.")

    # 출력 디렉토리 설정 (해상도별 폴더 구분)
    output_directory = Path(args.output_directory)
    # target_size_str = str(load_image.target_size[::-1][0])+"x"+str(load_image.target_size[::-1][1]) # WxH 표기
    output_directory = os.path.join(output_directory, target_size_str)
    os.makedirs(output_directory, exist_ok=True)

    with torch.no_grad():
        left_images = natsorted(glob.glob(args.left_imgs, recursive=True))
        right_images = natsorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        map_time_accumulators = {
            'costmap': 0.0,
            'terrain': 0.0,
            'drivable': 0.0,
            'voxel': 0.0,
            'total': 0.0,
            'count': 0
        }

        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            # 2. 전처리 (로드, 리사이즈, 패딩)
            image1_padded, image2_padded, (h_ori1, w_ori1), (h_ori2, w_ori2), padder = preprocess(imfile1, imfile2)
            
            # 3. 추론 수행
            disp_pr = inference(model, image1_padded, image2_padded, iters=args.valid_iters, scale_iters=args.scale_iters)
            
            # 4. 후처리 (패딩제거, 원본복원, 시각화)
            disp_pr, disp_rescaled, img_vis_colored = postprocess(disp_pr, h_ori1, w_ori1, padder)

            # 파일명 추출
            file_stem = os.path.splitext(os.path.basename(imfile1))[0]
            
            # 5. 결과 저장
            # (옵션) Numpy 배열 저장
            if args.save_numpy:
                np.save(os.path.join(output_directory, f"{file_stem}.npy"), disp_rescaled)

            # (5-1) Disparity Map 저장 (컬러)
            os.makedirs(os.path.join(output_directory, 'disp_colored'), exist_ok=True)
            out_png = os.path.join(output_directory, 'disp_colored', f'{file_stem}_disp_colored.png')
            # 필요한 경우 주석 해제하여 저장
            # cv2.imwrite(out_png, img_vis_colored, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            # (5-2) Depth Map 변환 및 저장
            # ---------------------------------------------------------
            # [ZED X 카메라 파라미터 적용 (빠른 생성을 위해 disp_pr 320x224 활용)]
            # Baseline: 0.12m
            # Focal Length: 2.2mm
            # Pixel Size: 0.003mm
            # ---------------------------------------------------------
            
            # disp_pr은 원본 복원 전의 320x224 해상도 disparity입니다. 
            # 모델 입력 크기로 줄어든 비율만큼 disparity 값도 스케일다운 되었으므로 보정합니다.
            scale_w = float(disp_pr.shape[1]) / float(w_ori1)
            disp_scaled_fast = disp_pr / scale_w

            depth_map = scaled_disp_to_depth(disp_scaled_fast, baseline_m=0.12, f_mm=2.2, pixel_size_mm=0.003) # ZED X
            
            # depth_m = np.clip(depth_m, 0.0, 4.0) # 시각화를 위해 Depth 클리핑 (0m ~ 4m)
            depth_map = np.clip(depth_map, 0.0, 20.0) # 시각화를 위해 Depth 클리핑 (0m ~ 4m)

            # Depth 정규화 및 시각화
            depth_m = normalize_disp(depth_map)
            img_vis_colored_depth = visualize_from_normalized_disp(depth_m)

            os.makedirs(os.path.join(output_directory, 'depth_colored'), exist_ok=True)
            out_color = os.path.join(output_directory, 'depth_colored', f'{file_stem}_depth_colored.png')
            cv2.imwrite(out_color, img_vis_colored_depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            # (5-3) 3D Point Cloud 생성 및 PLY 파일 저장
            # 카메라 내부 파라미터도 줄어든 스케일(scale_w)에 맞춰 보정 (320x224 기준)
            H_d, W_d = depth_map.shape
            f_px_original = 2.2 / 0.003
            f_px_scaled = f_px_original * scale_w
            
            intrinsics = {
                'fx': f_px_scaled,
                'fy': f_px_scaled,
                'cx': W_d / 2.0,
                'cy': H_d / 2.0
            }
            # Point Cloud 변환 (320x224 해상도 기준, 데이터량 대폭 감소)
            point_cloud = depth_to_point_cloud(depth_map, intrinsics)
            
            # 원본 이미지에서 픽셀 색상 추출 (포인트 클라우드 시각화를 위해 매핑)
            img_color_orig_bgr = cv2.imread(imfile1)
            img_color = cv2.cvtColor(img_color_orig_bgr, cv2.COLOR_BGR2RGB)
            if img_color.shape[:2] != (H_d, W_d):
                img_color = cv2.resize(img_color, (W_d, H_d))
            valid_mask = (depth_map > 0.0) & (depth_map < float('inf'))
            colors = img_color[valid_mask]
            
            # PLY 파일로 저장
            os.makedirs(os.path.join(output_directory, 'pcd_origin'), exist_ok=True)
            out_ply = os.path.join(output_directory, 'pcd_origin', f'{file_stem}_pcd.ply')
            save_point_cloud_ply(point_cloud, out_ply, colors)

            # ---------------------------------------------------------
            # [3D Map Generation & Merge Canvas]
            # ---------------------------------------------------------
            try:
                from map_generators import (
                    generate_geometric_costmap,
                    generate_pure_terrain_map,
                    generate_drivable_region,
                    generate_occupancy_voxel_map,
                    save_voxel_as_ply,
                    save_pcd_and_voxel_as_ply,
                    get_voxel_points_and_colors,
                    get_combined_pcd_and_voxel_points_and_colors
                )
                
                # Generation
                pcd_np = point_cloud
                
                t0_maps = time()
                
                # 1. Costmap
                t_c0 = time()
                costmap, _, cost_mask, map_info_2d = generate_geometric_costmap(pcd_np, res=0.05)
                t_c1 = time()
                
                # 2. Terrain
                t_t0 = time()
                terrain_map, terrain_mask, _ = generate_pure_terrain_map(pcd_np, res=0.05)
                t_t1 = time()
                
                # 3. Drivable
                t_d0 = time()
                drivable_mask = generate_drivable_region(costmap, cost_mask, map_info_2d, seed_x=0.001, seed_z=0.001, cost_thresh=100, method='cca')
                t_d1 = time()
                
                # 4. Voxel Map
                t_v0 = time()
                voxel_map, voxel_info = generate_occupancy_voxel_map(pcd_np, voxel_size=0.1)
                t_v1 = time()
                
                t_maps_end = time()
                
                # Print Profiling Results
                print(f"--- Map Generation Time Profiling ({file_stem}) ---")
                print(f"Costmap:  {t_c1 - t_c0:.4f} s")
                print(f"Terrain:  {t_t1 - t_t0:.4f} s")
                print(f"Drivable: {t_d1 - t_d0:.4f} s")
                print(f"Voxel:    {t_v1 - t_v0:.4f} s")
                print(f"Total Maps: {t_maps_end - t0_maps:.4f} s")
                print("---------------------------------------------------")

                map_time_accumulators['costmap'] += (t_c1 - t_c0)
                map_time_accumulators['terrain'] += (t_t1 - t_t0)
                map_time_accumulators['drivable'] += (t_d1 - t_d0)
                map_time_accumulators['voxel'] += (t_v1 - t_v0)
                map_time_accumulators['total'] += (t_maps_end - t0_maps)
                map_time_accumulators['count'] += 1

                # Colored 2D maps
                costmap_colored = cv2.applyColorMap(costmap, cv2.COLORMAP_JET)
                terrain_colored = cv2.applyColorMap(terrain_map, cv2.COLORMAP_JET)
                drivable_colored = np.zeros_like(costmap_colored)
                drivable_colored[:, :, 1] = drivable_mask

                # Map scaling and grid drawing
                scale_factor = 4
                new_size = (costmap_colored.shape[1] * scale_factor, costmap_colored.shape[0] * scale_factor)
                costmap_colored = cv2.resize(costmap_colored, new_size, interpolation=cv2.INTER_NEAREST)
                terrain_colored = cv2.resize(terrain_colored, new_size, interpolation=cv2.INTER_NEAREST)
                drivable_colored = cv2.resize(drivable_colored, new_size, interpolation=cv2.INTER_NEAREST)
                
                cost_mask_scaled = cv2.resize(cost_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
                terrain_mask_scaled = cv2.resize(terrain_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0
                drivable_mask_scaled = cv2.resize(drivable_mask.astype(np.uint8), new_size, interpolation=cv2.INTER_NEAREST) > 0

                def create_grid_canvas(shape, res, scale, width_m=20.0, depth_m=20.0):
                    img = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
                    ppm = int((1.0 / res) * scale)
                    h, w = shape[:2]
                    grid_alpha = 100
                    color = (grid_alpha, grid_alpha, grid_alpha)
                    cx = w // 2
                    for i in range(0, int(width_m / 2) + 1):
                        x_off = int(i * ppm)
                        if x_off > 0:
                            cv2.line(img, (cx + x_off, 0), (cx + x_off, h), color, 2)
                            cv2.line(img, (cx - x_off, 0), (cx - x_off, h), color, 2)
                    cv2.line(img, (cx, 0), (cx, h), (150, 150, 150), 3)
                    for i in range(0, int(depth_m) + 1):
                        y_pos = int(h - i * ppm)
                        if 0 <= y_pos < h:
                            cv2.line(img, (0, y_pos), (w, y_pos), color, 2)
                    cv2.circle(img, (cx, h), radius=5, color=(0, 0, 255), thickness=-1)
                    return img

                c_grid1 = create_grid_canvas(costmap_colored.shape, res=0.05, scale=scale_factor)
                c_grid2 = create_grid_canvas(terrain_colored.shape, res=0.05, scale=scale_factor)
                c_grid3 = create_grid_canvas(drivable_colored.shape, res=0.05, scale=scale_factor)

                # Overlay map contents ON TOP of grid masks out the grid in mapping region
                c_grid1[cost_mask_scaled] = costmap_colored[cost_mask_scaled]
                c_grid2[terrain_mask_scaled] = terrain_colored[terrain_mask_scaled]
                c_grid3[drivable_mask_scaled] = drivable_colored[drivable_mask_scaled]

                costmap_colored = c_grid1
                terrain_colored = c_grid2
                drivable_colored = c_grid3

                # Save 2D maps and voxel
                os.makedirs(os.path.join(output_directory, 'pcd'), exist_ok=True)
                os.makedirs(os.path.join(output_directory, 'pcd', 'costmap'), exist_ok=True)
                os.makedirs(os.path.join(output_directory, 'pcd', 'terrain'), exist_ok=True)
                os.makedirs(os.path.join(output_directory, 'pcd', 'drivable'), exist_ok=True)
                os.makedirs(os.path.join(output_directory, 'pcd', 'voxel_npy'), exist_ok=True)
                cv2.imwrite(os.path.join(output_directory, 'pcd', 'costmap', f'{file_stem}_costmap.png'), costmap_colored)
                cv2.imwrite(os.path.join(output_directory, 'pcd', 'terrain', f'{file_stem}_terrain.png'), terrain_colored)
                cv2.imwrite(os.path.join(output_directory, 'pcd', 'drivable', f'{file_stem}_drivable.png'), drivable_colored)
                np.save(os.path.join(output_directory, 'pcd', 'voxel_npy', f'{file_stem}_voxel.npy'), voxel_map)

                # Save standalone/combined voxel PLYs
                voxel_dir = os.path.join(output_directory, 'pcd', 'voxel')
                voxel_comb_dir = os.path.join(output_directory, 'pcd', 'voxel_comb')
                os.makedirs(voxel_dir, exist_ok=True)
                os.makedirs(voxel_comb_dir, exist_ok=True)

                save_voxel_as_ply(voxel_map, voxel_info, os.path.join(voxel_dir, f'{file_stem}_voxel.ply'), color_mode='height')
                img_color_flat = img_color.reshape(-1, 3)
                
                # Check point logic: map sizes need to be same for combined saving
                pcd_flat_full = point_cloud.reshape((-1, 3))
                save_pcd_and_voxel_as_ply(pcd_flat_full, img_color_flat, voxel_map, voxel_info, 
                                          os.path.join(voxel_comb_dir, f'{file_stem}_combined_voxel.ply'), color_mode='height')

                # Render PLY projection for canvas
                rgb_img = cv2.cvtColor(img_color, cv2.COLOR_RGB2BGR)
                ply_proj_img = np.zeros_like(rgb_img)
                target_h, target_w = rgb_img.shape[:2]

                theta = np.radians(30)
                R_cam = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
                rvec, _ = cv2.Rodrigues(R_cam)
                tvec = np.array([0, 1.0, 2.0], dtype=np.float32)
                camera_matrix = np.array([[intrinsics['fx'], 0, intrinsics['cx']], 
                                          [0, intrinsics['fy'], intrinsics['cy']], 
                                          [0, 0, 1]], dtype=np.float32)

                # Use original depth filtering for rendering
                valid = point_cloud[:, 2] > 0
                pts_valid = point_cloud[valid]
                colors_valid = img_color_flat[valid]
                pts_2d, _ = cv2.projectPoints(pts_valid.astype(np.float32), rvec.astype(np.float32), tvec, camera_matrix, None)
                if pts_2d is not None:
                    pts_2d = np.round(pts_2d.reshape(-1, 2)).astype(int)
                    z_sort = np.argsort(-pts_valid[:, 2])
                    pts_2d = pts_2d[z_sort]
                    colors_valid = colors_valid[z_sort]

                    mask = (pts_2d[:, 0] >= 0) & (pts_2d[:, 0] < target_w) & (pts_2d[:, 1] >= 0) & (pts_2d[:, 1] < target_h)
                    pts_2d = pts_2d[mask]
                    colors_valid = colors_valid[mask]
                    
                    ply_proj_img[pts_2d[:, 1], pts_2d[:, 0]] = colors_valid[:, ::-1]
                    ply_proj_img = cv2.morphologyEx(ply_proj_img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))

                # Add Voxel Projection
                def make_screenshot(pts_3d, colors_rgb, scale=3):
                    # 렌더링을 scale(기본 3배) 해상도로 키워서, 복셀의 점들이 너무 커져 뭉개지는 현상 방지.
                    render_w = target_w * scale
                    render_h = target_h * scale
                    
                    # 스케일된 카메라 매트릭스 적용
                    cam_mat_scaled = camera_matrix.copy()
                    cam_mat_scaled[0, 0] *= scale
                    cam_mat_scaled[1, 1] *= scale
                    cam_mat_scaled[0, 2] *= scale
                    cam_mat_scaled[1, 2] *= scale

                    img = np.zeros((render_h, render_w, 3), dtype=np.uint8)
                    if len(pts_3d) == 0: return cv2.resize(img, (target_w, target_h))
                    
                    p2d, _ = cv2.projectPoints(pts_3d.astype(np.float32), rvec.astype(np.float32), tvec, cam_mat_scaled, None)
                    p2d = np.round(p2d.reshape(-1, 2)).astype(int)
                    z_vals = pts_3d[:, 2]
                    y_vals = pts_3d[:, 1] # 높이(y축) 기준으로 바닥/객체 구분
                    v_mask = z_vals > 0
                    if not np.any(v_mask): return cv2.resize(img, (target_w, target_h))
                    
                    p2d, colors_rgb, z_vals, y_vals = p2d[v_mask], colors_rgb[v_mask], z_vals[v_mask], y_vals[v_mask]
                    
                    # 바닥 평면과 객체의 극적인 대비를 위한 색상 톤 조절 (Y-axis gradient)
                    # 일반적으로 카메라 좌표계에서 Y축은 아래로 향하므로 y값이 작을수록(위) 붉게, 클수록(아래 지면) 푸르게
                    y_min, y_max = np.min(y_vals), np.max(y_vals)
                    norm_y = (y_vals - y_min) / (y_max - y_min + 1e-6)
                    
                    # 가파른 곡선을 적용하여 지면과 물체의 경계를 뚜렷하게 (x^3 or sigmoid-like)
                    sharp_grade = np.clip(np.power(norm_y, 4), 0, 1)

                    # 객체(상단)는 주황/노랑 계열, 바닥면은 어두운 파랑/검정 계열로 대비
                    col_r = (255 * (1.0 - sharp_grade)).astype(np.uint8)
                    col_g = (150 * (1.0 - sharp_grade)).astype(np.uint8)
                    col_b = (100 + 155 * sharp_grade).astype(np.uint8)
                    
                    # 기존 색상 모드(color_mode='height')에서 넘어온 원본 색상이 맘에 든다면
                    # 이 커스텀 컬러링보다는 형태적인 음영으로만 대비를 줄 수도 있습니다.
                    # 여기서는 색상 자체를 강력하게 대비시켰습니다.
                    colors_rgb = np.stack([col_r, col_g, col_b], axis=-1)

                    # 뒤부터 앞으로 그리기 (Depth Sorting)
                    s_idx = np.argsort(-z_vals)
                    p2d, colors_rgb = p2d[s_idx], colors_rgb[s_idx]
                    
                    b_mask = (p2d[:,0]>=0) & (p2d[:,0]<render_w) & (p2d[:,1]>=0) & (p2d[:,1]<render_h)
                    if not np.any(b_mask): return cv2.resize(img, (target_w, target_h))
                    
                    p2d, colors_rgb = p2d[b_mask], colors_rgb[b_mask]
                    
                    img[p2d[:,1], p2d[:,0]] = colors_rgb[:, ::-1]
                    
                    # 복셀 간 빈틈을 메우되, 해상도가 커졌으므로 kernel 크기도 비례해서 세팅. 
                    # 이전보다 조금 더 강하게 타원(ellipse) 구조 요소를 사용하여 부드럽고 둥글게 블렌딩
                    kernel_sz = scale + 2 
                    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_sz, kernel_sz)))
                    
                    # 날카로움을 주기 위한 언샤프 마스킹(Unsharp Masking) / 엣지 강화 필터 적용
                    blurred = cv2.GaussianBlur(img, (0, 0), 2.0)
                    img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
                    
                    # 원래 해상도로 부드럽게 복구 (Anti-aliasing 효과 포함하여 작고 세밀하게 보임)
                    return cv2.resize(img, (target_w, target_h), interpolation=cv2.INTER_AREA)

                v_pts, v_cols = get_voxel_points_and_colors(voxel_map, voxel_info, color_mode='height')
                voxel_proj_img = make_screenshot(v_pts, v_cols)

                vc_pts, vc_cols = get_combined_pcd_and_voxel_points_and_colors(pcd_np, img_color_flat, voxel_map, voxel_info, color_mode='height')
                voxel_comb_proj_img = make_screenshot(vc_pts, vc_cols)

                # Create Canvas and save components
                merge_dir = os.path.join(output_directory, 'merge')
                os.makedirs(merge_dir, exist_ok=True)
                
                # Save standalone projection images
                os.makedirs(os.path.join(output_directory, 'pcd', 'ply_proj'), exist_ok=True)
                cv2.imwrite(os.path.join(output_directory, 'pcd', 'ply_proj', f'{file_stem}_ply_proj.png'), ply_proj_img)
                os.makedirs(os.path.join(output_directory, 'pcd', 'voxel_proj'), exist_ok=True)
                cv2.imwrite(os.path.join(output_directory, 'pcd', 'voxel_proj', f'{file_stem}_voxel_proj.png'), voxel_proj_img)
                
                # Also save the newly generated standalone voxel images to the same folders as the ply files
                cv2.imwrite(os.path.join(voxel_dir, f'{file_stem}_voxel.png'), voxel_proj_img)
                cv2.imwrite(os.path.join(voxel_comb_dir, f'{file_stem}_combined_voxel.png'), voxel_comb_proj_img)

                def add_label(img, text):
                    img_labeled = img.copy()
                    font_scale = max(0.8, img.shape[1] / 600.0)
                    thickness = max(2, int(font_scale * 2))
                    label_size, _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
                    cv2.rectangle(img_labeled, (0, 0), (label_size[0] + 10, label_size[1] + 20), (0, 0, 0), -1)
                    cv2.putText(img_labeled, text, (5, label_size[1] + 10), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
                    return img_labeled

                # Target display size for 4x2 grid to make sure final width >= 2560 (QHD width)
                panel_w = max(640, target_w)
                panel_h = int(panel_w * (target_h / target_w))
                panel_size = (panel_w, panel_h)

                c_cost = add_label(cv2.resize(costmap_colored, panel_size, interpolation=cv2.INTER_AREA), "Costmap")
                c_terr = add_label(cv2.resize(terrain_colored, panel_size, interpolation=cv2.INTER_AREA), "Terrain")
                c_driv = add_label(cv2.resize(drivable_colored, panel_size, interpolation=cv2.INTER_AREA), "Drivable")
                c_rgb = add_label(cv2.resize(img_color_orig_bgr, panel_size, interpolation=cv2.INTER_AREA), "RGB")
                c_depth = add_label(cv2.resize(img_vis_colored_depth, panel_size, interpolation=cv2.INTER_AREA), "Depth")
                c_ply = add_label(cv2.resize(ply_proj_img, panel_size, interpolation=cv2.INTER_AREA), "PLY Camera View")
                c_vox = add_label(cv2.resize(voxel_proj_img, panel_size, interpolation=cv2.INTER_AREA), "Voxel Camera View")
                c_blank = np.zeros_like(c_cost)

                row1 = np.hstack((c_rgb, c_depth, c_ply, c_vox))
                row2 = np.hstack((c_cost, c_terr, c_driv, c_blank))
                canvas = np.vstack((row1, row2))

                cv2.imwrite(os.path.join(merge_dir, f'{file_stem}_canvas.png'), canvas)

            except Exception as e:
                print(f"Failed to generate 3D maps or canvas: {e}")

            # ---------------------------------------------------------
            # [Visualizaion Option]
            # --visualize 인자가 있을 때만 창을 띄움
            # ---------------------------------------------------------
            if args.visualize:
                filename = os.path.join(output_directory, f'{file_stem}.png')
                print(f"Viewing: {file_stem} (Saved to {out_color})")
                
                win_name = "Depth Viewer (Press 'ESC' to quit, any key for next)"
                cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
                cv2.setMouseCallback(win_name, mouse_callback, depth_m)
                
                # 이미지 표시 및 키 대기
                cv2.imshow(win_name, img_vis_colored_depth)
                key = cv2.waitKey(0) & 0xFF # 무한 대기 (사용자가 키를 누를 때까지)
                
                if key == 27:  # Press 'Esc' key to exit completely
                    print("Exiting visualization...")
                    break
        
        if map_time_accumulators['count'] > 0:
            count = map_time_accumulators['count']
            print("\n=== Average Map Generation Time Profiling ===")
            print(f"Costmap:  {map_time_accumulators['costmap'] / count:.4f} s")
            print(f"Terrain:  {map_time_accumulators['terrain'] / count:.4f} s")
            print(f"Drivable: {map_time_accumulators['drivable'] / count:.4f} s")
            print(f"Voxel:    {map_time_accumulators['voxel'] / count:.4f} s")
            print(f"Total Maps: {map_time_accumulators['total'] / count:.4f} s")
            print("=============================================\n")

        if args.visualize:
            cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # ---------------------------------------------------------
    # [Checkpoint Path History]
    # Original: checkpoints/defomstereo_vits_sceneflow.pth
    # Adjusted: checkpoints/defomstereo_vits_rvc.pth (Fine-tuned Model)
    # ---------------------------------------------------------
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        # default= "checkpoints/defomstereo_vits_rvc.pth",
                        default= "checkpoints/defomstereo_vits_sceneflow.pth",
                        # default="checkpoints/defomstereo_vits_sceneflow/defomstereo_vits_sceneflow_200000.pth",
                        # default = "results/train_rvc_default/defomstereo_vitn_rvc/defomstereo_vitn_rvc_020000.pth",
                        # default = "results/train/defomstereo_custom_vitn_sceneflow/defomstereo_custom_vitn_sceneflow_200000.pth",
                        # required=True
                        )
    
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')

    # ---------------------------------------------------------
    # [Input Dataset Paths]
    # Original: ./demo-imgs/*/im0.png
    # Adjusted: datasets/clobot_office/*_0.png (Custom Dataset)
    # ---------------------------------------------------------
    # 입력 데이터 경로 (원본) - Reference
    # parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="./demo-imgs/*/im0.png")
    # parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="./demo-imgs/*/im1.png")
    # 입력 데이터 경로 (CloBot 사무실 데이터셋) - Current Default
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="datasets/clobot_office/*_0.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="datasets/clobot_office/*_1.png")

    # ---------------------------------------------------------
    # [Output Directory]
    # Original: images_DEFOM_clobot_protocol1_results/
    # Adjusted: results/demo/images_clobot_office/
    # ---------------------------------------------------------
    # parser.add_argument('--output_directory', help="directory to save output", default="images_DEFOM_clobot_protocol1_results/")
    parser.add_argument('--output_directory', help="directory to save output", default="results/demo/images_clobot_office/")
    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # ---------------------------------------------------------
    # [New Arguments for Flexibility]
    # ---------------------------------------------------------
    # 해상도 변경: --input_size 200 320 (Height Width)
    parser.add_argument('--input_size', nargs='+', type=int, default=[224, 320], 
                        help='Resize input image to (Height, Width). e.g., --input_size 200 320. If not set, use original size.')
    
    # 시각화 활성화: --visualize (마우스 클릭 기능 포함)
    parser.add_argument('--visualize', action='store_true', help='Enable interactive visualization with mouse callback')

    # ---------------------------------------------------------
    # [Inference Iterations (Speed vs Accuracy)]
    # valid_iters: Original(32) -> Adjusted(8) for faster inference
    # scale_iters: Original(8) -> Adjusted(2) for faster inference
    # ---------------------------------------------------------
    parser.add_argument('--valid_iters', type=int, default=15, help='number of flow-field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=5, help="number of scaling updates to the disparity field in each forward pass.")

    # ---------------------------------------------------------
    # [Model Architecture]
    # dinov2_encoder: Original(vits) -> Added choices(vitn, vitt)
    # ---------------------------------------------------------
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vitn', 'vitt', 'vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--idepth_scale', type=float, default=0.5, help="the scale of inverse depth to initialize disparity")
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
    parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
    parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
    parser.add_argument('--corr_levels', type=int, default=2, help="number of levels in the correlation pyramid")
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    # parser.add_argument('--corr_radius', type=int, default=3, help="width of the correlation pyramid")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    # parser.add_argument('--n_gru_layers', type=int, default=2, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)