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
            out_png = os.path.join(output_directory, f'{file_stem}_disp_colored.png')
            # 필요한 경우 주석 해제하여 저장
            # cv2.imwrite(out_png, img_vis_colored, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

            # (5-2) Depth Map 변환 및 저장
            # ---------------------------------------------------------
            # [ZED X 카메라 파라미터 적용]
            # Baseline: 0.12m
            # Focal Length: 2.2mm
            # Pixel Size: 0.003mm
            # ---------------------------------------------------------
            depth_m = scaled_disp_to_depth(disp_rescaled, baseline_m=0.12, f_mm=2.2, pixel_size_mm=0.003) # ZED X
            
            depth_m = np.clip(depth_m, 0.0, 4.0) # 시각화를 위해 Depth 클리핑 (0m ~ 4m)

            # Depth 정규화 및 시각화
            depth_m_normalized = normalize_disp(depth_m)
            img_vis_colored_depth = visualize_from_normalized_disp(depth_m_normalized)

            out_color = os.path.join(output_directory, f'{file_stem}_depth_colored.png')
            cv2.imwrite(out_color, img_vis_colored_depth, [int(cv2.IMWRITE_PNG_COMPRESSION), 0])

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
                        default= "checkpoints/defomstereo_vits_rvc.pth",
                        # default= "checkpoints/defomstereo_vits_sceneflow.pth",
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
    parser.add_argument('--input_size', nargs='+', type=int, default=[200, 320], 
                        help='Resize input image to (Height, Width). e.g., --input_size 200 320. If not set, use original size.')
    
    # 시각화 활성화: --visualize (마우스 클릭 기능 포함)
    parser.add_argument('--visualize', action='store_true', help='Enable interactive visualization with mouse callback')

    # ---------------------------------------------------------
    # [Inference Iterations (Speed vs Accuracy)]
    # valid_iters: Original(32) -> Adjusted(8) for faster inference
    # scale_iters: Original(8) -> Adjusted(2) for faster inference
    # ---------------------------------------------------------
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=2, help="number of scaling updates to the disparity field in each forward pass.")

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