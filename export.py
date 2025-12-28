import sys
sys.path.append('core')

import argparse
import glob
import numpy as np
import torch
import cv2
import os
from pathlib import Path
from tqdm import tqdm
from matplotlib import pyplot as plt
from natsort import natsorted

# Core Model
from core.defom_stereo import DEFOMStereo
from utils.utils import InputPadder

# OpenVINO & NNCF
import openvino as ov
from openvino.runtime import Core, serialize
from openvino.tools.mo import convert_model
import nncf
from torch.utils.data import Dataset, DataLoader

DEVICE = 'cpu' # Export는 주로 CPU에서 수행

# ---------------------------------------------------------
# [Export Wrapper]
# 모델의 출력을 ONNX/OpenVINO가 이해하기 쉬운 단일 텐서로 고정
# ---------------------------------------------------------
class ExportWrapper(torch.nn.Module):
    def __init__(self, net, iters=16, scale_iters=8):
        super().__init__()
        self.net = net
        self.iters = iters
        self.scale_iters = scale_iters
        
    def forward(self, left, right):
        # test_mode=True: 학습용 output 제외
        output = self.net(left, right, iters=self.iters, scale_iters=self.scale_iters, test_mode=True)
        
        # 리스트/튜플/딕셔너리 출력을 단일 텐서(Pred Disp)로 변환
        if isinstance(output, (list, tuple)):
            return output[-1]
        elif isinstance(output, dict):
            if 'pred_disp' in output:
                return output['pred_disp']
            return list(output.values())[-1]
            
        return output

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
    
    # 함수 객체에 target_size가 설정되어 있으면 해당 크기로 리사이징
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

    padder = InputPadder(image1.shape, divis_by=32)
    image1_padded, image2_padded = padder.pad(image1, image2)

    return image1_padded, image2_padded, (h_ori1, w_ori1), (h_ori2, w_ori2), padder

# ---------------------------------------------------------
# [Calibration Dataset]
# NNCF INT8 Quantization을 위한 데이터셋 로더
# 실제 데이터의 분포(Distribution)를 파악하기 위해 사용됨
# ---------------------------------------------------------
class CustomImageDataset(Dataset):
    def __init__(self, root_path, ext_list=(".jpg", "*.png", "*.ppm")):
        if root_path == "im01":
            folders = ["demo-imgs/"] # Calibration용 이미지가 있는 폴더
            left_list = ['im0.', 'view1.'] 
            
            self.left_img_list = []
            self.right_img_list = []
            
            print("Scanning for calibration images...")
            for folder in tqdm(folders):
                for root, _, files in os.walk(folder):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in [os.path.splitext(ext)[1].lower() for ext in ext_list]:
                            if "image_2" in root: # KITTI
                                left_file_path = os.path.join(root, f)
                                right_file_path = left_file_path.replace("image_2", "image_3")
                                if os.path.exists(right_file_path):
                                    self.left_img_list.append(left_file_path)
                                    self.right_img_list.append(right_file_path)
                                    continue
                            else: # General
                                for left_name in left_list:    
                                    if left_name in os.path.basename(f):
                                        left_file_path = os.path.join(root, f)
                                        n4 = left_file_path.replace(left_name, "im4.")
                                        n1 = left_file_path.replace(left_name, "im1.")
                                        if os.path.exists(n4):
                                            self.left_img_list.append(left_file_path)
                                            self.right_img_list.append(n4)
                                            break
                                        elif os.path.exists(n1):
                                            self.left_img_list.append(left_file_path)
                                            self.right_img_list.append(n1)
                                            break
                                    elif left_name in os.path.basename(f):
                                        left_file_path = os.path.join(root, f)
                                        n5 = f.replace(left_name, "view5.")
                                        if os.path.exists(n5):
                                            self.left_img_list.append(left_file_path)
                                            self.right_img_list.append(n5)
                                            break
                                        
    def __len__(self):
        return len(self.left_img_list)
    
    def __getitem__(self, idx):
        l_path = self.left_img_list[idx]
        r_path = self.right_img_list[idx]
        
        # load_image 사용 (타겟 사이즈 적용됨)
        left_img, _ = load_image(l_path)
        right_img, _ = load_image(r_path)
        
        padder = InputPadder(left_img.shape, divis_by=32)
        pad_left_img , pad_right_img = padder.pad(left_img, right_img)
        
        return pad_left_img.cpu().numpy(), pad_right_img.cpu().numpy()
    
def demo(args):
    # 1. 모델 로드
    model = DEFOMStereo(args)
    # CUDA에서 학습된 가중치라도 Export를 위해 CPU로 로드 가능
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda' if torch.cuda.is_available() else 'cpu')
    
    if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()
    
    # Export Wrapper 적용 (ONNX 입출력 고정)
    wrapper = ExportWrapper(model, iters=args.valid_iters, scale_iters=args.scale_iters).eval()

    output_directory = Path(os.path.dirname(args.restore_ckpt))
    output_onnx = args.restore_ckpt.replace(".pth", ".onnx")
    output_ir = args.restore_ckpt.replace(".pth", ".xml")

    with torch.no_grad():
        left_images = natsorted(glob.glob(args.left_imgs, recursive=True))
        right_images = natsorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Output directory: {output_directory}")

        # ---------------------------------------------------------
        # [Export Input Resolution]
        # [New] Argument '--input_size'를 통해 Export할 모델의 입력 해상도를 결정합니다.
        # 이 사이즈가 ONNX/OpenVINO 모델의 Static Shape으로 고정됩니다.
        # ---------------------------------------------------------
        if args.input_size is not None:
            load_image.target_size = tuple(args.input_size) # (Height, Width)
            print(f" Exporting model with fixed input size: {load_image.target_size} (H, W)")
        else:
            # Default fallback (if None)
            load_image.target_size = (200, 320)
            print(f" Exporting model with default size: {load_image.target_size} (H, W)")

        # Dummy Input 생성 (Static Shape 결정을 위해 필요)
        if len(left_images) > 0:
            sample_image1, _ = load_image(left_images[0])
            sample_image2, _ = load_image(right_images[0])
            padder = InputPadder(sample_image1.shape, divis_by=32)
            pad_sample_image1, pad_sample_image2 = padder.pad(sample_image1, sample_image2)
            dummy_input = (pad_sample_image1.cpu().float(), pad_sample_image2.cpu().float())
        else:
            print("Warning: No images found. Using random dummy input.")
            h, w = load_image.target_size
            dummy_input = (torch.randn(1, 3, h, w), torch.randn(1, 3, h, w))

        # ---------------------------------------------------------
        # Step 1. ONNX Export
        # ---------------------------------------------------------
        if os.path.exists(output_onnx):
            print(f"[Skip] ONNX model already exported: {output_onnx}")
        else:
            print("Exporting ONNX model...")
            torch.onnx.export(
                wrapper,
                dummy_input,
                output_onnx,
                input_names=["left","right"], 
                output_names=["pred_disp"],
                opset_version=16,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL
            )
            print(f"ONNX model exported to {output_onnx}")
        
        # ---------------------------------------------------------
        # Step 2. OpenVINO IR Conversion
        # ---------------------------------------------------------
        if os.path.exists(output_ir) and os.path.exists(output_ir.replace(".xml", ".bin")):
            print(f"[Skip] OpenVINO IR model already exported: {output_ir}")
        else:
            print("Converting to OpenVINO IR...")
            # FP16 변환을 원하면 compress_to_fp16=True 설정
            ov_model = convert_model(input_model=output_onnx, compress_to_fp16=False)
            ov_model.outputs[0].tensor.set_names({"pred_disp"})
            ov.serialize(ov_model, output_ir, output_ir.replace('.xml', '.bin'))
            print(f"IR model saved to {output_ir}")
            
        # ---------------------------------------------------------
        # Step 3. NNCF INT8 Quantization
        # ---------------------------------------------------------
        q_model_path = output_ir.replace('.xml','_quant.xml')
        
        if os.path.exists(q_model_path) and os.path.exists(q_model_path.replace(".xml", ".bin")):
            print(f"[Skip] NNCF Quantized model already exported: {q_model_path}")
        else:
            print("Starting NNCF INT8 Quantization...")
            ie = Core()
            ir_model = ie.read_model(output_ir, output_ir.replace('.xml', '.bin'))
            
            # Calibration 데이터셋 준비
            calib_dataset = CustomImageDataset(root_path="im01")
            calib_loader = DataLoader(calib_dataset, batch_size=1)
            
            def transform_fn(data_item):
                return data_item[0][0].numpy(), data_item[1][0].numpy()
            
            calibration_dataset = nncf.Dataset(calib_loader, transform_fn)
            
            print(" - Quantizing model...")
            q_model = nncf.quantize(
                    model=ir_model,
                    calibration_dataset=calibration_dataset,
            )
            
            serialize(q_model, q_model_path, q_model_path.replace('.xml', '.bin'))
            print(f"Quantized model saved to {q_model_path}")
            
        # ---------------------------------------------------------
        # Step 4. Inference Test (Visualization with PyTorch)
        # PyTorch 원본 모델을 사용하여 결과가 정상적인지 확인
        # ---------------------------------------------------------
        print("Running inference visualization...")
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1_padded, image2_padded, _, _, padder = preprocess(imfile1, imfile2)

            with torch.no_grad():
                disp_pr = model(image1_padded, image2_padded, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
            
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()

            file_stem = os.path.splitext(os.path.basename(imfile1))[0] + '_' + args.restore_ckpt.split('/')[-1][:-4]
            
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp_pr)
            
            plt.imsave(output_directory / f"{file_stem}.png", disp_pr, cmap='jet')
        print("Visualization complete.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    # ---------------------------------------------------------
    # [Checkpoint Path History]
    # Original: checkpoints/defomstereo_vits_sceneflow.pth
    # Adjusted: checkpoints/defomstereo_vits_rvc.pth (Fine-tuned Model)
    # ---------------------------------------------------------
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        default="checkpoints/defomstereo_vits_rvc.pth")
    
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    
    # ---------------------------------------------------------
    # [Input Dataset Paths]
    # Used for dummy input generation & calibration
    # ---------------------------------------------------------
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="demo/*_left.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="demo/*_right.png")
    
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # ---------------------------------------------------------
    # [Export Input Resolution]
    # ---------------------------------------------------------
    # 고정된 해상도로 모델을 내보냅니다.
    # Default: 200 320 (Edge Device Speed Optimization)
    parser.add_argument('--input_size', nargs='+', type=int, default=[200, 320], 
                        help='Fixed input resolution (H W) for the exported model. e.g., --input_size 200 320')
    
    # ---------------------------------------------------------
    # [Inference Iterations (Speed vs Accuracy)]
    # Export된 모델은 이 반복 횟수를 그래프 내에 고정합니다.
    # ---------------------------------------------------------
    # valid_iters: Original(32) -> Adjusted(8) for faster inference
    parser.add_argument('--valid_iters', type=int, default=8, help='number of flow-field updates during forward pass')
    # scale_iters: Demo(2) vs Export(4)
    # Export 시에는 정확도를 위해 약간 높게 설정 (필요시 2로 낮출 수 있음)
    parser.add_argument('--scale_iters', type=int, default=4, help="number of scaling updates to the disparity field in each forward pass.")

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
    parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
    
    parser.add_argument('--scale_list', type=float, nargs='+', default=[0.125, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5, 2.0],
                        help='the list of scaling factors of disparity')
    parser.add_argument('--scale_corr_radius', type=int, default=2,
                        help="width of the correlation pyramid for scaled disparity")

    parser.add_argument('--n_downsample', type=int, default=2, choices=[2, 3], help="resolution of the disparity field (1/2^K)")
    parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
    parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
    
    args = parser.parse_args()

    demo(args)