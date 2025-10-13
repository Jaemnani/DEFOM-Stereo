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
import os
import skimage.io
import cv2
# from utils.frame_utils import readPFM
import onnxruntime as ort
from time import time

import openvino as ov
from openvino.runtime import Core, serialize
from openvino.tools.mo import convert_model
import nncf
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset, DataLoader

from tqdm import tqdm

DEVICE = 'cpu'
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# iters/test_mode 고정 래퍼 (ONNX/IR에서 kwargs 보존 안 되므로)
class ExportWrapper(torch.nn.Module):
    def __init__(self, net, iters=16, scale_iters=8):
        super().__init__()
        self.net = net
        self.iters = iters
        self.scale_iters = scale_iters
        
    def forward(self, left, right):
        # 좌/우 입력은 pad가 적용된 텐서(NCHW, float32, 0~255)라고 가정
        # disp_pr = model(image1, image2, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
        return self.net(left, right, iters=self.iters, scale_iters=self.scale_iters ,test_mode=True)
    
def load_image(imfile):
    img = np.array(Image.open(imfile)).astype(np.uint8)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    return img[None].to(DEVICE)

def custom_transform(img : Image.Image):
    img_f32 = np.array(img).astype(np.float32)
    img_transposed = (img_f32, (2, 0, 1))
    
class CustomImageDataset(Dataset):
    def __init__(self, root_path, ext_list=(".jpg", "*.png", "*.ppm")):
        if root_path == "im01":
            
            folders = ["datasets/"]
            left_list = ['im0.', 'view1.']
            
            self.left_img_list = []
            self.right_img_list = []
            for folder in tqdm(folders):
                for root, _, files in os.walk(folder):
                    for f in files:
                        if os.path.splitext(f)[1].lower() in [os.path.splitext(ext)[1].lower() for ext in ext_list]:
                            if "image_2" in root: # for kitti 2015
                                left_file_path = os.path.join(root, f)
                                right_file_path = left_file_path.replace("image_2", "image_3")
                                if os.path.exists(right_file_path):
                                    self.left_img_list.append(left_file_path)
                                    self.right_img_list.append(right_file_path)
                                    continue
                            else:
                                for left_name in left_list:    
                                    if left_name in f[0]:
                                        n4 = f.replace(left_name, "im4.")
                                        n1 = f.replace(left_name, "im1.")
                                        if os.path.exists(n4):
                                            self.left_img_list.append(os.path.join(root, f))
                                            self.right_img_list.append(os.path.join(root, n4))
                                            break
                                        elif os.path.exists(n1):
                                            self.left_img_list.append(os.path.join(root, f))
                                            self.right_img_list.append(os.path.join(root, n1))
                                            break
                                    elif left_name in f[1]:
                                        n5 = f.replace(left_name, "view5.")
                                        if os.path.exists(n5):
                                            self.left_img_list.append(os.path.join(root, f))
                                            self.right_img_list.append(os.path.join(root, n5))
                                            break
                                        
    def __len__(self):
        return len(self.left_img_list)
    
    def __getitem__(self, idx):
        l_path = self.left_img_list[idx]
        r_path = self.right_img_list[idx]
        left_img = load_image(l_path)
        right_img = load_image(r_path)
        padder = InputPadder(left_img.shape, divis_by=32)
        pad_left_img , pad_right_img =padder.pad(left_img, right_img)
        input_left_npimg = pad_left_img.cpu().numpy()
        input_right_npimg = pad_right_img.cpu().numpy()
        # return {"left":input_left_npimg, "right":input_right_npimg}
        return input_left_npimg, input_right_npimg
    
def demo(args):
    model = DEFOMStereo(args)
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')
    if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)

    model.to(DEVICE)
    model.eval()#.cpu()
    
    wrapper = ExportWrapper(model, iters=args.valid_iters, scale_iters=args.scale_iters).eval()


    output_directory = Path(args.output_directory)
    os.makedirs(output_directory, exist_ok=True)
    os.makedirs(Path(os.path.dirname(args.output_onnx)), exist_ok=True)
    os.makedirs(Path(os.path.dirname(args.output_ir)), exist_ok=True)
    
    # output_directory.mkdir(exist_ok=True)

    with torch.no_grad():
        left_images = sorted(glob.glob(args.left_imgs, recursive=True))
        right_images = sorted(glob.glob(args.right_imgs, recursive=True))
        print(f"Found {len(left_images)} images. Saving files to {output_directory}/")

        left_name = left_images[0]
        right_name = right_images[0]
        sample_image1 = load_image(left_name)
        sample_image2 = load_image(right_name)
        padder = InputPadder(sample_image1.shape, divis_by=32)
        pad_sample_image1, pad_sample_image2 = padder.pad(sample_image1, sample_image2)
        
        
        # ONNX CONVERTING
        dummy_input = (pad_sample_image1.cpu().float(), pad_sample_image2.cpu().float())
        dummy_input = (pad_sample_image1.float(), pad_sample_image2.float())
        
        dynamic_axes = {
            "left": {0:"N", 2:"H", 3:"W"},
            "right": {0:"N", 2:"H", 3:"W"},
            "pred_disp": {0:"N", 2:"H", 3:"W"},
        }

        if os.path.exists(args.output_onnx):
            print("ONNX model already exported.")
        else:
            torch.onnx.export(
                # model_cpu,
                wrapper,
                dummy_input,
                args.output_onnx,
                input_names=["left","right"], 
                output_names=["pred_disp"],
                dynamic_axes=dynamic_axes,
                opset_version=16,
                do_constant_folding=True,
                training=torch.onnx.TrainingMode.EVAL
            )
            print(f"ONNX model exported to {args.output_onnx}")
        
        # OPENVINO CONVERTING
        if os.path.exists(args.output_ir) and os.path.exists(args.output_ir.replace(".xml", ".bin")):
            print("OPENVINO IR model already exported.")
        else:
            ov_model = convert_model(input_model = args.output_onnx, compress_to_fp16=False)
            ov_model.outputs[0].tensor.set_names({"pred_disp"})
            ov.serialize(ov_model, args.output_ir, args.output_ir.replace('.xml', '.bin'))
            
        # Openvino Quantize
        print("OPENVINO QUANT INT8")
        q_model_path = args.output_ir.replace('.xml','_quant.xml')
        if os.path.exists(q_model_path) and os.path.exists(q_model_path.replace(".xml", ".bin")):
            print("OPENVINO QUANT model already exported.")
        else:
            ie = Core()
            ir_model = ie.read_model(args.output_ir, args.output_ir.replace('.xml', '.bin'))
            print("openvino original model opened")
            calib_dataset = CustomImageDataset(root_path="im01")
            calib_loader = DataLoader(calib_dataset, batch_size=1)
            print(" calibration dataset loaded ")
            def transform_fn(data_item):
                return data_item[0][0].numpy(), data_item[1][0].numpy()
            calibration_dataset = nncf.Dataset(calib_loader, transform_fn)
            print("calibration start")
            q_model = nncf.quantize(
                    model=ir_model,
                    calibration_dataset = calibration_dataset,
            )
            print("serialize start")
            serialize(q_model, q_model_path, q_model_path.replace('.xml', '.bin'))
            
        # Inference TEST
        for (imfile1, imfile2) in tqdm(list(zip(left_images, right_images))):
            image1 = load_image(imfile1)
            image2 = load_image(imfile2)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            with torch.no_grad():
                disp_pr = model(image1, image2, iters=args.valid_iters, scale_iters=args.scale_iters, test_mode=True)
            disp_pr = padder.unpad(disp_pr).cpu().squeeze().numpy()

            file_stem = imfile1.split('/')[-1].split('_')[0]+'_'+args.restore_ckpt.split('/')[-1][:-4]
            if args.save_numpy:
                np.save(output_directory / f"{file_stem}.npy", disp_pr)
            plt.imsave(output_directory / f"{file_stem}.png", disp_pr, cmap='jet')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--restore_ckpt', help="restore checkpoint", 
                        default="checkpoints/defomstereo_vits_rvc.pth",
                        # required=True
                        )
    parser.add_argument('--save_numpy', action='store_true', help='save output as numpy arrays')
    parser.add_argument('-l', '--left_imgs', help="path to all first (left) frames", default="demo/*_left.png")
    parser.add_argument('-r', '--right_imgs', help="path to all second (right) frames", default="demo/*_right.png")
    parser.add_argument('--output_directory', help="directory to save output", default="demo")
    parser.add_argument("--output_onnx", help="path to save onnx", default="demo/models/model.onnx")
    parser.add_argument("--output_ir", help="path to save openvino ir", default="demo/models/openvino_ir.xml")
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')
    parser.add_argument('--scale_iters', type=int, default=8, help="number of scaling updates to the disparity field in each forward pass.")

    # Architecture choices
    # parser.add_argument('--dinov2_encoder', type=str, default='vitl', choices=['vits', 'vitb', 'vitl', 'vitg'])
    parser.add_argument('--dinov2_encoder', type=str, default='vits', choices=['vits', 'vitb', 'vitl', 'vitg'])
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
