import argparse
import torch
import cv2
import sys
import os
import numpy as np
import yaml
from omegaconf import OmegaConf

os.environ['OMP_NUM_THREADS'] = '1'
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['VECLIB_MAXIMUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.inpainting.saicinpainting.evaluation.utils import move_to_device
from src.inpainting.saicinpainting.training.trainers import load_checkpoint
from src.inpainting.saicinpainting.evaluation.data import pad_tensor_to_modulo
from util import rs

def inpaint_alone(img, mask, ckpt_path, configs_path, mod=8, device='cuda'):
    if not isinstance(mask, np.ndarray):
        print("Userwarning: mask is not numpy array, convert to numpy array")
        try:
            mask = np.array(mask)
        except:
            print("Error: mask convertion failed")
    else:
        if not len(mask.shape) == 2:
            raise ValueError("mask should be 2D")
        
    device = torch.device(device)
    
    mask = mask * 255 if mask.max() <= 1 else mask
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()

    predict_configs = OmegaConf.load(configs_path)
    predict_configs.model.path = ckpt_path
    
    train_config_path = os.path.join(predict_configs.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_configs.model.path, 'models', predict_configs.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_configs.get('refine', False):
        model.to(device)
        
    batch, unpad_to_size = preprocess(img, mask, mod, device)
    batch = model(batch)
    
    cur_res = batch[predict_configs.out_key][0].permute(1, 2, 0).detach().cpu().numpy()
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
        
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res

def inpaint(model, batch, unpad_to_size):  
    batch = model(batch)

    cur_res = batch['inpainted'][0].permute(1, 2, 0).detach().cpu().numpy()
    if unpad_to_size is not None:
        orig_height, orig_width = unpad_to_size
        cur_res = cur_res[:orig_height, :orig_width]
        
    cur_res = np.clip(cur_res * 255, 0, 255).astype('uint8')
    return cur_res
    
def preprocess(img, mask, mod=8, device='cuda'):
    if not isinstance(mask, np.ndarray):
        print("Userwarning: mask is not numpy array, convert to numpy array")
        try:
            mask = np.array(mask)
        except:
            print("Error: mask convertion failed")
    else:
        if not len(mask.shape) == 2:
            raise ValueError("mask should be 2D")
        
    device = torch.device(device)
    
    mask = mask * 255 if mask.max() <= 1 else mask
    img = torch.from_numpy(img).float().div(255.)
    mask = torch.from_numpy(mask).float()
    
    batch = {}
    batch['mask'] = mask[np.newaxis, np.newaxis, :]
    batch['image'] = img.permute(2, 0, 1).unsqueeze(0)

    unpad_to_size = [batch['image'].shape[2], batch['image'].shape[3]]

    batch['mask'] = pad_tensor_to_modulo(batch['mask'], mod)
    batch['image'] = pad_tensor_to_modulo(batch['image'], mod)
    batch = move_to_device(batch, device)
    batch['mask'] = (batch['mask'] > 0) * 1
    
    return batch, unpad_to_size

def build_inpainter(configs_path, ckpt_path, device='cuda'):
    device = torch.device(device)
    
    predict_configs = OmegaConf.load(configs_path)
    predict_configs.model.path = ckpt_path
    
    train_config_path = os.path.join(predict_configs.model.path, 'config.yaml')
    with open(train_config_path, 'r') as f:
            train_config = OmegaConf.create(yaml.safe_load(f))
            
    train_config.training_model.predict_only = True
    train_config.visualizer.kind = 'noop'

    checkpoint_path = os.path.join(predict_configs.model.path, 'models', predict_configs.model.checkpoint)
    model = load_checkpoint(train_config, checkpoint_path, strict=False, map_location='cpu')
    model.freeze()
    if not predict_configs.get('refine', False):
        model.to(device)
        
    return model

def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('-img_path', '--img_path', type=str, required=True, default=None, help='Path to the image to be processed')
    parser.add_argument('-matte_path', '--matte_path', type=str, default=None, help='Path to the matte to be processed')
    parser.add_argument('-mask_path', '--mask_path', type=str, default=None, help='Path to the mask to be processed')
    parser.add_argument('-out_path', '--out_path', type=str, default='results', help='Path to the output results')
    parser.add_argument('-inp_config', '--inp_config', type=str, default='configs/lama_configs/predict/predict.yaml', help='lama config path')
    parser.add_argument('-inp_ckpt', '--inp_ckpt', type=str, default='./pretrained/big-lama/', help='lama checkpoint')
    parser.add_argument('-mod', '--mod', type=int, default=8, help='modulo')
    parser.add_argument('-device', '--device', type=str, default='cuda', help='device')

    return parser.parse_args()
def main(args):
    if not args.mask_path:
        if not args.matte_path:
            raise ValueError("mask_path or matte_path should be provided")
        matte = rs.read_matte(args.matte_path)
        mask = rs.matte2mask(matte, thres=0)
    else:
        mask = rs.read_mask(args.mask_path)
    img = rs.read_img(args.img_path)
    result = inpaint_alone(img, mask, args.inp_ckpt, args.inp_config, args.mod, args.device)
    
    out_path = os.path.join(args.out_path, 'inpaint_result.png')
    rs.save_img(result, out_path)
    print("Successfully saved inpainted image")

if __name__ == '__main__':
    args = args_parse()
    main(args)
    