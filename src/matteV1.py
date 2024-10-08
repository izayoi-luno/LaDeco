import argparse
import torch
import toml
import os
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import sys

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import matting
from src.segment import ResizeLongestSide
from util import rs, mam_util, plotter
from configs import CONFIG, load_config

@torch.no_grad()
def matte(
    input, 
    model, 
    maskguide=False, 
    twoside=False, 
    postprocess=False, 
    os8_width=10,
    os4_width=20,
    os1_width=10
    ):
    _, pred, post_mask = model.forward_inference(input)
    
    alpha_pred_os1, alpha_pred_os4, alpha_pred_os8 = pred['alpha_os1'], pred['alpha_os4'], pred['alpha_os8']
    alpha_pred_os8 = alpha_pred_os8[..., : input['pad_shape'][0], : input['pad_shape'][1]]
    alpha_pred_os4 = alpha_pred_os4[..., : input['pad_shape'][0], : input['pad_shape'][1]]
    alpha_pred_os1 = alpha_pred_os1[..., : input['pad_shape'][0], : input['pad_shape'][1]]

    alpha_pred_os8 = F.interpolate(alpha_pred_os8, input['ori_shape'], mode="bilinear", align_corners=False)
    alpha_pred_os4 = F.interpolate(alpha_pred_os4, input['ori_shape'], mode="bilinear", align_corners=False)
    alpha_pred_os1 = F.interpolate(alpha_pred_os1, input['ori_shape'], mode="bilinear", align_corners=False)
    
    if maskguide:
        if twoside:
            weight_os8 = mam_util.get_unknown_tensor_from_mask(post_mask, rand_width=os8_width, train_mode=False)
        else:
            weight_os8 = mam_util.get_unknown_tensor_from_mask_oneside(post_mask, rand_width=os8_width, train_mode=False)
        post_mask[weight_os8>0] = alpha_pred_os8[weight_os8>0]
        alpha_pred = post_mask.clone().detach()
    else:
        if postprocess:
            weight_os8 = mam_util.get_unknown_box_from_mask(post_mask)
            alpha_pred_os8[weight_os8>0] = post_mask[weight_os8>0]
        alpha_pred = alpha_pred_os8.clone().detach()
    
    if twoside:
        weight_os4 = mam_util.get_unknown_tensor_from_pred(alpha_pred, rand_width=os4_width, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = mam_util.get_unknown_tensor_from_pred(alpha_pred, rand_width=os1_width, train_mode=False)
    else:
        weight_os4 = mam_util.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=os4_width, train_mode=False)
        alpha_pred[weight_os4>0] = alpha_pred_os4[weight_os4>0]
        weight_os1 = mam_util.get_unknown_tensor_from_pred_oneside(alpha_pred, rand_width=os1_width, train_mode=False)
    
    alpha_pred[weight_os1>0] = alpha_pred_os1[weight_os1>0]
    alpha_pred = alpha_pred[0].cpu().numpy() * 255
    return alpha_pred.transpose(1, 2, 0).astype('uint8')

def build_matter(config_path, ckpt):
    with open(config_path) as f:
        load_config(toml.load(f)) 
    
    if CONFIG.is_default:
        raise ValueError("No .toml config loaded.") 

    # build model
    model = matting.get_generator_m2m(seg=CONFIG.model.arch.seg, m2m=CONFIG.model.arch.m2m)
    model.cuda()
    
    # load checkpoint
    checkpoint = torch.load(ckpt)
    model.m2m.load_state_dict(mam_util.remove_prefix_state_dict(checkpoint['state_dict']), strict=True)
    
    model = model.eval()
    n_parameters = sum(p.numel() for p in model.m2m.parameters() if p.requires_grad)
    print(f'Inferece with MAM model containing {n_parameters} params.')

    return model

def preprocess(img, target_length, prompt_mode, prompt):
    ori_size =  img.shape[:2]

    transform = ResizeLongestSide(target_length)
    
    img = transform.apply_image(img)
    img = torch.as_tensor(img).cuda()
    img = img.permute(2, 0, 1).contiguous()
    
    # normalize
    pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(3,1,1).cuda()
    pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(3,1,1).cuda()
    img = (img - pixel_mean) / pixel_std
    
    # padding
    h, w = img.shape[-2:]
    pad_size = img.shape[-2:]
    padh = 1024 - h
    padw = 1024 - w
    img = F.pad(img, (0, padw, 0, padh))

    input = None
    if prompt_mode == 'box':
        assert isinstance(prompt, np.ndarray) and prompt.shape[-1] == 4, \
            ValueError("box prompt should be a numpy array with shape [n, 4]")
        bbox = prompt
        bbox = transform.apply_boxes(bbox, ori_size)
        bbox = torch.as_tensor(bbox, dtype=torch.float).cuda()

        input = {'image': img[None, ...], 'bbox': bbox[None, ...], \
                  'ori_shape': ori_size, 'pad_shape': pad_size}
        
    elif prompt_mode == 'point':
        assert isinstance(prompt, np.ndarray) and prompt.shape[-1] == 3, \
            ValueError("point prompt should be a numpy array with shape [n, 3]")
        point = prompt[..., :2]
        point = transform.apply_coords(point, ori_size)
        label = prompt[..., -1]
        point = torch.as_tensor(point, dtype=torch.float).cuda()
        label = torch.as_tensor(label, dtype=torch.float).cuda()
        
        input = {'image': img[None, ...], 'point': point[None, ...], \
                  'label': label[None, ...], 'ori_shape': ori_size, 'pad_shape': pad_size}
    
    elif prompt_mode == 'text':
        # TODO
        pass

    return input
    
def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-img_path', '--img_path', type=str, required=True, default=None, help='Path to the image to be processed')
    parser.add_argument('-out_path', '--out_path', type=str, default='./results', help='Path to the output results')
    parser.add_argument('-mat_config', '--mat_config', type=str, default='configs/mam_configs/MAM-ViTB-8gpu.toml', help='MAM config path')
    parser.add_argument('-mat_ckpt', '--mat_ckpt', type=str, default='./pretrained/mam_vitb.pth', help='MAM checkpoint')
    parser.add_argument('-os8_width', '--os8_width', type=int, default=10, help="guidance threshold")
    parser.add_argument('-os4_width', '--os4_width', type=int, default=20, help="guidance threshold")
    parser.add_argument('-os1_width', '--os1_width', type=int, default=10, help="guidance threshold")
    parser.add_argument('-twoside', '--twoside', action='store_true', default=False, help='post process with twoside of the guidance')        
    parser.add_argument('-sam', '--sam', action='store_true', default=False, help='return mask')    
    parser.add_argument('-maskguide', '--maskguide', action='store_true', default=False, help='mask guidance')    
    parser.add_argument('-postprocess', '--postprocess', action='store_true', default=False, help='postprocess to remove bg')    
    parser.add_argument('-prompt_mode', '--prompt_mode', type=str, default='point', choices=['box', 'point', 'text'])
    parser.add_argument('-prompt', '--prompt', type=int, nargs='+', required=True, help='The coordinate of the point prompt, [coord_W coord_H].')
    parser.add_argument('-target_length', '--target_length', type=int, default=1024, help='target length of input image')
        
    return parser.parse_args()

def main():
    args = args_parse()
    img = rs.read_img(args.img_path)
    img_orin = img.copy()
    model = build_matter(args.mat_config, args.mat_ckpt)
    
    prompt = []
    if args.prompt_mode == 'point':
        if len(args.prompt) % 3 == 0:
            for i in range(0, len(args.prompt), 3):
                prompt.append(args.prompt[i : i + 3])
        else:
            raise ValueError("prompt should be a list of 3-element list")
    prompt = np.array(prompt)
    
    input = preprocess(img, args.target_length, args.prompt_mode, prompt)
    alpha_pred = matte(input, model, args.maskguide, args.twoside, args.postprocess, \
                       args.os8_width, args.os4_width, \
                       args.os1_width)
    
    out_path = os.path.join(args.out_path, 'matte_result.png')
    if args.prompt_mode == 'point':
        pointed_img_path = os.path.join(args.out_path, 'pointed_img.png')
    matted_img_path = os.path.join(args.out_path, 'matted_img.png')
    
    rs.save_matte(alpha_pred, out_path)
    
    plt.figure()
    plt.imshow(img_orin)
    plt.axis('off')
    ax = plt.gca() 
    if args.prompt_mode == 'point':
        plotter.draw_points(ax, prompt[..., :2], prompt[..., -1])
        plt.savefig(pointed_img_path, bbox_inches='tight', pad_inches=0)
    plotter.draw_mask(ax, alpha_pred)
    plt.savefig(matted_img_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    

if __name__ == '__main__':
    main()