import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src.segment import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
from util import rs, plotter

def segment_alone(img, ckpt_path, model_type, seg_infer_type, prompt=None, multimask=False, device='cuda'):
    assert seg_infer_type == 'auto' or 'manual', ValueError("Invalid seg_infer_type")
    
    model = sam_model_registry[model_type](checkpoint=ckpt_path)
    model.to(device=device)
    
    if seg_infer_type == 'auto':
        mask_generator = SamAutomaticMaskGenerator(
            model=model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=0,
        )
        masks = mask_generator.generate(img)
    else:
        if not prompt or (isinstance(prompt, np.ndarray) and len(prompt) == 0):
            print("[User Warning]: No prompt provided, using default prompt")
            h, w = img.shape[:2]
            prompt = np.array([[w // 2, h // 2, 1]])
        point_coords = prompt[..., :2]
        point_labels = prompt[..., -1]
        mask_generator = SamPredictor(model)
        mask_generator.set_image(img)
        masks, _, _ = mask_generator.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )

    return masks

def segment(img, model, prompt=None, multimask=False):
    if isinstance(model, SamAutomaticMaskGenerator):
        return model.generate(img)
    elif isinstance(model, SamPredictor):
        if not prompt or (isinstance(prompt, np.ndarray) and len(prompt) == 0):
            print("[User Warning]: No prompt provided, using default prompt")
            h, w = img.shape[:2]
            prompt = np.array([[w // 2, h // 2, 1]])
        point_coords = prompt[..., :2]
        point_labels = prompt[..., -1]
        model.set_image(img)
        masks, _, _ = model.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=multimask,
        )
        return masks    

def build_segmenter(seg_infer_type, ckpt_path, model_type, device='cuda'):
    assert seg_infer_type == 'auto' or 'manual', ValueError("Invalid seg_infer_type")
    
    if seg_infer_type == 'auto':
        model = sam_model_registry[model_type](checkpoint=ckpt_path)
        model.to(device=device)
        return SamAutomaticMaskGenerator(
            model=model,
            points_per_side=32,
            pred_iou_thresh=0.88,
            stability_score_thresh=0.95,
            crop_n_layers=0,
            crop_n_points_downscale_factor=1,
            min_mask_region_area=0,
        )
    else:
        model = sam_model_registry[model_type](checkpoint=ckpt_path)
        model.to(device=device)
        return SamPredictor(model)
    
def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-img_path', '--img_path', type=str, required=True, default=None, help='Path to the image to be processed')
    parser.add_argument('-out_path', '--out_path', type=str, default='results', help='Path to the output results')
    parser.add_argument('-seg_type', '--seg_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='Segment Anything Model type')
    parser.add_argument('-seg_ckpt', '--seg_ckpt', type=str, default='./pretrained/sam_vit_h_4b8939.pth', help='SAM checkpoint')
    parser.add_argument('-seg_infer_type', '--seg_infer_type', type=str, default='auto', choices=['auto', 'manual'], help='Segment inferencer type')
    parser.add_argument('-prompt', '--prompt', type=float, nargs='+', default=None, help='Prompt for manual inference')    
    parser.add_argument('-multimask', '--multimask', action='store_true', type=bool, default=False, help='Whether to use multimask output')
    
    return parser.parse_args()

def main():
    args = args_parse()
    img = rs.read_img(args.img_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    prompt = []
    if args.prompt:
        if len(args.prompt) % 3 == 0:
            for i in range(0, len(args.prompt), 3):
                prompt.append(args.prompt[i : i + 3])
        else:
            raise ValueError("prompt should be a list of 3-element list")
    prompt = np.array(prompt)
    masks = segment_alone(img, args.seg_ckpt, args.seg_type, args.seg_infer_type, prompt, args.multimask, device=device)
    
    if args.seg_infer_type == 'auto':
        plt.figure()
        plotter.show_seg_result(masks)
        plt.axis('off')
        plt.savefig(masked_img_path)
    else:
        for i, mask in enumerate(masks):
            out_path = os.path.join(args.out_path, f'matte_result_{i}.png')
            if args.seg_infer_type == 'manual':
                pointed_img_path = os.path.join(args.out_path, 'pointed_img.png')
            masked_img_path = os.path.join(args.out_path, f'masked_img_{i}.png')
            
            rs.save_matte(mask, out_path)
        
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            ax = plt.gca()
            plotter.draw_points(ax, prompt[..., :2], prompt[..., -1])
            plt.savefig(pointed_img_path)
            plotter.draw_mask(ax, mask)
            plt.savefig(masked_img_path)
            plt.close()
    
if __name__ == '__main__':
    main()
    
    