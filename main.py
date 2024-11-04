import argparse
import torch
import os
import matplotlib.pyplot as plt
import numpy as np

from src.decomposer import LayerDecomposer
from util import plotter, rs
from aug import manu

def args_parse():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-img_path', '--img_path', type=str, required=True, default=None, help='Path to the image to be processed')
    parser.add_argument('-out_path', '--out_path', type=str, default='./results', help='Path to the output results')

    parser.add_argument('-infer_type', '--infer_type', type=str, default='photo', choices=['photo', 'video', 'photo_set'], help='inferencer type')

    parser.add_argument('-front_side', '--front_side', type=str, default='matte', choices=['matte', 'segment'], help='front side of the decomposition')
    parser.add_argument('-back_side', '--back_side', type=str, default='inpaint', help='back side of the decomposition')
    parser.add_argument('-input_mode', '--input_mode', type=str, default='keyboard', choices=['keyboard', 'mouse'], help='input mode')
    parser.add_argument('-prompt_mode', '--prompt_mode', type=str, default='point', choices=['box', 'point', 'text'])
    parser.add_argument('-prompt', '--prompt', type=int, nargs='+', default=None, help='The coordinate of the point prompt, [coord_W coord_H].')
    
    parser.add_argument('-seg_type', '--seg_type', type=str, default='vit_h', choices=['vit_h', 'vit_l', 'vit_b'], help='Segment Anything Model type')
    parser.add_argument('-seg_ckpt', '--seg_ckpt', type=str, default='./pretrained/sam_vit_h_4b8939.pth', help='SAM checkpoint')
    parser.add_argument('-seg_infer_type', '--seg_infer_type', type=str, default='manual', choices=['auto', 'manual'], help='Segment inferencer type')   
    parser.add_argument('-multimask', '--multimask', action='store_true', default=False, help='Whether to use multimask output')
    
    parser.add_argument('-mat_config', '--mat_config', type=str, default='./configs/mam_configs/MAM-ViTB-8gpu.toml', help='matte model configurations')
    parser.add_argument('-mat_ckpt', '--mat_ckpt', type=str, default='./pretrained/mam_vitb.pth', help='MAM checkpoint')
    parser.add_argument('-osn_width', '--osn_width', type=int, nargs='+', default=[10, 20, 10], help='guidance threshold')
    parser.add_argument('-twoside', '--twoside', action='store_true', default=False, help='post process with twoside of the guidance')        
    parser.add_argument('-sam', '--sam', action='store_true', default=False, help='return mask')    
    parser.add_argument('-maskguide', '--maskguide', action='store_true', default=False, help='mask guidance')    
    parser.add_argument('-postprocess', '--postprocess', action='store_true', default=False, help='postprocess to remove bg')    
    parser.add_argument('-target_length', '--target_length', type=int, default=1024, help='target length of input image')
    
    parser.add_argument('-inp_config', '--inp_config', type=str, default='configs/lama_configs/predict/predict.yaml', help='lama config path')
    parser.add_argument('-inp_ckpt', '--inp_ckpt', type=str, default='./pretrained/big-lama/', help='lama checkpoint')
    parser.add_argument('-mod', '--mod', type=int, default=8, help='inpainting modulo')
    parser.add_argument('-device', '--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='device')
    
    return parser.parse_args()

def photo_infer(args):
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    decomposer = LayerDecomposer(args.front_side, args.prompt_mode, args.back_side, \
                                 args.mod, device, args.multimask, args.seg_infer_type)
    
    if args.front_side == 'segment':
        decomposer.load_seg_model(args.seg_ckpt, args.seg_type, args.seg_infer_type)
    elif args.front_side == 'matte':
        decomposer.load_mat_model(args.mat_config, args.mat_ckpt)
    else:
        raise NotImplementedError("[Error]: front side not supported")
    
    if args.back_side == 'inpaint':
        decomposer.load_inp_model(args.inp_config, args.inp_ckpt)
    else:
        raise NotImplementedError("[Error]: back side not supported")
    
    img = rs.read_img(args.img_path)
    if args.input_mode == 'mouse':
        handler = manu.ImageClickHandler(args.img_path)
        handler.load_image()
        got_prompt = handler.get_clicked_point()
        prompt = [item for sublist in got_prompt for item in sublist]
        # print(prompt)
    elif args.input_mode == 'keyboard':
        if args.prompt:
            prompt = args.prompt
            # print(prompt)
        else:
            raise ValueError("[Error]: prompt should be provided")
    
    process_input = decomposer.prompt_preprocess(img, prompt, args.target_length)
    if args.front_side == 'segment':
        masks = decomposer.segment(process_input['img'], process_input['prompt'])
        if isinstance(masks, list):
            masked_img_path = os.path.join(args.out_path, f'masked_img.png')
            plt.figure()
            plt.imshow(img)
            plt.axis('off')
            ax = plt.gca()
            plotter.show_seg_result(masks, ax)
            plt.savefig(masked_img_path, bbox_inches='tight', pad_inches=0)
            plt.close()

            mask = [decomposer.inter_process(m) for m in masks]

        elif isinstance(masks, np.ndarray):
            masks = masks.astype(np.uint8) * 255
            for i, m in enumerate(masks):
                out_path = os.path.join(args.out_path, f'mask_result_{i}.png')
                if args.seg_infer_type == 'manual':
                    pointed_img_path = os.path.join(args.out_path, f'pointed_img_{i}.png')
                masked_img_path = os.path.join(args.out_path, f'masked_img_{i}.png')
                foreground_path = os.path.join(args.out_path, f'foreground_{i}.png')
                rs.save_mask(m, out_path)
                rs.save_foreground(img, m, foreground_path)
            
                plt.figure()
                plt.imshow(img)
                plt.axis('off')
                ax = plt.gca()
                if args.prompt_mode == 'point':
                    plotter.draw_points(ax, process_input['prompt'][..., :2], process_input['prompt'][..., -1])
                    plt.savefig(pointed_img_path, bbox_inches='tight', pad_inches=0)
                plotter.draw_mask(ax, m)
                plt.savefig(masked_img_path, bbox_inches='tight', pad_inches=0)
                plt.close()

            mask = [decomposer.inter_process(m) for m in masks]
       
    elif args.front_side == 'matte':
        osn_width = args.osn_width
        if not len(osn_width) == 3:
            print("[User Warning]: osn_width should be a list of 3 integers, using default values instead.")
            osn_width = [10, 20, 10]
        matte = decomposer.matte(process_input['mam_input'], args.maskguide, args.twoside, \
                                 args.postprocess, osn_width)
        out_path = os.path.join(args.out_path, 'matte_result.png')
        if args.prompt_mode == 'point':
            pointed_img_path = os.path.join(args.out_path, 'pointed_img.png')
        matted_img_path = os.path.join(args.out_path, 'matted_img.png')
        foreground_path = os.path.join(args.out_path, 'foreground.png')
        rs.save_matte(matte, out_path)
        rs.save_foreground(img, matte, foreground_path)
        
        plt.figure()
        plt.imshow(img)
        plt.axis('off')
        ax = plt.gca() 
        if args.prompt_mode == 'point':
            plotter.draw_points(ax, process_input['prompt'][..., :2], process_input['prompt'][..., -1])
            plt.savefig(pointed_img_path, bbox_inches='tight', pad_inches=0)
        plotter.draw_mask(ax, matte)
        plt.savefig(matted_img_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        
        mask = decomposer.inter_process(matte)
    else:
        raise NotImplementedError("[Error]: front side not supported")
    
    if args.back_side == 'inpaint':
        if isinstance(mask, list):
            for i, m in enumerate(mask):
                inpainted = decomposer.bg_inpaint(img, mask=m)
                out_path = os.path.join(args.out_path, f'inpaint_result_{i}.png')
                rs.save_img(inpainted, out_path)
        elif isinstance(mask, np.ndarray):
            inpainted = decomposer.bg_inpaint(img, mask=mask)
            out_path = os.path.join(args.out_path, 'inpaint_result.png')
            rs.save_img(inpainted, out_path)
    else:
        raise NotImplementedError("[Error]: back side not supported")
def main():
    args = args_parse()
    if args.infer_type == 'photo':
        photo_infer(args)
    
if __name__ == '__main__':
    main()