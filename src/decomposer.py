import sys
import os
import yaml
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from src import segmentV1, matteV1, inpaintV1
from aug import manu, morph
from util import rs

class LayerDecomposer:
    def __init__(
            self, 
            front_side = 'matte',
            prompt_mode = 'point',
            back_side = 'inpaint',
            mod = 8,
            device = 'cuda',
            multimask = False,
            seg_infer_type = 'manual'
            ) -> None:
        self.seg_model = None
        self.mat_model = None
        self.inp_model = None
        self.front_side = front_side
        self.prompt_mode = prompt_mode
        self.back_side = back_side
        self.mod = mod
        self.device = device
        self.multimask = multimask
        self.seg_infer_type = seg_infer_type
        self.dilate_factor = 15
    
    def load_seg_model(self, seg_path, seg_type='vit_h', seg_infer_type='manual'):
        if not os.path.exists(seg_path):
            raise FileNotFoundError("segmentation model weight file not found")
        self.seg_model = segmentV1.build_segmenter(seg_infer_type, seg_path, seg_type, self.device)
    
    def load_mat_model(self, mat_config, mat_ckpt):
        if not os.path.exists(mat_config):
            raise FileNotFoundError("matting model configs file not found")
        if not os.path.exists(mat_ckpt):
            raise FileNotFoundError("matting model weight file not found")
        self.mat_model = matteV1.build_matter(mat_config, mat_ckpt)
    
    def load_inp_model(self, inp_config, inp_ckpt):
        if not os.path.exists(inp_config):
            raise FileNotFoundError("inpainting model configs file not found")
        if not os.path.exists(inp_ckpt):
            raise FileNotFoundError("inpainting model weight file not found")
        self.inp_model = inpaintV1.build_inpainter(inp_config, inp_ckpt, self.device)
    
    def prompt_preprocess(self, img, prompt, target_length=1024):
        input = None
        p = []
        
        if self.front_side == 'matte':
            if self.prompt_mode == 'point':
                if len(prompt) % 3 == 0:
                    for i in range(0, len(prompt), 3):
                        p.append(prompt[i : i + 3])
                else:
                    raise ValueError("prompt should be a list of 3-element list")
            elif self.prompt_mode == 'bbox':
                pass
            elif self.prompt_mode == 'text':
                pass
            
            p = np.array(p)
            mam_input = matteV1.preprocess(img, target_length, self.prompt_mode, p)
            input = {'mam_input': mam_input, 'prompt': p}
            
        elif self.front_side == 'segment':
            if self.prompt_mode == 'point':
                if len(prompt) % 3 == 0:
                    for i in range(0, len(prompt), 3):
                        p.append(prompt[i : i + 3])
                else:
                    raise ValueError("prompt should be a list of 3-element list")
            else:
                raise ValueError("prompt mode {} not supported yet".format(self.prompt_mode))
            
            p = np.array(p)
            input = {'img': img, 'prompt': p}
            
        else:
            raise ValueError("front side should be 'matte' or 'segment'")
        
        return input
    
    def inter_process(self, mask):
        if self.front_side == 'matte':
            mask = rs.matte2mask(mask, thres=0)
        return morph.mask_dilate(mask, factor=self.dilate_factor)
    
    def segment(self, img, prompt=None):
        if not self.seg_model:
            print("[User Warning]: segmentation model not loaded, try to load it first")
        return segmentV1.segment(img, self.seg_model, prompt, self.multimask)

    def matte(self, input, maskguide=False, twoside=False, postprocess=False, osn_width=[10, 20, 10]):
        if not self.mat_model:
            print("[User Warning]: matting model not loaded, try to load it first")
            return
        if not isinstance(input, dict):
            print("[User Warning]: input should be a dict")
        return matteV1.matte(input, self.mat_model, maskguide, twoside, postprocess, \
                       osn_width[0], osn_width[1], osn_width[2])

    def bg_inpaint(self, img, mask=None):
        if self.front_side == 'matte':
            mask = rs.matte2mask(mask)
        if not self.inp_model:
            print("[User Warning]: inpainting model not loaded, try to load it first")
            
        batch, unpad_to_size = inpaintV1.preprocess(img, mask, self.mod, self.device)
        return inpaintV1.inpaint(self.inp_model, batch, unpad_to_size)
        
    def obj_inpaint(self, img, mask=None, manual=False):
        if manual:
            mask = manu
        if not mask:
            print("[User Warning]: mask is not provided")
            return
        if not self.inp_model:
            print("[User Warning]: inpainting model not loaded, try to load it first")
            
        batch, unpad_to_size = inpaintV1.preprocess(img, mask, self.mod, self.device)
        return inpaintV1.inpaint(img, self.inp_model, mask, batch, unpad_to_size)

    