# ------------------------------------------------------------------------
# Modified from MGMatting (https://github.com/yucornetto/MGMatting)
# ------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from . import m2ms
from segment.segment_anything import sam_model_registry

class sam_m2m(nn.Module):
    def __init__(self, seg, m2m):
        super(sam_m2m, self).__init__()
        if m2m not in m2ms.__all__:
            raise NotImplementedError("Unknown M2M {}".format(m2m))
        self.m2m = m2ms.__dict__[m2m](nc=256)
        if seg == 'sam_vit_b':
            self.seg_model = sam_model_registry['vit_b'](checkpoint='pretrained/sam_vit_b_01ec64.pth')
        elif seg == 'sam_vit_l':
            self.seg_model = sam_model_registry['vit_l'](checkpoint='pretrained/sam_vit_l_0b3195.pth')
        elif seg == 'sam_vit_h':
            self.seg_model = sam_model_registry['vit_h'](checkpoint='pretrained/sam_vit_h_4b8939.pth')
        self.seg_model.eval()

    def forward(self, image, guidance):
        self.seg_model.eval()
        with torch.no_grad():
            feas, masks = self.seg_model.forward_m2m(image, guidance, multimask_output=True)
        pred = self.m2m(feas, image, masks)
        return pred

    def forward_inference(self, image_dict):
        self.seg_model.eval()
        with torch.no_grad():
            feas, masks, post_masks = self.seg_model.forward_m2m_inference(image_dict, multimask_output=True)
        pred = self.m2m(feas, image_dict["image"], masks)
        return feas, pred, post_masks

def get_generator_m2m(seg, m2m):
    if 'sam' in seg:
        generator = sam_m2m(seg=seg, m2m=m2m)
    return generator