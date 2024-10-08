import cv2
import numpy as np
def mask_erode(mask, factor=15):
    if isinstance(mask, np.ndarray):
        mask = mask.astype('uint8')
        mask = cv2.erode(mask, np.ones((factor, factor), np.uint8))
        return mask
    else:
        raise TypeError("mask should be numpy array")

def mask_dilate(mask, factor=15):
    if isinstance(mask, np.ndarray):
        mask = mask.astype('uint8')
        mask = cv2.dilate(mask, np.ones((factor, factor), np.uint8))
        return mask
    else:
        raise TypeError("mask should be numpy array")

def mask_open(mask, factor_e=15, factor_d=15):
    return mask_erode(mask_dilate(mask, factor_d), factor_e)

def mask_close(mask, factor_e=15, factor_d=15):
    return mask_dilate(mask_erode(mask, factor_e), factor_d)