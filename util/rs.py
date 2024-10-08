# Read and Save Util:
# Read input images
# Save output images, layers, and intermediate results like masks and mattes
# Convert different types of inputs and outputs

import cv2
import os
import numpy as np

def read_img(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    img = cv2.imread(img_path)
    if len(img.shape) == 3 and img.shape[2] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  
    return img

def read_mask(mask_path):
    if not os.path.exists(mask_path):
        raise FileNotFoundError(mask_path)
    mask = cv2.imread(mask_path)
    if len(mask.shape) > 2: 
        if mask.shape[2] > 1:
            print("[User Warning]: May not be a mask or matte, might be a color image.")
            print("[User Warning]: Try to convert it to a mask or matte.")
            mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    return mask

def read_matte(matte_path):
    if not os.path.exists(matte_path):
        raise FileNotFoundError(matte_path)
    return read_mask(matte_path)

def read_img_alpha(img_path):
    if not os.path.exists(img_path):
        raise FileNotFoundError(img_path)
    img = read_img(img_path, cv2.IMREAD_UNCHANGED)
    alpha = img[:, :, 3] if img.shape[2] == 4 else None
    img = img[:, :, :3] if img.shape[2] == 4 else img
    if not alpha:
        print("[User Warning]: No alpha channel found in the image.")   
    return img, alpha

def save_img(img, save_path):
    if isinstance(img, np.ndarray):
        img = img.astype('uint8')
        if len(img.shape) == 3:
            if img.shape[2] == 4:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2RGBA)
            elif img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        cv2.imwrite(save_path, img)
    else:
        try:
            cv2.imwrite(save_path, img)
        except cv2.error as e:
            print(e)

def save_matte(alpha, save_path):
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
    if isinstance(alpha, np.ndarray):
        alpha = alpha.astype('uint8')
        if len(alpha.shape) == 2 or (len(alpha.shape) == 3 and alpha.shape[2] == 1):
            cv2.imwrite(save_path, alpha)
        else:
            print("[User Warning]: alpha should be 2D or 3D array with 1 channel")
            cv2.cvtColor(alpha, cv2.COLOR_BGR2GRAY)
            cv2.imwrite(save_path, alpha)
    else:
        raise TypeError('matte or mask should be numpy array')

def save_mask(mask, save_path):
    save_matte(mask, save_path)
    
def matte2mask(alpha, thres=127):
    if isinstance(alpha, np.ndarray):
        if len(alpha.shape) == 3 and alpha.shape[2] == 1:
            alpha = np.squeeze(alpha)
            assert len(alpha.shape) == 2
        if len(alpha.shape) == 2:
            if np.max(alpha) == 1:
                alpha *= 255
            alpha[alpha > thres] = 255
            alpha[alpha <= thres] = 0
            return alpha
        if len(alpha.shape) > 2:
            raise TypeError('alpha should be 2D or 3D array with 1 channel')
    else:
        raise TypeError('alpha should be numpy array')
        