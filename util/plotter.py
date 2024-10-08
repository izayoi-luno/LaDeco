import matplotlib.pyplot as plt
import numpy as np
import copy

def show_seg_result(result):
    if len(result) == 0:
        return
    sorted_result = sorted(result, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_result[0]['segmentation'].shape[0], sorted_result[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for res in sorted_result:
        m = res['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def draw_points(ax, points, labels, size=375):
    if points is None or labels is None:
        return
    if isinstance(points, list) and isinstance(labels, list):
        if len(points) == 0 or len(labels) == 0:
            return
        if len(points) != len(labels):
            raise ValueError("points and labels should have the same length")
        points = np.array(points)
        labels = np.array(labels)
    elif isinstance(points, np.ndarray) and isinstance(labels, np.ndarray):
        pass
    else:
        raise ValueError("points and labels should be a list of points or a numpy array of points")
    
    for i in range(points.shape[0]):
        if labels[i] == 1:
            ax.scatter(points[i, 0], points[i, 1], color='blue', s=size, marker='*', edgecolor='white', linewidth=1.25)
        else:
            ax.scatter(points[i, 0], points[i, 1], color='red', s=size, marker='*', edgecolor='white', linewidth=1.25)

def draw_mask(ax, mask):
    if not isinstance(mask, np.ndarray):
        raise ValueError("masks should be a numpy array")
    if len(mask.shape) > 2:
        mask = np.squeeze(mask)
    mask = mask.astype('uint8')
    mask = mask / 255 if mask.max() > 1 else mask
    mask_show = copy.deepcopy(mask)
    mask_color = np.concatenate([np.random.random(3), [0.3]])
    mask_show = mask_show.reshape(mask_show.shape[0], mask_show.shape[1], 1) * mask_color.reshape(1, 1, -1)
    ax.imshow(mask_show)
    
    