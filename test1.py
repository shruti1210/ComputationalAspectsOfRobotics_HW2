import os 
import numpy as np

dataset_dir = 'hw2\dataset\li'
idx = 1
rgb_img_path = os.path.join(dataset_dir, 'rgb', f'{idx}_rgb.jpeg')

print(rgb_img_path)

def gen_obj_depth(obj_id, depth, mask):
    """
    In:
        obj_id: int, indicating an object in LIST_OBJ_FOLDERNAME.
        depth: Numpy array [height, width], where each value is z depth in meters.
        mask: Numpy array [height, width], where each value is an obj_id.
    Out:
        obj_depth: Numpy array [height, width] of float64, where depth value of all the pixels that don't belong to the object is 0.
    Purpose:
        Generate depth image for a specific object given obj_id.
        Generate depth for all objects when obj_id == -1. You should filter out the depth of the background, where the ID is 0 in the mask. We want to preserve depth only for object 1 to 5 inclusive.
    """
    # TODO
    
    if obj_id == -1:
        obj_ids = list(range(1, 6))
    else:
        obj_ids = [obj_id]
    
    obj_depth = np.zeros_like(depth, dtype=np.float64)
    
    for obj_id in obj_ids:
        obj_mask = np.where(mask == obj_id, 1, 0)
        obj_depth += depth * obj_mask
    #generate a depth image that only contains a specific object given by a object id 

    #obj_depth = None
    return obj_depth
