import nibabel as nib
import os
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


class_dict = {
    "background":0,
    "aorta":1,
    "kidney_left":2,
    "liver":3,
    "postcava":4,
    "stomach":5,
    "gall_bladder":6,
    "kidney_right":7,
    "pancreas":8,
    "spleen":9
}

class_dict_BTCV = {
        0:(0, 0, 0),
        1:(255, 60, 0),
        2:(255, 60, 232),
        3:(134, 79, 117),
        4:(125, 0, 190),
        5:(117, 200, 191),
        6:(230, 91, 101),
        7:(255, 0, 155),
        8:(75, 205, 155),
        9:(100, 37, 200)
}

class_dict_ACDC = {
        0:(0, 0, 0),
        1:(230, 91, 101),
        2:(255, 0, 155),
        3:(117, 200, 91),
}

def onehot_to_rgb(onehot, color_dict=class_dict_BTCV):
    onehot = np.int64(onehot)
    output = np.zeros(onehot.shape[:2]+(3,))
    for k in color_dict.keys():
        output[onehot==k] = color_dict[k]
    return np.uint8(output)

files = os.listdir("/data2/onkar/atlasmini")

# input_nii = nib.load("/home/awd8324/onkar/Diff_SceneTextEraser/ct.nii.gz")


for f in tqdm(files):

    try:
        image = nib.load(f"/data2/onkar/atlasmini/{f}/ct.nii.gz").get_fdata()
        data_normalized = (image - np.min(image)) / (np.max(image) - np.min(image)) * 255
        data_normalized = data_normalized.astype(np.uint8)

        d = os.listdir(f"/data2/onkar/atlasmini/{f}/segmentations")
        coll = []
        label = np.concatenate([np.expand_dims(nib.load(f"/data2/onkar/atlasmini/{f}/segmentations/{k}").get_fdata().astype("uint8"), axis=3) for k in d], axis=3)
        label = label.argmax(-1).astype("uint8")

        if image.shape[-1] == label.shape[-1]:
            for i in range(image.shape[-1]):
                slice = data_normalized[:,:,i]
                slice = np.stack([slice]*3, axis=-1)
                l_slice = onehot_to_rgb(label[:,:,i])

                if len(np.unique(label[:,:,i])) > 1:
                    plt.imsave(f"/data2/onkar/altasSlices/images/{f}_{i}.png", slice)
                    plt.imsave(f"/data2/onkar/altasSlices/labels/{f}_{i}.png", l_slice)

    except:
        print(f)


            



# lab_data = nib.Nifti1Image(data, input_nii.affine, input_nii.header)
# nib.save(lab_data, "./label.nii.gz")
