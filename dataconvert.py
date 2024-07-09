import numpy as np
import glob
import matplotlib.pyplot as plt
import nibabel as nib


files = glob.glob("/data2/onkar/BTCV/BTCV/data/BTCV/train_npz/*.npz")
print(len(files))
data = np.load(files[23])

d = nib.load("/data2/onkar/atlasmini/BDMAP_00000001/ct.nii.gz").get_fdata()

print(d.shape)




# print(data['image'].shape, data['label'].shape)