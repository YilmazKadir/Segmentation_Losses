import os
import numpy as np
import h5py

data_dir = "./data/Synapse/test_vol_h5/"
save_path = "./data/Synapse/validation_npz/"
for volume in os.listdir(data_dir):
    data = h5py.File(data_dir + volume)
    image, label = data['image'][:], data['label'][:]
    num_images = image.shape[0]
    for i in range(num_images):
        save_filename = save_path + f"{volume[:8]}_slice{i:03}"
        np.savez(save_filename, image=image[i, :, :], label=label[i, :, :])
