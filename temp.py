import pickle as pkl
import cv2
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import trange

ADE_20K_PATH = Path("../datasets/")
with open(ADE_20K_PATH / "ADE20K_2021_17_01/index_ade20k.pkl", "rb") as f:
    index_ade20k = pkl.load(f)
num_data = len(index_ade20k["filename"])
for i in trange(num_data):
    full_path = ADE_20K_PATH / index_ade20k["folder"][i] / index_ade20k["filename"][i]
    image = cv2.imread(str(full_path))
    fileseg = str(full_path).replace('.jpg', '_seg.png')
    with Image.open(fileseg) as io:
        seg = np.array(io)
    ObjectClassMasks = (seg[:,:,0]/10).astype(np.int32)*256+(seg[:,:,1].astype(np.int32))
a=5