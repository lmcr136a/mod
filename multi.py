import os
import shutil
import numpy as np
from random import sample

pth = 'data/imagenet/train/'
# for i in range(1000):
#     file_list = [name for name in os.listdir(pth+str(i))]
#     for name in file_list:
#         os.rename(pth+f"{i}/"+name, 'data/imagenet/train/'+f"{i}/"+name)

for class_idx in range(1000):
    cls_dir = pth+str(class_idx)+"/"
    new_class_dir = f"data/imagenet/val/{class_idx}/"
    if not os.path.exists(new_class_dir):
        os.makedirs(new_class_dir)

    names = [name for name in os.listdir(cls_dir)]
    names_to_move = sample(names, 100)
    for filename in names_to_move:
        shutil.move(cls_dir+filename, new_class_dir+filename)