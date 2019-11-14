import torch
import os
import shutil
def prepare_val_folder(folder, wnids):
    img_files = sorted([os.path.join(folder, file) for file in os.listdir(folder)])

    for wnid in set(wnids):
        os.mkdir(os.path.join(folder, wnid))

    for wnid, img_file in zip(wnids, img_files):
        shutil.move(img_file, os.path.join(folder, wnid, os.path.basename(img_file)))
root='/home/xy/pan/data/'
prepare_val_folder(os.path.join(root,'val'),torch.load(os.path.join(root, 'meta.bin'))[1])