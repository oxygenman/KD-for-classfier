from __future__ import print_function, division
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils, models

from PIL import Image
import os
TRAIN_IMG_PATH = "dog/train"
TEST_IMG_PATH = "dog/test"
LABELS_CSV_PATH = "dog/labels.csv"
SAMPLE_SUB_PATH = "dog/sample_submission.csv"

class DogsDataset(Dataset):
    """Dog breed identification dataset."""
    def __init__(self, root, train=False,transform=None,download=False):
        """
        Args:
            root (string): Directory of all dataset.
            train(bool):Is the train procedure
            transform (callable, optional): Optional transform to be applied
                on a sample.
            download(bool): Wheather to download the dataset
        """
        if download:
            print('Warning:Download module is on the way, this code just support the local file!')
        else:
            self.root = root
            self.transform=transform
            self.train=train
            dframe = pd.read_csv(os.path.join(self.root,LABELS_CSV_PATH))
            class_names = pd.read_csv(os.path.join(self.root,SAMPLE_SUB_PATH)).keys()[1:]
            idxs = range(len(class_names))
            class_to_idx = dict(zip(class_names, idxs))
            dframe['target'] = [class_to_idx[x] for x in dframe.breed]
            cut = int(len(dframe) * 0.8)
            train, test = np.split(dframe, [cut], axis=0)
            test = test.reset_index(drop=True)
            if self.train:
                self.labels_frame = train
            else:
                self.labels_frame = test

    def __len__(self):
        return len(self.labels_frame)

    def __getitem__(self, idx):
        img_name = os.path.join(os.path.join(self.root,TRAIN_IMG_PATH), self.labels_frame.id[idx]) + ".jpg"
        image = Image.open(img_name)
        label = self.labels_frame.target[idx]

        if self.transform:
            image = self.transform(image)

        return [image, label]






