import os
import pandas as pd
import numpy as np
import shutil

data_root='/home/xy/projects/data.dog/'
TRAIN_IMG_PATH = "train"
TEST_IMG_PATH = "val"
LABELS_CSV_PATH = "labels.csv"
IMAGE_FOLDER='imagefolder'
def prpare_imagefolder_from_dframe(dframe,set):
    i=0
    for index, row in dframe.iterrows():
        i+=1
        print(i)
        img_path=os.path.join(data_root,TRAIN_IMG_PATH,row['id']+'.jpg')
        classname=row['breed']
        class_folder=os.path.join(data_root,IMAGE_FOLDER,set,classname)
        if os.path.exists(class_folder):
            shutil.move(img_path,class_folder)
        else:
            os.mkdir(class_folder)
            shutil.move(img_path, class_folder)
dframe = pd.read_csv(os.path.join(data_root, LABELS_CSV_PATH))
cut=int(len(dframe)*0.8)
train,val = np.split(dframe,[cut],axis=0)
prpare_imagefolder_from_dframe(train,'train')
prpare_imagefolder_from_dframe(val,'val')
