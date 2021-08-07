from pdb import set_trace
import torch

import operator as op
import time
from functools import reduce
from random import shuffle

import re
import cv2
import os 
import pandas as pd
from collections import defaultdict
from os.path import join , abspath , exists
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import numpy as np
from itertools import combinations
from sklearn.model_selection import train_test_split
from typing import List , Tuple

from .dataset_constants import TRAIN , TEST , VAL, ROOT_DIR , IMAGE_DATASET_PKL
import torchvision.transforms as transforms
import pickle
from PIL import Image


def ncr(n : int , r : int) -> int :
    r = min(r, n-r)
    numer = reduce(op.mul, range(n, n-r, -1), 1)
    denom = reduce(op.mul, range(1, r+1), 1)
    return numer // denom  # or / in Python 2


def read_fmt(fp : str) -> List[str]:
    return_form = []
    with open(fp) as explorer:
        lines = [ line.strip() for line in explorer.readlines()] 
        return_form += lines
    return return_form

def read_img_gray(fp : str , grayscale=False, sizes=(224,224)) -> np.array:
    img = cv2.imread(fp)
    if grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # img = cv2.resize(img, sizes)
    return img

def read_img_pil(fp: str):
    img = Image.open(fp).convert("RGB")
    return img

class IncomeTaxImageDataSet(Dataset):
    def __init__(self, df : pd.DataFrame ,  indexs : List[Tuple[int]], transformer= None, fmt=False, debug=False, VGG=True) -> None:
        self.transform = transformer
        self.indexs = indexs
        
        self.fmt = df.fmt
        self.x = df.image
        self.y = df.label
        self.vgg = VGG

        self.debug = debug

    def __len__(self) -> int:
        return len(self.indexs)

    def __getitem__(self, index: int):
        start_time = time.time()    
        source, dest = self.indexs[index]

        if self.transform and self.vgg:
            image_src = self.transform(read_img_pil(self.x[source]))
            image_dest = self.transform(read_img_pil(self.x[dest]))
        else:
            image_src = torch.Tensor(read_img_gray(self.x[source]))
            image_dest = torch.Tensor(read_img_gray(self.x[dest] ))

        label = torch.Tensor([self.y[source] == self.y[dest]])
        if self.debug: print(f"reading the income {(time.time() - start_time) * 1000:.3f} ms")
        return image_src , image_dest , label.float()


def preprocessing_income_tax_return_1988(path : str):
    paths =os.listdir(abspath(path))
    
    dataset = {
        'fmt': [],
        'image': [],
        'page_num': []
    }

    for folder in tqdm(paths):
        current_folder = join(abspath(path), folder)
        
        if not os.path.isdir(join(current_folder)): 
            continue     
        
        for return_form in os.listdir(current_folder):
            

            current_document = os.listdir(join(current_folder, return_form ))
            filled_content = [ 
                join(
                    join(current_folder, return_form), fi) 
                for fi in current_document if fi.endswith("fmt") ]
            filled_image = [ 
                join(
                        join(current_folder, return_form), fi)
                for fi in current_document if fi.endswith("png") ]
            file_type = [ 
                int(re.sub(".png","",fi.split("_")[-1])) for fi in current_document if fi.endswith("png")
            ]
            dataset['fmt'] += filled_content
            dataset['image'] += filled_image
            dataset['page_num'] += file_type 
    return dataset


def generate_dataset(path=f"{ROOT_DIR}/sd02/data",read_img=False, clean=True):
    dataframe = defaultdict(lambda : [])
    dataset = preprocessing_income_tax_return_1988(path)
    for (img, fmt) in tqdm(zip(dataset['image'], dataset['fmt']), total=len(dataset['image'])):
        if read_img:
            dataframe['image'].append(read_img_gray(img, False))
            dataframe['image_gray'].append(read_img_gray(img))
        else:
            dataframe['image'].append(img)
            dataframe['fmt'].append(fmt)
    dataframe['label'] = dataset['page_num']
    
    return pd.DataFrame(dataframe) if not clean else data_cleaning(pd.DataFrame(dataframe))

def data_cleaning(df :pd.DataFrame ) -> pd.DataFrame:
    df = df.drop(df[df.label == 10].index) 
    df = df.drop(df[df.label == 9].index) 
    df = df.drop(df[df.label == 8].index) 
    df = df.reset_index()
    return df

def create_combination_income_dataset_with_split(df :pd.DataFrame , 
    random_shuffle=True ,
    sampling_size = [ 0.33, 0.1 ] ,
    VGG = True  ) \
    -> Tuple[ IncomeTaxImageDataSet , 
              IncomeTaxImageDataSet , 
              IncomeTaxImageDataSet  ]:
    assert len(sampling_size) == 2, "Sampling size must be size 2 providing test and validation size"
    
    indexs = list(combinations(df.index,2))
    
    if random_shuffle: shuffle(indexs)
    if exists(IMAGE_DATASET_PKL):
        print(f"found {IMAGE_DATASET_PKL} loading the pkl ...", end=' ')
        with open(IMAGE_DATASET_PKL, 'rb') as handle:
            res = pickle.load(handle)
            print("end")
            return res

    X_train , X_test  = train_test_split(indexs,   test_size=0.33)
    X_train , X_val  = train_test_split( X_train,  test_size=0.10)
    
    transform_pipeline = \
        transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    train_dataset = IncomeTaxImageDataSet(df, X_train, VGG=VGG, transformer=transform_pipeline)
    test_dataset =  IncomeTaxImageDataSet(df, X_test , VGG=VGG, transformer=transform_pipeline)
    val_dataset =   IncomeTaxImageDataSet(df, X_val  , VGG=VGG, transformer=transform_pipeline)
    
    res = train_dataset, test_dataset, val_dataset
    
    with open(IMAGE_DATASET_PKL, 'wb') as handle:
        pickle.dump(res, handle)
    return res


def create_combination_dataloader(df: pd.DataFrame, batch_size=32, num_workers=1, sampling_size=[0.33, 0.1], VGG=True) \
    -> dict:
    
    combination_bundles = create_combination_income_dataset_with_split(
        df, True, sampling_size, VGG)


    print("generating all of the dataloaders ...  ",end=' ')
    dataloaders = { associate_tag: DataLoader( 
            dataset, 
            batch_size=batch_size, 
            num_workers=num_workers, 
            shuffle=False) 
        for associate_tag ,dataset in zip([TRAIN, TEST, VAL] , combination_bundles )}
    print("done")
    return dataloaders
    