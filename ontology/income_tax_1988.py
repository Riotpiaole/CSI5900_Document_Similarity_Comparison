import re
import pandas as pd
from collections import defaultdict
import cv2
from os import listdir 
from os.path import join , abspath
from tqdm import tqdm
from pdb import set_trace

def preprocessing_income_tax_return_1988(path):
    paths =listdir(abspath(path))
    
    dataset = {
        'fmt': [],
        'image': [],
    }

    for folder in tqdm(paths):
        current_folder = join(abspath(path), folder)
        for return_form in listdir(current_folder):
            current_document = listdir(join(current_folder, return_form ))
            set_trace()
            filled_content = [ 
                join(
                    join(current_folder, return_form), fi) 
                for fi in current_document if fi.endswith("fmt") ]
            filled_image = [ 
                join(
                        join(current_folder, return_form), fi)
                for fi in current_document if fi.endswith("png") ]
            file_type = [ 
                re.sub(".png","",fi.split("_")[-1]) for fi in current_document if fi.endswith("png")
            ]
            dataset['fmt'] += filled_content
            dataset['image'] += filled_image
            dataset['page_num'] += file_type 
    return dataset

def read_fmt(fp):
    return_form = []
    with open(fp) as explorer:
        lines = [ line.strip() for line in explorer.readlines()] 
        return_form += lines
    return return_form

def read_img_gray(fp, grayscale=True):
    img = cv2.imread(fp)
    if grayscale: img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def generate_dataset(path="../sd02/data",read_img=False):
    dataframe = defaultdict(lambda : [])
    dataset = preprocessing_income_tax_return_1988(path)
    for (img, fmt) in tqdm(zip(dataset['image'], dataset['fmt'])):
        if read_img:
            dataframe['image'].append(read_img_gray(img, False))
            dataframe['image_gray'].append(read_img_gray(img))
        else:
            dataframe['image'].append(img)
        
    return pd.DataFrame(dataframe)

if __name__ == "__main__":
    # dataset = preprocessing_income_tax_return_1988("../sd02/data")
    # test_fmt = read_fmt(dataset['fmt'][0])
    # test_img =read_img_gray(dataset['image'][0])
    df = generate_dataset() 
    for img in df.image:
        print(img)
