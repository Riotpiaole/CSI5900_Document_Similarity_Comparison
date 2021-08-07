from os.path import abspath , basename
from os import getcwd

if basename(getcwd()) == 'dataset': 
    ROOT_DIR = abspath("..")
else:
    ROOT_DIR = abspath('.')
TRAIN = 'train'
TEST = 'test'
VAL = 'val'
IMAGE_DATASET_PKL = f"{ROOT_DIR}/dataset/income_tax_1988_image.pkl"
OCR_EXTRACTED_TEXT_PKL=f'{ROOT_DIR}/dataset/income_tax_1988_text.pkl'
OCR_EXTRACTED_LAYOUT_OBJ_PKL=f"{ROOT_DIR}/dataset/income_tax_1988_layout_ocr.pkl"

def default_array():
    return []

print("")