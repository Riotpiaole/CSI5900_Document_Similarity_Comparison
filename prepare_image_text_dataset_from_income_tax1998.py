from tqdm.utils import Comparable
from dataset.income_tax_1988_image import generate_dataset 
from dataset.income_tax_1988_text import  IncomeTaxTextDataSet
from utils import  multi_processing_layout_text_ocr
from itertools import combinations

if __name__ == "__main__":
    from dataset.preprocessing.preprocessing_constants import (TESSACT_PATH) 
    from dataset.dataset_constants import OCR_EXTRACTED_LAYOUT_OBJ_PKL
    from dataset.layout_parser import  OCRLayoutExtractor
    df = generate_dataset()
    # image_path = df.image[0]
    # custom_oem_psm_config = r"--oem 2"
    import pickle
    with open(OCR_EXTRACTED_LAYOUT_OBJ_PKL, 'rb') as handler:
        dataset = pickle.load(handler)['ocr_obj']
    # outs = multi_processing_layout_text_ocr(df.image)
    # indexs = list(combinations(df.index,2))
    
    # ocr_i1_res= OCRLayoutExtractor(df.image[0], 0, True)
    
    # shuffle(indexs)
    # dataset = IncomeTaxTextDataSet(outs['tokens'], df.label, indexs, filter_fmt=df.fmt)
    # loader = DataLoader(dataset, 32, False)
    
    # Demo Term Frequency Index for converting the document  