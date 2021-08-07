import layoutparser as lp
try:
    from dataset.preprocessing.nlp import preprocessing
except  ModuleNotFoundError:
    from .preprocessing.nlp import preprocessing
import os
import cv2 as cv
import time

class OCRLayoutExtractor(object):
    def __init__(self, image_path, index , rgb=False , tesseract_path="D:\\tesseract-ocr\\tesseract") -> None:
        if not os.path.exists(tesseract_path ):
            raise FileNotFoundError(f"{tesseract_path} not exists tesseract path")
        self.agent = lp.TesseractAgent()
        try:
            self.agent = self.agent.with_tesseract_executable(tesseract_path)
        except:
            raise FileNotFoundError(f"{tesseract_path} not a tesseract executable please download from https://tesseract-ocr.github.io/tessdoc/Downloads.html ")
        self.img_path = image_path
        image = cv.imread(image_path)

        self.data = image if rgb else cv.cvtColor(image, cv.COLOR_BGR2RGB) 
        self.index = index
        
        self.response = None
        self.layout = None

        # ocr extraction along with layout default is word level
        self.extraction()
        self.data = None
        self.tokenized_word = self.full_tokenized_word_annotation() 
        self.sentences = self.full_tokenized_word_annotation(lp.TesseractFeatureType.LINE)
        

    def extraction(self):
        st = time.time()
        self.response = self.agent.detect(self.data ,return_response=True)
        self.layout = self.agent.gather_data( 
            self.response, 
            agg_level=lp.TesseractFeatureType.WORD)
        et = time.time()    
    
    def line_level_gather(self):
        return self.agent.gather_data(self.response, agg_level=lp.TesseractFeatureType.BLOCK)
        
    def full_tokenized_word_annotation(self, agg_level=lp.TesseractFeatureType.WORD):
        text_blocks =[ block.text for block in self.agent.gather_data( 
            self.response, 
            agg_level=agg_level)]

        if agg_level == lp.TesseractFeatureType.WORD:
            return preprocessing(text_blocks)
        return text_blocks

    def show(self):
        if self.data is None:
            self.data =  cv.imread(self.image_path)
        lp.draw_text(
            self.data, 
            self.layout, 
            # font controls
            font_size=12, with_box_on_text=True, text_box_width=1).show()
    
    def __len__(self):
        return len(self.layout)
    
    def __getitem__(self, key):
        assert isinstance(key, int)
        assert key <= len(self) - 1
        return self.layout[key]
    
    def parse_syntic_tree(self):
        layouts = self.line_level_gather()
        
        