from collections import defaultdict
import cv2 as cv
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer

import pytesseract
from incorme_tax_1988 import generate_dataset

import sys
from time import time
from pdb import set_trace
sys.path.append("..")
from constants import PUNCTUATION, ROOT_DIR , TESSACT_PATH

from nltk.tokenize import word_tokenize
from preprocessing.nlp import part_of_speech_tagging , preprocessing 


def showImage(img, tags=""):
    cv.imshow(f"{tags}->sample image", img)
    cv.waitKey(0) # waits until a key is pressed
    cv.destroyAllWindows() # destroys the window showing image

def orb_feature(image, dim=True):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    orb = cv.ORB_create()
    return orb.detectAndCompute(image, None)
    
def orb_feature_matcher(query , target , show=False):
    matcher = cv.BFMatcher()
    query_keypoints, query_descriptor = orb_feature(query)
    train_keypoints, target_descriptor = orb_feature(target)
    matches = matcher.match(query_descriptor, target_descriptor)
    if show:
        final_img = cv.drawMatches(
            query, query_keypoints, 
            target, train_keypoints, 
                matches[:20], None)
        
        final_img = cv.resize(final_img, (1000, 650))
        cv.imshow("Matches", final_img)
        k = cv.waitKey(0)
        if k == 27: 
            return 

def ssim(query, target):
    return ssim(query, target)


def layout_cutting(image,  vertical_ratio=(1,1,3), show=False):
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    assert len(vertical_ratio) >= 2, "ratio must be two integer ie 1:4 or 1:2:3"
    assert sum(vertical_ratio) == 5, "ratio must be sum up to 5"
    r, _ = image.shape
    scales = r//5
    r_start = 0
    cutted_layout_img = []
    for ratio in vertical_ratio:
        cutted_layout_img.append(image[ r_start:ratio*scales + r_start,: ])
        if show: showImage(cutted_layout_img[-1])
        r_start += ratio*scales

    return cutted_layout_img


def extract_text_pos_from_image(image, 
        threshold_filter=False, 
        tesseract_cmd_path = TESSACT_PATH, 
        debug=True):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
    
    if threshold_filter: image = cv.threshold(image, 0, 255,  cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if debug and threshold_filter : showImage(image, "threshold" if threshold_filter else "origin")
    
    st = time()
    text = pytesseract.image_to_data(image, lang='eng').strip().lower()
    image_to_text_time = time() - st 
    
    tokens = word_tokenize(text)
    tokenizes_time = time() - image_to_text_time 

    tokens = preprocessing(tokens)
    preprocessing_time = time() - tokenizes_time

    pos = part_of_speech_tagging(text)
    pos_time = time() - preprocessing_time 
    if debug: 
        print(
            f"ocr_time taken {image_to_text_time * 1000} ms "
            f" tokenization {tokenizes_time * 1000} ms"
            f" preprocessing_time : {preprocessing_time * 1000}ms "
            f"part of speech tagging {pos_time * 1000}ms")
    return tokens , pos

def term_frequency_vectorizer():
    vectorizer = TfidfVectorizer()
    return vectorizer


if __name__ == "__main__":
    img1 = cv.imread(f"{ROOT_DIR}/../sd02/data/sfrs_0/r0000/r0000_00.png")
    img2 = cv.imread(f"{ROOT_DIR}/../sd02/data/sfrs_0/r0000/r0000_01.png")
    
    text , pos = extract_text_pos_from_image(img1)
    vectorizer = term_frequency_vectorizer()
    df = generate_dataset()

    dataframe = defaultdict(lambda : [])

    for img in tqdm(df.image):
        text, pos = extract_text_pos_from_image(img)
        tfidf = vectorizer.fit_transform(text)
        dataframe['tfidf'].append( tfidf )
        dataframe['part_of_speech_tagging'].append( pos )
    

    # for i1, i2 in zip(cutted_layout_img1, cutted_layout_img2):
    #     score , diff = structural_similarity( i1, i2, full=True)
    #     print(score)