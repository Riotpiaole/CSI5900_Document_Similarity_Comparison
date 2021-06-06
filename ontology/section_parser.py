import cv2 as cv
import numpy as np 
 
from skimage.metrics import structural_similarity
import pytesseract

import sys
from time import time
sys.path.append("..")
from constants import ROOT_DIR , TESSACT_PATH


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

def extract_text_from_image(image, 
        threshold_filter=False, 
        numeric=False,
        tesseract_cmd_path = TESSACT_PATH, 
        debug=False):
    pytesseract.pytesseract.tesseract_cmd = tesseract_cmd_path
    
    if threshold_filter: image = cv.threshold(image, 0, 255,  cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if debug: showImage(image, "threshold" if threshold_filter else "origin")
    
    text = pytesseract.image_to_data(image, lang='eng').split()
    taged_text = text.split()
    
    text_numeric = list(filter(lambda x: x.isdigit() and int(x) != 0 )) if numeric else []
    text_alpha = list(filter(lambda x: x.isalpha(), taged_text ))
    
    term_frequencies = text_alpha + text_numeric
    
    





if __name__ == "__main__":
    img1 = cv.imread(f"{ROOT_DIR}/../sd02/data/sfrs_0/r0000/r0000_00.png")
    img2 = cv.imread(f"{ROOT_DIR}/../sd02/data/sfrs_0/r0000/r0000_01.png")
    

    # orb_feature_matcher(img1, img2, True)

    img1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    img2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)

    score , diff  = structural_similarity( img1 , img2,  full=True)
    
    cutted_layout_img1 = layout_cutting(img1,)
    cutted_layout_img2 = layout_cutting(img2,)
    

    threshold_img1 =cv.threshold(img1, 0, 255,  cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    details = pytesseract.image_to_data(threshold_img1, lang='eng')
    


    for i1, i2 in zip(cutted_layout_img1, cutted_layout_img2):
        score , diff = structural_similarity( i1, i2, full=True)
        print(score)