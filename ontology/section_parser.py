from os import closerange
from pdb import set_trace
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.model_selection import train_test_split
from os.path import exists
from collections import defaultdict
import cv2 as cv
import numpy as np 

from sklearn.feature_extraction.text import TfidfVectorizer

import pytesseract
from income_tax_1988 import generate_dataset
from multiprocessing import Process , Manager
from multiprocessing.managers import BaseManager, DictProxy
import pickle 

import sys
from time import strftime, time , gmtime
sys.path.append("..")
from constants import PUNCTUATION, ROOT_DIR , TESSACT_PATH , WORDS

from preprocessing.nlp import part_of_speech_tagging , preprocessing 
pytesseract.pytesseract.tesseract_cmd = TESSACT_PATH
import nltk


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

def format_time_stamps(st, et):
    ms = (et - st) %  1
    total_duration = strftime("%H hr : %Mmin :%Ss ",gmtime(et - st)) + '%d ms'%(ms * 1000 )
    return total_duration


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


def extract_text_pos_from_image(image_path, 
        threshold_filter=False, 
        debug=True, 
        output_dict=None,
    ): 
    image = cv.imread(image_path)
    if threshold_filter: image = cv.threshold(image, 0, 255,  cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if debug and threshold_filter : showImage(image, "threshold" if threshold_filter else "origin")
    
    st = time()
    text = pytesseract.image_to_data(image, lang='eng').strip().lower()
    image_to_text_time = time()
    image_to_data_duration = format_time_stamps(st, image_to_text_time)
    if debug: print( f" image to text  {image_to_data_duration}")


    st = time()
    tokens = preprocessing(text)
    preprocessing_time_interval = time() 
    preprocessing_time = format_time_stamps(st, preprocessing_time_interval)
    if debug: print(f" preprocessing_time : {preprocessing_time}")

    tokens = list(filter( lambda x: len(x) > 1, tokens))

    st = time()    
    pos = part_of_speech_tagging(' '.join(tokens))
    pos_duration_time = time() 
    pos_time = format_time_stamps(st, pos_duration_time)


    if debug: print(f"part of speech tagging {pos_time}")
    if output_dict:
        output_dict['tokens'].append(tokens)
        output_dict['pos'].append(pos)
    return tokens , pos ,  (image_to_data_duration , preprocessing_time , pos_time)

def term_frequency_vectorizer():
    vectorizer = TfidfVectorizer()
    return vectorizer

def chunks(lst , n):
    for i in range(0, len(lst)):
        yield lst[i: i + n]

def one_job(folder, job_id, output_dict):
    for i, path in enumerate(folder):
        _, _, time = extract_text_pos_from_image(
            path, output_dict=output_dict, debug=False)
        print(f"{i+1}/{len(folder)} Job {job_id}: takes {time}")

def job_splitting(jobs, n_jobs=5):
    if exists("../token_pos.pkl"):
        print("there exists pre computed dataset")
        with open( "../token_pos.pkl" , "rb")  as fio:
            return pickle.load(fio)
    
    st = time()
    chunked_list = list(chunks(jobs, len(jobs)//n_jobs))
    mger = Manager()
    threads = []
    outs = [ ]
    for i in range(n_jobs):
        dicts = mger.dict()
        dicts['tokens'] = mger.list()
        dicts['pos'] = mger.list()
        outs.append(dicts)
    
    for i in range(n_jobs):
        p = Process(
            target=one_job,
            args=(chunked_list[i], i ),
            kwargs={ 'output_dict': outs[i]})
        threads.append(p)
        p.start()
    
    for p in threads:
        p.join()
    et = time()
    print(f"multiple jobs with {n_jobs} has total duration of { format_time_stamps(st, et)}")
    outs = unpack_proxy_result(outs)
    with open("../token_pos.pkl", "wb") as fio:
        pickle.dump(outs,  fio)
    return  outs 

def default_array():
    return []

def unpack_proxy_result(outs):
    unpacked_result = defaultdict(default_array)
    for val in outs:
        for k , v in val.items():
            unpacked_result[k] += list(v)
    return unpacked_result

def try_join(l):
    try:
        return ','.join(map(str, l))
    except TypeError:
        return np.nan

if __name__ == "__main__":
    df = generate_dataset()
    outs = job_splitting(df.image)
    docs_idfs = [ TfidfVectorizer(min_df=1, max_df= .95) for i in df.label.unique()] 

    X=  defaultdict(default_array)
    res = pd.DataFrame({ 
        'image_path': df.image,
        'tokenized_text': list(map(try_join, outs['tokens'])),
        'label': df.label
    })

    scores , n = 0 , 100
    random_test_entries = res.sample(n=n, replace=True,random_state=42)
    print(random_test_entries)
    for i, random_test_entry  in random_test_entries.iterrows():
        avg = []
        class_docs , associate_img_path , associate_label = [] , [] , []
        for cls in set(df.label):
            document = res[res.label == cls].sample(n=200, replace=True)
            class_docs+= document.tokenized_text.tolist()
            associate_img_path += document.image_path.tolist()
            associate_label += document.label.tolist()

        class_docs += [random_test_entry.tokenized_text]
        idf = TfidfVectorizer()
        embedding = idf.fit_transform(class_docs)
        
        cosine_similarities = cosine_similarity( embedding[-1], embedding[:-1]).flatten()

        top_3 = []
        for i in cosine_similarities.argsort()[-5:]:
            top_3.append(associate_label[i])
        if random_test_entry.label in top_3:
            scores += 1
        print( top_3, random_test_entry.label, random_test_entry.label in top_3)
    print(f"{scores}/{n}")