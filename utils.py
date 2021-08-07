import enum
import time 
from time import strftime, time , gmtime
from dataset.dataset_constants import  ( OCR_EXTRACTED_LAYOUT_OBJ_PKL ,OCR_EXTRACTED_TEXT_PKL)
import pickle

from dataset.preprocessing.nlp import extract_text_pos_from_image

from os.path import exists
import pickle
from multiprocessing import Process , Manager
import time
from collections import defaultdict
from time import strftime, time , gmtime
from dataset.layout_parser import OCRLayoutExtractor

def format_time_stamps(st, et):
    ms = (et - st) %  1
    total_duration = strftime("%H hr : %Mmin :%Ss ",gmtime(et - st)) + '%d ms'%(ms * 1000 )
    return total_duration

def chunks(lst , n):
    for i in range(0, len(lst), n):
        yield lst[i: i + n]

def format_time_stamps(st, et):
    ms = (et - st) %  1
    total_duration = strftime("%H hr : %Mmin :%Ss ",gmtime(et - st)) + '%d ms'%(ms * 1000 )
    return total_duration

def one_job(folder, job_id, output_dict):
    for i, path in enumerate(folder):
        _, _, time = extract_text_pos_from_image(
            path, output_dict=output_dict, debug=False)
        print(f"{i+1}/{len(folder)} Job {job_id}: takes {time}")

def multi_processing_text_ocr(jobs, n_jobs=5):
    if exists(OCR_EXTRACTED_TEXT_PKL):
        print("there exists pre computed dataset")
        with open(OCR_EXTRACTED_TEXT_PKL , "rb")  as fio:
            return pickle.load(fio)
    
    st = time()
    chunked_list = list(chunks(jobs, len(jobs)//n_jobs))
    mger = Manager()
    threads = []
    outs = [ ]

    for i in range(len(chunked_list)):
        dicts = mger.dict()
        dicts['tokens'] = mger.list()
        dicts['pos'] = mger.list()
        outs.append(dicts)

    for i in range(len(chunked_list)):
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
    with open(OCR_EXTRACTED_TEXT_PKL, "wb") as fio:
        pickle.dump(outs,  fio)
    return  outs 


def unpack_proxy_result(outs):
    unpacked_result = defaultdict(list)
    for val in outs:
        for k , v in val.items():
            unpacked_result[k] += list(v)
    return unpacked_result

def one_job(folder, job_id, output_dict):
    for i, path in enumerate(folder):
        _, _, time = extract_text_pos_from_image(
            path, output_dict=output_dict, debug=False)
        print(f"{i+1}/{len(folder)} Job {job_id}: takes {time}")

def unpack_proxy_result(outs):
    unpacked_result = defaultdict(list)
    for val in outs:
        for k , v in val.items():
            unpacked_result[k] += list(v)
    return unpacked_result


def create_ocr_with_image(folder, job_id,  output_dict):
    for i , path in enumerate(folder):
        st = time()
        ocr_extractor = OCRLayoutExtractor(path, i ,False)
        
        output_dict['ocr_obj'].append(ocr_extractor)
        
        duration = time() - st
        
        print(f"{i+1}/{len(folder)} Job {job_id}: takes {duration/1000:.3f} ms")


def multi_processing_layout_text_ocr(jobs, n_jobs=2):
    if exists(OCR_EXTRACTED_LAYOUT_OBJ_PKL):
        print("there exists pre computed dataset")
        with open(OCR_EXTRACTED_LAYOUT_OBJ_PKL , "rb")  as fio:
            return pickle.load(fio)
    
    st = time()
    chunked_list = list(chunks(jobs, len(jobs)//n_jobs))
    mger = Manager()
    threads = []
    outs = [ ]

    for i in range(len(chunked_list)):
        dicts = mger.dict()
        dicts['ocr_obj'] = mger.list()
        outs.append(dicts)

    for i in range(len(chunked_list)):
        p = Process(
            target=create_ocr_with_image,
            args=(chunked_list[i], i ),
            kwargs={ 'output_dict': outs[i]})
        threads.append(p)
        p.start()
    
    for p in threads:
        p.join()
    et = time()
    print(f"multiple jobs with {n_jobs} has total duration of { format_time_stamps(st, et)} ms")
    outs = unpack_proxy_result(outs)
    with open(OCR_EXTRACTED_LAYOUT_OBJ_PKL, "wb") as fio:
        pickle.dump(outs,  fio)
    return  outs 


def load_prepared_dataset(model_pth):
    with open(model_pth, 'rb') as handler:
        st = time()
        print(f"loading {model_pth}" , end=" ")
        datasets = pickle.load(handler)
        et = time()
        print(f"done {(et-st)/1000:.3f} ms")
    return datasets