from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string
from nltk.corpus import stopwords
from pdb import set_trace
from nltk.tokenize import word_tokenize
import pytesseract
import cv2 as cv

from time import time
from .image_util import showImage
from .preprocessing_constants import PUNCTUATION , WORDS , TESSACT_PATH ,STOPWORDS

def preprocessing(text : str,  
        remove_stop_word=True, 
        porter_stemming=False, 
        lemmatization=False, 
        remove_digit=False,
        remove_punctuation=True,
        remove_nonalpha=True,
        remove_non_english_words=False):
    
    text = word_tokenize(text) if isinstance(text, str) else text

    # stemming and lemmatization stop words
    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    stop_words = [ word 
                for word in STOPWORDS] + STOPWORDS
    stop_words = set(stop_words)
    #  removing digits
    if remove_digit:
        removing_digits = [ digit for digit in string.digits ]
        set_trace()
        text = list(filter( lambda x: x not in removing_digits, text))
    
    # removing punctuation 
    if remove_punctuation:
        text = list(filter( lambda x: x not in PUNCTUATION, text))
    
    if remove_nonalpha:
        text = [ 
            word for word in text if word.isalnum()]

    # removing stop word
    if remove_stop_word:
        text = list(filter( 
            lambda x: x not in STOPWORDS,
            text
        ))
    
    if remove_non_english_words:
        text = list(filter( 
            lambda x: x in WORDS,
            text
        ))

    for i in range(len(text)): 
        if porter_stemming: text[i] = stemmer.stem(text[i])
        if lemmatization: text[i] = lemmatizer.lemmatize(text[i])
    
    tokens = list(
        filter( lambda x: len(x) > 1 or x in STOPWORDS, text))
    return tokens 

def part_of_speech_tagging(text):
    assert type(text) is str , "Pos only works with text "
    result = TextBlob(text)
    return result.tags


def extract_text_pos_from_image(image_path, 
        threshold_filter=False, 
        debug=False, 
        output_dict=None,
    ): 
    # page segment mode with osd  so 2
    # the additional flag is --psm 2 --oem 2
    custom_oem_psm_config = r"--oem 2 --psm 6"
    pytesseract.pytesseract.tesseract_cmd = TESSACT_PATH
    image = cv.imread(image_path)
    if threshold_filter: image = cv.threshold(image, 0, 255,  cv.THRESH_BINARY | cv.THRESH_OTSU)[1]
    if debug and threshold_filter : showImage(image, "threshold" if threshold_filter else "origin")
    
    st = time()
    text = pytesseract.image_to_string(image, config=custom_oem_psm_config  ,lang='eng').strip().lower()
    image_to_text_time = time()
    image_to_data_duration = image_to_text_time - st
    if debug: print( f" image to text  {image_to_data_duration}")


    st = time()
    tokens = preprocessing(text)
    preprocessing_time_interval = time() 
    preprocessing_time =  preprocessing_time_interval - st
    if debug: print(f" preprocessing_time : {preprocessing_time}")

    tokens = list(filter( lambda x: len(x) > 1, tokens))

    st = time()    
    pos = part_of_speech_tagging(' '.join(tokens))
    pos_duration_time = time() 
    pos_time = pos_duration_time - st


    if debug: print(f"part of speech tagging {pos_time}")
    if output_dict:
        output_dict['tokens'].append(tokens)
        output_dict['pos'].append(pos)
    return tokens , pos ,  (image_to_data_duration , preprocessing_time , pos_time)


