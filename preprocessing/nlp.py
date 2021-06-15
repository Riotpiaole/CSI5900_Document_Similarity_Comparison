from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob
import string
from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize
from pdb import set_trace
import sys
sys.path.append("..")
from constants import PUNCTUATION, WORDS

def preprocessing(text : str,  
        remove_stop_word=True, 
        porter_stemming=True, 
        lemmatization=True, 
        remove_digit=True,
        remove_punctuation=True):

    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    stop_words = [ lemmatizer.lemmatize(stemmer.stem(word)) 
                for word in stopwords.words('english')] + stopwords.words('english')
    stop_words = set(stop_words)
    if remove_digit:
        removing_digits = str.maketrans('', '', string.digits)
        text = text.translate(removing_digits)
    
    if remove_punctuation:
        removing_puncutation = str.maketrans('', '', PUNCTUATION)
        text = text.translate(removing_puncutation)
    

    tokens = word_tokenize(text)
    tokens = list(filter( 
        lambda x: x in WORDS,
        tokens
    ))

    for i in range(len(tokens)): 
        if porter_stemming: tokens[i] = stemmer.stem(tokens[i])
        if lemmatization: tokens[i] = lemmatizer.lemmatize(tokens[i])

    
    tokens = list(
        filter( lambda x: len(x) > 1 or x in stop_words, tokens))
    return tokens 

def part_of_speech_tagging(text):
    assert type(text) is str , "Pos only works with text "
    result = TextBlob(text)
    return result.tags