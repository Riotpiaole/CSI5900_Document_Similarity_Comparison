from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from textblob import TextBlob

import string
from nltk.corpus import stopwords

def preprocessing(text,  
        remove_stop_word=True, 
        porter_stemming=True, 
        lemmatization=True, 
        remove_digit=True,
        remove_punctuation=True):

    assert type(text) is list , "Text is not tokenzied"
    stemmer = PorterStemmer()
    lemmatizer=WordNetLemmatizer()
    
    if remove_stop_word:
        text = list(filter(
            lambda x: 
                    x not in stopwords.words('english') \
                        and (x not in string.punctuation and remove_punctuation)
                        and (not x.isdigit() and remove_digit ), 
            text))

    for i in range(len(text)): 
        if porter_stemming: text[i] = stemmer.stem(text[i])
        if lemmatization: text[i] = lemmatizer.lemmatize(text[i])
    return text 

def part_of_speech_tagging(text):
    assert type(text) is str , "Pos only works with text "
    result = TextBlob(text)
    return result.tags