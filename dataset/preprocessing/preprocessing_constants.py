import nltk
import string

PUNCTUATION = [ punc for punc in string.punctuation]
TESSACT_PATH="D:\\tesseract-ocr\\tesseract"
try:
    WORDS = set(nltk.corpus.words.words())
except LookupError:
    nltk.download('words')
    WORDS = set(nltk.corpus.words.words())

from nltk.corpus import stopwords
STOPWORDS= stopwords.words('english')