from os.path import dirname ,abspath
import string
import nltk

ROOT_DIR = abspath(".")
TESSACT_PATH="D:\\tesseract-ocr\\tesseract"
PUNCTUATION = string.punctuation
try:
    WORDS = set(nltk.corpus.words.words())
except LookupError:
    nltk.download('words')
    WORDS = set(nltk.corpus.words.words())