from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from nltk import sent_tokenize

def bert_embedding_similarity(base_document, target_document):
    sbert_model = SentenceTransformer('bert-base-nli-mean-tokens')
    sentences = sent_tokenize(base_document)
    base_embeddings_sentences = sbert_model.encode(sentences)

    return  