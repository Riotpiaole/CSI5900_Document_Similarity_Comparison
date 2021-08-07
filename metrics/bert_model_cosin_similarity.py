import torch.nn as nn
from pdb import set_trace
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

'''
1. vectorized the documnt representation using bert  
2. compute based on cosin similarity as part of bert score
'''


def bert_vectorized_similarity(bert ,base_document, target_document, loss_fn=nn.CosineSimilarity()):
    # base_res = tokenizer.encode_plus(
    #         base_document, add_special_tokens=True,
    #         max_length = 128,           # Pad & truncate all sentences.
    #         truncation=True, padding='max_length', return_tensors='pt')
    # tar_res =  tokenizer.encode_plus(
    #         target_document, add_special_tokens=True,
    #         max_length = 128,           # Pad & truncate all sentences.
    #         truncation=True, padding='max_length', return_tensors='pt')


    base_input_id, base_attention_mask = base_document
    target_input_id , target_attention_mask = target_document
    
    base_bert_vec_repr = bert(
        input_ids=base_input_id,
        attention_mask = base_attention_mask)['logits']
    target_bert_vec_repr = bert(
         input_ids=target_input_id, 
        attention_mask=target_attention_mask)['logits']
    return  loss_fn(base_bert_vec_repr, target_bert_vec_repr)

def sentence_cosine_similarity(model, base_document, target_document ):
    base_document , target_document = " ".join(base_document) ,  " ".join(target_document)
    src_embedding = model.encode(base_document)
    target_embedding =  model.encode(target_document)
    cos_loss =  cosine_similarity([src_embedding], [target_embedding])
    return cos_loss