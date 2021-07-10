import pandas as pd
from tfidf_classifier import job_splitting
from income_tax_1988 import generate_dataset
import numpy as np
from nltk.tokenize import word_tokenize
from pdb import set_trace


def caclulate_jaccard(word_token1, word_token2):
    word_token1 = word_tokenize(word_token1) if isinstance(word_token1, str) else word_token1
    word_token2 = word_tokenize(word_token2) if isinstance(word_token2, str) else word_token2
    both_tokens  = word_token1 + word_token2

    union = set(both_tokens)

    intersection = set()
    for w in word_token1:
        if w in word_token2:
            intersection.add(w)
    
    jaccard_score = len(intersection)/len(set(union))
    return jaccard_score

def jaccard_score_for_one_class( documents : pd.Series , one_class_population : pd.DataFrame , cls=0 ):
    total_scores = []
    for _, document in documents.iterrows():  
        scores = []
        for  _, sample in one_class_population.iterrows():
            scores.append(
                caclulate_jaccard(
                    document.tokenized_text, 
                    sample.tokenized_text))

            # print(f"{scores[-1]} with {document.label} vs {sample}")
        total_scores.append(sum(scores)/len(scores))

    print(total_scores)
    return total_scores

def jaccard_score_over_entire_population( document : pd.Series, overall_population):
    assert "label" in overall_population.columns
    results = [ [] for i in range(document.shape[0]) ]
    for cls in overall_population.label.unique():
                
        jaccard_scores = jaccard_score_for_one_class(
            document, 
            overall_population[ overall_population.label == cls ],
            cls=cls
        )
        for doc_id in range(len(jaccard_scores)):
            results[doc_id].append(
                jaccard_scores[doc_id])
    
    final_selection , scores = [] , 0
    results = np.array(results)
    for i , result in enumerate(results):
        top_3_result = result.argsort()[-3:].tolist()
        final_selection.append(  top_3_result )
        set_trace()
        if document.iloc[i].label in top_3_result :
            scores += 1

    return final_selection , scores / document.shape[0]

def try_join(l):
    try:
        return ','.join(map(str, l))
    except TypeError:
        return np.nan

def default_array():
    return []


if __name__ == "__main__":
    df = generate_dataset()
    n = 100
    outs = job_splitting(df.image)
    res_df = pd.DataFrame({ 
        'image_path': df.image,
        'tokenized_text': outs['tokens'],
        'label': df.label
    })
    random_test_entries = res_df.sample(n=n, replace=True,random_state=42)
    jaccard_score_over_entire_population(random_test_entries,res_df)
