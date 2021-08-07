# from itertools import combinations
# from dataset.income_tax_1988_text import IncomeTaxTextDataSet
import os 
from pdb import set_trace
from dataset.dataset_constants import ROOT_DIR
from tqdm import tqdm
from utils import load_prepared_dataset
#TODO 
#  - [*] TF idf vecotorizer for comparing the document against the remaining dataset. 
#  - [*] Layout Parser for tree based similarity 
#  - [*] Image similarity 
#  - [*] bert token similarity
#  - [ ] jaccard similarity

if __name__ == "__main__":
    
    from dataset.income_tax_1988_image import generate_dataset
    from dataset.dataset_constants import OCR_EXTRACTED_LAYOUT_OBJ_PKL
    from dataset.layout_parser import OCRLayoutExtractor
    import numpy as np
    
    from random import shuffle
    from time import time 
    df = generate_dataset()
    st =time()
    datasets = load_prepared_dataset(OCR_EXTRACTED_LAYOUT_OBJ_PKL)['ocr_obj']
    duration= time() - st
    print(f"takes {duration/1000} ms")
    documents = []
    for ocr_layout in datasets:
        documents.append(
            ocr_layout.tokenized_word)

    documents = np.array(documents)
    
    # ==========================================
    # tfidf classifier 
    # ==========================================
    # from metrics.tfidf_classifier import IncomeTaxWordReprTFClassifier
    # tfidf_classifier = IncomeTaxWordReprTFClassifier(documents, df.label)
    # res =  tfidf_classifier(documents[0])

    # ==========================================
    # bert classifier
    # ==========================================
    # from metrics.bert_model_cosin_similarity import bert_vectorized_similarity
    # from transformers import BertTokenizer 
    # from transformers import BertForSequenceClassification
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    # src_doc , target_doc = documents[0] , documents[2]
    # src_label , target_label = df.label[0] , df.label[2]

    # bert = BertForSequenceClassification.from_pretrained(
    #     'bert-base-uncased',num_labels=8,)
    # import torch
    # from metrics.cca_loss import cca_loss
    # cos_score = bert_vectorized_similarity(bert , src_doc, target_doc, tokenizer)
    # cca_score = bert_vectorized_similarity(bert , src_doc, target_doc, tokenizer, loss_fn=cca_loss(8,False, torch.device('cpu')))

    # =========================================
    # image similarity using transfer learning image vector
    # =========================================
    # from metrics.visual_feature_encoding import Img2VecEncoder
    # from dataset.income_tax_1988_image import create_combination_income_dataset_with_split , IncomeTaxImageDataSet
    # import torchvision.transforms as transforms
    # from torch.nn import CosineSimilarity
    # from metrics.img_cmp_func import ssim
    # model = Img2VecEncoder()

    # normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                                     std=[0.229, 0.224, 0.225])

    # train_dataset , test_dataset, val_dataset = create_combination_income_dataset_with_split(df, VGG=True )

    # image_src , image_dest , label = train_dataset[30]

    # src_vecs , dest_vecs = model(image_src.unsqueeze(0)) , model(image_dest.unsqueeze(0))
    # cosLoss = CosineSimilarity()
    
    # cos_loss = cosLoss(
    #     src_vecs.unsqueeze(0), 
    #     dest_vecs.unsqueeze(0))
    
    # ssim_loss = ssim(image_src.unsqueeze(0), image_dest.unsqueeze(0))
    # print(f"cos loss func {cos_loss} {label}")
    # print(f"cos loss func {ssim_loss} {label}")
    
    # ===========================================
    # Sentence encoder 
    # ===========================================
    # from dataset.income_tax_1988_text import IncomeTaxTextDataSet
    # import torch
    # from transformers import BertForSequenceClassification
    # from dataset.layout_parser import OCRLayoutExtractor
    # from sentence_transformers import SentenceTransformer
    # from metrics.bert_model_cosin_similarity import bert_vectorized_similarity , sentence_cosine_similarity
    # from transformers import BertTokenizer
    
    # tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # sentence_bert = SentenceTransformer('bert-base-nli-mean-tokens')
    # bert = BertForSequenceClassification.from_pretrained('bert-base-uncased')
    # indexs = list(combinations(df.index, 2))
    # shuffle(indexs)
    # dataset = IncomeTaxTextDataSet(
    #     documents, df.label, indexs)
    
    # tokenizer = dataset.tokenizer
    # loader = DataLoader(dataset, batch_size=16)
    # count = 0

    # for (src_id, target_id) in dataset.indexs:
    #     src_doc , tar_doc = documents[src_id] , documents[target_id]
    #     with torch.no_grad():
    #         similarity = sentence_cosine_similarity(sentence_bert, src_doc, tar_doc)
    #         if similarity[0][0] > 0.9 and dataset.labels[src_id] == dataset.labels[target_id]:
    #             count += 1
    #             print(f"{similarity[0][0]  > 0.9} vs {dataset.labels[src_id] == dataset.labels[target_id]}",
    #                 end='\r')
    # print(f"{count}/{len(indexs)}")
    
    # ===========================================
    # Layout parser
    # ===========================================
    import layoutparser as lp
    import cv2
    experiment_dataset = datasets[0]
    experiment_dataset.image_path = df.image[0]
    # img = cv2.imread(df.image[0])
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
    line_level_layout = experiment_dataset.line_level_gather()
    # lp.draw_box(img, line_level_layout, box_width=2).show()   
    from metrics.tree_edit_distance.tree import Node
    # for layout in line_level_layout:
    #     print(layout)
        
    