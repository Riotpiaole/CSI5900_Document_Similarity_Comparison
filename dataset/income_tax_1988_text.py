from re import S
from typing import List, Tuple

from torch.utils.data import Dataset 
from transformers import BertTokenizer
from pdb import set_trace

class IncomeTaxTextDataSet(Dataset):
    def __init__(self, text, labels,  indexs: List[Tuple[int]], 
        filter_fmt=None, bert_encode=True,  seq_len=68 ) -> None:
        super().__init__()
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.indexs = indexs
        self.text = text
        self.labels = labels
        self.bert_encode = bert_encode
        self.max_length =seq_len

    def __len__(self) -> int:
        return len(self.indexs)
    
    def __getitem__(self, index: int, text=False):
        src_index , target_index = self.indexs[index]
        src_text = self.text[src_index]
        target_text = self.text[target_index]
        label = self.labels[src_index] == self.labels[target_index]

        if not self.bert_encode: 
            return src_text , target_text , label 
        
        src_tokens_dict = self.tokenizer.encode_plus(
            src_text,add_special_tokens=True,
            max_length = self.max_length,           # Pad & truncate all sentences.
            truncation=True,
            add_special_token=True,
            padding='max_length',
            return_tensors='pt')
        
        target_tokens_dict = self.tokenizer.encode_plus(
            target_text,
            add_special_tokens=True,
            max_length = self.max_length,           # Pad & truncate all sentences.
            truncation=True,
            add_special_token=True,
            padding='max_length',
            return_tensors='pt')
        
        src_input_id , src_mask= src_tokens_dict['input_ids'][0] , src_tokens_dict['attention_mask'][0]
        target_input_id , target_mask= target_tokens_dict['input_ids'][0] , target_tokens_dict['attention_mask'][0]


        return ( 
            src_input_id, src_mask,
            target_input_id, target_mask,
            label
        )

