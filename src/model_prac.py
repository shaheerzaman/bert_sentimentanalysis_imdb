import config
import transformers
import torch.nn as nn

class BERTBaseUncased(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.BertModel.from_pretrained(config.BERT_PATH)
        self.bert_drop = nn.Dropout(0.3)
        self.out = nn.Linear(768, 1)
        
    def forward(self, ids, mask, token_type_ids):
        _, o2 = self.bert(
            ids=ids, 
            mask=mask, 
            token_type_ids=token_type_ids
        )
        drop_out = self.bert_drop(o2)
        out = self.out(drop_out)
        return out