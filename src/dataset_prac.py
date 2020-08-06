import config
import torch

class BERTDataset:
    def __init__(self, review, target):
        self.review = review
        self.target = target
        self.tokenizer = config.TOKENIZER
        self.max_len = config.MAX_LEN

    def __len__(self):
        return len(self.target)

    def __getitem__(self, item):
        review = str(self.review[item])
        review = ' '.jon(review.split())

        inputs = self.tokenizer.encode_plus(
            review, 
            None, 
            add_special_tokens=True, 
            max_length=self.max_len
        )

        ids = inputs['ids']
        mask = inputs['mask']
        token_type_ids = inputs['token_type_ids']
        
        padding_length = self.max_len - len(ids)
        ids = ids + ([0]*padding_length)
        mask = mask + ([0]*padding_length)
        token_type_ids = token_type_ids + ([0]*padding_length)
        
        return {
            'ids':torch.tensor(ids, dtype=torch.long), 
            'mask':torch.tensor(mask, dtype=torch.long), 
            'token_type_ids':torch.tensor(token_type_ids, dtype=torch.long), 
            'targets':torch.tensor(targets[item], dtype=torch.float)
        }