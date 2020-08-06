import config
import dataset
import torch
import pandas as pd
from sklearn import model_selection
from sklearn import metrics
from transformers import get_linear_schedule_with_warmup
from transformers import AdamW
import numpy as np

import engine
from model import BERTBaseUncased
import dataset

def run():
    df = pd.read_csv(config.TRAINING_FILE)
    df.sentiment = df.sentiment.apply(
        lambda x:1 if x=='positive' else 0
    )
    df_train, df_valid = model_selection.train_test_split(
        df, 
        test_size=0.1,
        random_state=42, 
        stratify=df.sentiment.values         
    )
    df_train = df_train.reset_index(drop=True)
    df_valid = df_valid.reset_index(drop=True)

    train_dataset = dataset.BERTDataset(
        review=df_train.review.values, 
        target=df_train.sentiment.values
    )
    
    train_data_loader = torch.utils.data.DataLoader(
        train_data_loader,
        batch_size=config.TRAIN_BATCH_SIZE, 
        shuffle=True, 
        num_workers=2
    )

    valid_dataset = dataset.BERTDataset(
        review=df_valid.review.values, 
        target=df_valid.sentiment.values
    )

    valid_dataloader = torch.utils.data.DataLoader(
        valid_dataset, 
        batch_size=config.VALID_BATCH_SIZE, 
        num_workers=2
    )

    device = torch.device('cpu')
    model = BERTBaseUncased()
    model.to(device)
    
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_parameters = [
        {'params':[p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weigth_decay':0.001}, 
        {'params':[p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay':0.0}
    ]

    num_train_steps = int(len(df_train)/config.TRAIN_BATCH_SIZE*num_train_steps)
    optimizer = AdamW(optimizer_parameters, lr=2e-5)
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_train_steps=0,
        num_training_steps=num_train_steps
    )

    best_accuracy=0
    for epoch in range(config.EPOCHS):
        engine.train_fn(train_data_loader, model, optimizer, device, scheduler)
        outputs, targets = engine.eval_fn(valid_data_loader, model, device)
        outputs = np.array(outputs) > 0.5
        accuracy = model_selection.accuracy_score(targets, outputs)
        if accuracy > best_accuracy:
            torch.save(model.state_dict(), config.MODEL_PATH)
            best_accuracy = accuracy
        
if __name__ == '__main__':
    run()
        
    
    

    