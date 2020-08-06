import torch
from flask import Flask, jsonify, request

import config
from model import BERTBaseUncased

app = Flask(__name__)

MODEL = None
DEVICE = torch.device('cpu')

def sentence_prediction(sentence, model, device):
    tokenizer = config.TOKENIZER
    max_length = config.MAX_LEN
    review = str(sentence)
    inputs = tokenizer.encode_plus(
        review, 
        None, 
        add_special_tokens=True, 
        truncation=True, 
        max_length=max_length
    )
    ids = inputs['input_ids']
    mask  = inputs['attention_mask']
    token_type_ids = inputs['token_type_ids']
    
    padding_length = max_length - len(ids)
    ids = ids + ([0]*padding_length)
    mask = mask + ([0]*padding_length)
    token_type_ids = token_type_ids + ([0]*token_type_ids)
    
    ids = torch.tensor(ids, dtype=torch.long).unssqeeze(0)
    mask = torch.tensor(mask, dtype=torch.long).unsqueeze(0)
    token_type_ids = torch.tensor(token_type_ids, dtype=torch.long).unsqueeze(0)
    
    ids = ids.to(device, dtype=torch.long)
    mask = mask.to(devic, dtype=torch.long)
    token_type_ids = token_type_ids.to(device, dtype=torch.long)
    
    outputs = model(
        ids=ids, 
        mask=mask, 
        token_type_ids=token_type_ids
    )

    outputs = torch.sigmoid(outputs).cpu().detach().numpy()
    return outputs[0][0]
    
    

@app.route('/predict')
def predict():
    sentence = request.args.get('sentence')
    print(sentence)
    positive_prediction = sentence_prediction(sentence, model=MODEL, device=DEVICE)
    negative_prediction = 1 - positive_prediction
    response = {}
    response['response'] = {
        'positive':positive_prediction, 
        'negative':negative_prediction
    }

    return jsonify(response)

if __name__ == '__main__':
    MODEL = BERTBaseUncased()
    MODEL.load_state_dict(torch.load(config.MODEL_PATH))
    MODEL.to(DEVICE)
    MODEL.eval()
    app.run(debug=True)