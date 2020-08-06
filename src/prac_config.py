import  transformers

MAX_LEN=512
TRAIN_BATCH_SIZE=2
VALID_BATCH_SIZE=2
EPOCHS=10
ACCUMULATIONS=2
BERT_PATH='../input/bert_case_uncased/'
MODEL_PATH='model.bin'
TRANINING_FILE='../input/data.csv'
TOKENIZER=transformers.BertTokenizer.from_pretrained(BERT_PATH)

