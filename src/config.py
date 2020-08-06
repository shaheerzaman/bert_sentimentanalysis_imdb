import transformers

MAX_LEN=512
TRAIN_BATCH_SIZE=2
VALID_BATCH_SIZE=2
EPOCHS=10
ACCUMULATION=2
BERT_PATH='../input/bert_base_uncased/'
MODEL_PATH='model.bin'
TRAINING_FILE='../input/data.csv'
TOKENIZER=transformers.BertTokenizer.from_pretrained(BERT_PATH)

