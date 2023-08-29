# !pip install transformers
# !pip install datasets
# !pip install seqeval
import os
import sys

import pandas as pd
from hydra.utils import instantiate
import torch
from tqdm import tqdm
from transformers import (AutoConfig,
                          AutoModelForTokenClassification,
                          AutoTokenizer,
                          AutoModel,
                          pipeline,
                          Trainer,
                          TrainingArguments,
                          set_seed)
task_name = "ner"
model_name_or_path = 'bert-base-cased'
output_dir = 'tmp'
overwrite_output_dir = True
num_train_epochs = 1
train_file = "train_clean.csv"
preprocessing_num_workers = None
padding = "max_length"
label_all_tokens = True
seed = 42
set_seed(seed)

batch_size = 1
n_ways = 1
n_queries = 1
seq_len = 128

# label_list = ['O', 'B-PER', 'I-PER', 'B-ORG', 'I-ORG', 'B-LOC', 'I-LOC']
label_list = ['0', '1', '2', '3', '4', '5', '6']

label_to_id = {l: i for i, l in enumerate(label_list)}
id_to_label = {label_to_id[x]: x  for x in label_to_id}


config = AutoConfig.from_pretrained(model_name_or_path, num_labels=7, finetuning_task=task_name)
tokenizer_ = AutoTokenizer.from_pretrained(model_name_or_path, use_fast=True)
model_ = AutoModelForTokenClassification.from_pretrained(model_name_or_path,
                                                         config=config)
# model_ = model_.from_pretrained("/Users/jordanharris/SCAPT-ABSA/NER_BERT/test-un-ner/checkpoint-500")


enoder_config = {'_target_': 'fewie.encoders.transformer.TransformerEncoder', 'model_name_or_path': 'bert-base-uncased'}
# enoder_config['id2label'] = id_to_label
# enoder_config.id2label = label_to_id
encoder = instantiate(enoder_config)
encoder.model = encoder.model.from_pretrained('/Users/jordanharris/SCAPT-ABSA/NER_BERT/fewie_encoder-test')
# encoder.load_state_dict(torch.load('/Users/jordanharris/SCAPT-ABSA/NER_BERT/fewie_encoder.pt'))
# encoder = AutoModel.from_pretrained("/Users/jordanharris/SCAPT-ABSA/NER_BERT/test-un-ner/checkpoint-500/")

dataset_processor = {'_target_': 'fewie.dataset_processors.transformer.TransformerProcessor', 'tokenizer_name_or_path': 'bert-base-uncased', 'max_length': 128, 'label_all_tokens': False}
dataset_processor = instantiate(
    dataset_processor,
    label_to_id=label_to_id,
    id_to_label=id_to_label,
    text_column_name='tokens',
    label_column_name='ner_tags',
)

# enc_tkzr.eval()

enc_tkzr = dataset_processor.tokenizer
test_encodings = enc_tkzr(["Before proceeding further, I should like to inform members that action on draft resolution iv, entitled situation of human rights of Rohingya Muslims and other minorities in Myanmar is postponed to a later date to allow time for the review of its programme budget implications by the fifth committee. The assembly will take action on draft resolution iv as soon as the report of the fifth committee on the programme budget implications is available. I now give the floor to delegations wishing to deliver explanations of vote or position before voting or adoption."],
                           padding=padding, truncation=True, max_length=128)



# is_split_into_words=True


res = encoder(torch.tensor(test_encodings["input_ids"]).long(), attention_mask=torch.tensor(test_encodings["attention_mask"]).long())[0].argmax(dim=2)
collect = []
for i, enc in enumerate(test_encodings['input_ids'][0]):
    if enc:
        print(enc_tkzr.decode(enc))
    collect.append(enc_tkzr.decode(enc))
    # dataset_processor.tokenizer.decode(res[0][i])
    # print(enc_tkzr.decode(enc), "\t", id_to_label[res[0][i].item()])

print()
# tokenizer.convert_ids_to_tokens(res[0][1].item())
# res[0].numpy()

enc_ser = pd.Series(data=test_encodings, index=['input_ids', 'token_type_ids', 'attention_mask'])
# # enc_data = dataset_processor(enc_ser)
# dataloader = torch.utils.data.DataLoader(
#     dataset=test_encodings.data,
#     # dataset=enc_ser,
#     batch_size=batch_size,
# )

# dataset = {'_target_': 'datasets.load_dataset', 'path': 'wikiann', 'name': 'en', 'version': '1.1.0', 'split': 'test'}
# dataset = instantiate(dataset)
#
# with torch.no_grad():
#     for batch in tqdm(dataloader):
    # test_encodings = tokenizer(["heart disease, a new house, a dose of penicillin, and bowel cancer. The diagnosis of COPD, a flashy new car and a skin rash."],
    #                            padding=padding) #, truncation=True, return_tensors="pt").input_ids
    #     test_encodings = tokenizer(["heart disease, a new house, a dose of penicillin, and bowel cancer. The diagnosis of COPD, a flashy new car and a skin rash."],
    #                                padding=padding, truncation=True)

# enc_pipeline = pipeline(task="ner", model=model.base_model, tokenizer=tokenizer)
# enc_pipeline("heart disease, a new house, a dose of penicillin, and bowel cancer. The diagnosis of COPD, a flashy new car and a skin rash.")
# (**query).embeddings.view(
#                 batch_size, n_ways * n_queries, seq_len, -1
#             )



# for _ in test_encodings.data:
#     test_encodings.data[_] = torch.tensor(test_encodings.data[_]).view(batch_size * n_ways * n_queries, seq_len).long()
# # test_encodings = pd.Series(data=test_encodings, index=['input_ids', 'token_type_ids', 'attention_mask'])
# input_encodings = pd.Series(data=test_encodings['input_ids'], index=['input_ids'])

# test_encodings.data['input_ids'] = test_encodings.data['input_ids'].view(batch_size * n_ways * n_queries, seq_len).long()
# res = model(test_encodings['input_ids']).embeddings[0].argmax(dim=2)
