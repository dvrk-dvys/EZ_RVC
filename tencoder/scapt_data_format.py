import copy
import os
import pickle

# ~~~~~~~~~~~~~~~~~~~~~~ learners ~~~~~~~~~~~~~~~~~~~~~~
from transformers import BertTokenizerFast, DistilBertTokenizerFast, DistilBertConfig, EncoderDecoderModel, DistilBertTokenizer
from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
import torch
from accelerate import Accelerator

# ~~~~~~~~~~~~~~~~~~~~~~ parsers ~~~~~~~~~~~~~~~~~~~~~~
import json
from xml.dom import minidom
import xml.etree.ElementTree as ET
import os
import argparse
import yaml
import traceback
import logging

# ~~~~~~~~~~~~~~~~~~~~~~ local ~~~~~~~~~~~~~~~~~~~~~~
import train.trainer.aspect_finetune as aspect_eval
from train.model import build_absa_model, build_optimizer, LabelSmoothLoss
import argparse

from transformers import BertTokenizer, pipeline

from torch import nn
from torch.nn import functional as F

# from torch_ort import ORTModule
m_encoding = 'UTF-8'

Accelerator = Accelerator()

from sklearn.metrics import multilabel_confusion_matrix as mcm, classification_report
import numpy as np

import colorama
from colorama import Fore, Back, Style

import argparse as args
from train.train import train

import shap
from train.trainer.pretrain import pretrain_Lite, ABSADataset, create_dataloader #,fp16_multi_pretrain

def load_jsonl(path):
    data = []

    with open(path, "r", encoding="utf-8") as f:
        for idx, line in enumerate(f):
            t1 = json.loads(line.strip())
            data.append(t1)

    return data


def create_xml_file(read_path, write_path):
    with open(read_path, 'r') as fp:
        lines = fp.readlines()
    fp.close()
    root = minidom.Document()
    xml = root.createElement("sentences")
    root.appendChild(xml)
    # root = ET.Element("sentences")
    with open(write_path, 'w') as xfile:
        for index, line in enumerate(lines):
            cont= False
            print(line)
            if line == '------------------------------------------------\n':
                continue
            prep_line = line.lower().strip()

            s_id = root.createElement("sentence")
            s_id.setAttribute('id', '001')
            xml.appendChild(s_id)

            textElem = root.createElement('text')
            textElem.appendChild(root.createTextNode(prep_line))

            aterms = root.createElement("aspectTerms")

            end = True
            while end==True:
                ASPECT_TERM = None
                A_from: int = 0
                A_to = 0
                POLARITY = None
                iSENTIMENT = None

                iter = True
                while iter == True:
                    try:
                        ASPECT_TERM = input("Which words qualify as an Aspect Term?:  ").lower()
                        A_from: int = prep_line.index(ASPECT_TERM)
                        A_to = A_from + len(ASPECT_TERM)
                        iter = False
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        # Logs the error appropriately.
                        print("Invalid input please try again!")

                iter = True
                while iter == True:
                    try:
                        POLARITY = {"positive": "positive", "negative": "negative", "neutral": "neutral"}[input("What is the polarity of said Term? Positive, Negative, or Neutral:  ").lower()]
                        iter = False
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        # Logs the error appropriately.
                        print("Invalid input please try again!")

                iter = True
                while iter == True:
                    try:
                        iSENTIMENT = {"true": ['true', 't', 'yes', 'y'],  "false": ['false', 'f', 'no', 'n']}[str(input("Does the Aspect Term have an Implicit Sentiment? True or False:  ")).lower()]
                        iter = False
                    except Exception as e:
                        logging.error(traceback.format_exc())
                        # Logs the error appropriately.
                        print("Invalid input please try again!")
                iSENTIMENT= iSENTIMENT[0]
                if iSENTIMENT in ['false', 'f', 'no', 'n']:
                    iter = True
                    while iter == True:
                        try:
                            OPINION_WORD = input("What is the descriptor opinion word('s) attached to this term?:  ")
                            iter = False
                        except Exception as e:
                            logging.error(traceback.format_exc())
                            # Logs the error appropriately.
                            print("Invalid input please try again!")

                    aspect = root.createElement("aspectTerm")
                    aspect.setAttribute("term", ASPECT_TERM)
                    aspect.setAttribute("polarity", POLARITY)
                    aspect.setAttribute("from", str(A_from))
                    aspect.setAttribute("to", str(A_to))
                    aspect.setAttribute("implicit_sentiment", iSENTIMENT)
                    aspect.setAttribute("opinion_words", OPINION_WORD)
                    aterms.appendChild(aspect)
                elif iSENTIMENT in ['true', 't', 'yes', 'y']:
                    aspect = root.createElement("aspectTerm")
                    aspect.setAttribute("term", ASPECT_TERM)
                    aspect.setAttribute("polarity", POLARITY)
                    aspect.setAttribute("from", str(A_from))
                    aspect.setAttribute("to", str(A_to))
                    aspect.setAttribute("implicit_sentiment", iSENTIMENT)
                    aterms.appendChild(aspect)
                else:
                    print('huh?')

                s_id.appendChild(textElem)
                s_id.appendChild(aterms)

                iter = True
                while iter == True:
                    try:
                        stop = input("Are there any more Aspect Terms?? T or F: ").lower()

                        if stop in ['false', 'f', 'no', 'n', 'stop']:
                            iter = False
                            end = False
                            break
                        elif stop in ['true', 't', 'yes', 'y', 'more']:
                            print('reset')
                            xml_str = root.toprettyxml(indent="\t")
                            part1, part2 = xml_str.split('?>')
                            xfile.write(part2)
                            cont = True
                            iter = False

                    except Exception as e:
                        logging.error(traceback.format_exc())
                        # Logs the error appropriately.
                        print("Invalid input please try again!")
            if cont == True:
                continue

        xml_str = root.toprettyxml(indent="\t")
        part1, part2 = xml_str.split('?>')
        xfile.write (part2)
    xfile.close()


def base_infer(config):
    model = build_absa_model(config).to(torch.device('mps'))
    model_dict = torch.load(config['state_dict'], map_location=torch.device('cpu'))
    model.load_state_dict(state_dict=model_dict)
    model = model.to(torch.device('mps'))

    train_loader, dev_loader, test_loader = aspect_eval.prepare_dataset(config,
                                                            absa_dataset=aspect_eval.ABSADataset,
                                                            collate_fn=aspect_eval.collate_fn)

    val_acc, val_f1, _, explicit_acc, implicit_acc, output = aspect_eval.evaluate(config, model, dev_loader)
    # print("valid f1: {:.4f}, valid acc: {:.4f}, explicit acc: {:.4f}, implicits acc: {:.4f}".format(val_f1, val_acc, explicit_acc, implicit_acc))
        # return
    # return model, dev_loader
    accuracy, f1, average_loss, explicit_acc, implicit_acc, output = aspect_eval.evaluate(config, model, train_loader, criterion=None)
    return accuracy, f1, average_loss, explicit_acc, implicit_acc, output, model

if __name__ == '__main__':
    # ~~~~~~~~~~~ CONVERT TWEETS AND OTHERS TEXTS TO XML FORMAT ~~~~~~~~~~~
    # create_xml_file(read_path="/Users/jordanharris/SCAPT-ABSA/Thesis/usa_tweets_clean.txt",
    #                 write_path="/Users/jordanharris/SCAPT-ABSA/data/iSA/iSA_Train_v2_Implicit_Labeled.xml")

    tz = BertTokenizerFast.from_pretrained("bert-base-uncased")
    # config = yaml.safe_load(open(args.config))
    # torch.load(config)
    # config_iSA = yaml.safe_load(open('/Users/jordanharris/SCAPT-ABSA/config/iSA_BERT_finetune.yml'))

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    parser = argparse.ArgumentParser(description='Train model')
    parser.add_argument('--config', required=True, help='path to yaml config file')
    parser.add_argument('--checkpoint', help='path to model checkpoint')
    args = parser.parse_args()

    config = yaml.safe_load(open(args.config))
    if 'checkpoint' in args:
        config['checkpoint'] = args.checkpoint

    accuracy, f1, average_loss, explicit_acc, implicit_acc, out, model = base_infer(config)

    out[1] = out[1].to(device=torch.device('cpu'))
    val_targets = (np.array(out[1]) > 0.5).astype(int)

    out[2] = out[2].to(device=torch.device('cpu'))
    val_preds = (np.array(out[2]) > 0.5).astype(int)

    sentiment_dict = {
        0: 'positive',
        1: 'negative',
        2: 'neutral',
        3: 'conflict',
    }

    # out =  [output, labels_all, preds_all, batch]
    # out[3] =  [bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits] * len(batches) = batch

    label_list = out[1].tolist()
    pred_list = out[2].tolist()

    logits = out[0].sentiment
    preds_prob = F.softmax(out[0].logits, dim=-1)
    preds = logits.argmax(dim=-1)

    cls_hid = out[0].cls_hidden
    cls_prob = F.softmax(cls_hid, dim=-1)
    cls_hiddens = cls_hid.argmax(dim=-1)

    attns = []
    for attn in out[0].attentions:
        attn_prob = F.softmax(attn, dim=-1)
        attns.append(attn.argmax(dim=-1))

    # last_lables = out[3][-1]

    # pred_test_tensor = tz.convert_ids_to_tokens(preds[0])
    # pred_test_list = tz.convert_ids_to_tokens(preds.tolist()[0])
    # untokenized_text = tz.decode(token_ids=out[3][0][0])

    # aspct_msk = out[3][2]
    # decd = tz.decode(aspct_msk[0])
    # test = tz.convert_ids_to_tokens(aspct_msk[0])
    # token = decd.replace('[PAD]', "").strip()
    # aspect_index = test.index(token)
    # word_again = tz.vocab[aspct_msk[aspect_index]]

    # correct_samples += (pred.to('cpu') == labels.to('cpu')).long().sum().item()

    p_idx = 0
    for b in range(len(out[3])): #batches
        for ix, a_t in enumerate(out[3][b]):

            print('Actual Sentiment: ', sentiment_dict[label_list[p_idx]])

            if pred_list[p_idx] != label_list[p_idx]:
                print(Fore.RED + 'Predicted Sentiment: ', sentiment_dict[pred_list[p_idx]])
                # tz.decode(attns[-1].tolist()[0][1])
                # tz.decode(a_t.tolist()[ix])
                print(Fore.YELLOW + 'Aspect Term: ', out[3][b][5][ix], ', Raw Text: ', out[3][b][4][ix])
                print(Fore.WHITE + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(Style.RESET_ALL)
            else:
                print(Fore.GREEN + 'Predicted Sentiment: ', sentiment_dict[pred_list[p_idx]])
                print(Fore.YELLOW + 'Aspect Term: ', out[3][b][5][ix], ', Raw Text: ', out[3][b][4][ix])

                print(Fore.WHITE + '~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
                print(Style.RESET_ALL)
            p_idx+=1

    cm = mcm(val_targets, val_preds)
    print(classification_report(val_targets, val_preds))

    #
    # print(DistilBertTokenizerFast.convert_ids_to_tokens(encoding["input_ids"].squeeze())[:20])
    # tz = BertTokenizer.from_pretrained("bert-base-uncased")
    # test = tz.convert_tokens_to_ids(tz.tokenize("characteristically"))
    #
    # word_embedding = model.bert.embeddings.word_embeddings.weight
    #
    # preds_proba = F.softmax(out[0].logits, dim=-1)
    # preds = preds_proba.argmax(dim=-1)

    # out =  [output, labels_all, preds_all, batch]
    # out[3] =  bert_tokens, bert_masks, aspect_masks, labels, raw_texts, raw_aspect_terms, implicits = batch
    #
    # clsencodings!

    # tz = BertTokenizer.from_pretrained("bert-base-uncased")

    # help = tz.decode(self.labels[0])
    #
    # for target in out[1][:2]:
    #
    #     print("ActualTarget Values : {}".format(tz.decode([out[3][3][target] for target in out[1][:2]])))
    #
    #     print("Actual    Target Values : {}".format(tz.decode([out[3][3][target] for target in out[1][:2]])))
    #     print("Predicted Target Values : {}".format([out[3][3][target] for target in preds]))
    #     print("Predicted Probabilities : {}".format(preds_prob.max(dim=-1)))

    # retrieve index of [MASK]

    #
    # text = 'The mideval streets of the city were due for a good cleaning and power washing.'
    # output = model(text)
    # output.argmax(1).item() + 1
    #
    #
    # unmasker = pipeline('fill-mask', model='bert-base-uncased')
    # unmasker("The man worked as a [MASK].")
    #
    # untokenized_text = tz.decode(token_ids=out[3][0][0])
    # tz.batch_decode()
    # print(tz.decode(token_ids=out[3][0][5]))
    #
    #
    # masker = shap.maskers.Text(tokenizer=r"\W+")
    # explainer = shap.Explainer(make_predictions, masker=masker, output_names=target_classes)
    # shap_values = explainer(X_test[:2])
    # shap.text_plot(shap_values)
    #



# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# cls = out[0].cls_hidden #.tolist()
# preds_prob = F.softmax(out[0].logits, dim=-1)
# preds = preds_prob.argmax(dim=-1)
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
#
#
# # The streets of Paris were an unbearable medieval mess, that was in desperate need of clearing anyway.
# # test 4000 :: train 14000  :: 0.02857142857
#
# <sentence id="958:1">
#     <text>Other than not being a fan of click pads (industry standard these days) and the lousy internal speakers, it's hard for me to find things about this notebook I don't like, especially considering the $350 price tag.</text>
#     <aspectTerms>
#         <aspectTerm term="internal speakers" polarity="negative" from="86" to="103" implicit_sentiment="False" opinion_words="lousy"/>
#         <aspectTerm term="price tag" polarity="positive" from="203" to="212" implicit_sentiment="True"/>
#         <aspectTerm term="click pads" polarity="negative" from="30" to="40" implicit_sentiment="True"/>
#     </aspectTerms>
# </sentence>
# <sentence id="562:1">
#     <text>Did not enjoy the new Windows 8 and touchscreen functions.</text>
#     <aspectTerms>
#         <aspectTerm term="Windows 8" polarity="negative" from="22" to="31" implicit_sentiment="False" opinion_words="not enjoy"/>
#         <aspectTerm term="touchscreen functions" polarity="negative" from="36" to="57" implicit_sentiment="False" opinion_words="not enjoy"/>
#     </aspectTerms>
# </sentence>
# <sentence id="1141:1">
#     <text>I would have given it 5 starts was it not for the fact that it had Windows 8</text>
#     <aspectTerms>
#         <aspectTerm term="Windows 8" polarity="negative" from="67" to="76" implicit_sentiment="True"/>
#     </aspectTerms>
# </sentence>
#
# # {"text": "The design is sleek and elegant, yet the case can stand up to a good beating.",
# # "tokens": ["the", "design", "is", "sleek", "and", "elegant", ",", "yet", "the", "case", "can", "stand", "up", "to", "a", "good", "beating", "."],
# # "aspect_terms": [{"aspect_term": "design", "left_index": 4, "right_index": 10, "sentiment": 5},
# # {"aspect_term": "case", "left_index": 41, "right_index": 45, "sentiment": 5},
# # {"aspect_term": "stand", "left_index": 50, "right_index": 55, "sentiment": 5}]}