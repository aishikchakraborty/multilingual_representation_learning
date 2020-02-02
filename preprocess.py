import os
import random
import ast
import json
import numpy as np
import argparse
import random
from collections import Counter
from collections import defaultdict
import _pickle as pickle
from tqdm import tqdm
import random
import pandas as pd
from transformers import BertTokenizer


parser = argparse.ArgumentParser(description='Preprocessing wikipedia files')
parser.add_argument('--data', type=str, default='data/wikitext-2/',
                    help='location of the data corpus')

parser.add_argument('--dictionary', type=str, default='data/',
                    help='location of the data corpus')
parser.add_argument('--src', type=str, default='en',
                    help='location of the data corpus')
parser.add_argument('--tgt', type=str, default='de',
                    help='location of the data corpus')


args = parser.parse_args()
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)
tokenizer.save_vocabulary(args.data)


def get_sentence_list(file_path):
    f = open(file_path, 'r')
    sentence_list = []
    for lines in f:
        if lines[0] == '=':
            continue
        else:
            sentence_list.extend(lines.split('.'))
    return sentence_list

def get_bilingual_dict():
    f = open(args.dictionary + args.src + '-' + args.tgt + '.txt', 'r')
    biling_dict = {}
    for lines in f:
        lines = lines.strip().split()
        biling_dict[lines[0]] = lines[1]
    return biling_dict

biling_dict = get_bilingual_dict()
sentence_list_train = get_sentence_list(os.path.join(args.data, 'wiki.train.tokens'))
sentence_list_val = get_sentence_list(os.path.join(args.data, 'wiki.valid.tokens'))
sentence_list_test = get_sentence_list(os.path.join(args.data, 'wiki.test.tokens'))


def create_bert_tokens(sentence_list):
    input_sequence = []
    masked_labels_seq = []
    for sentence in tqdm(sentence_list):
        tokens = tokenizer.tokenize(sentence)
        tokens = ['[CLS]'] + tokens + ['[SEP]']
        bert_input = tokenizer.convert_tokens_to_ids(['[MASK]' if t in biling_dict.keys() else t for t in tokens])
        #mask = tokenizer.get_special_tokens_mask(bert_input, already_has_special_tokens=True)
		
        # print(bert_input)
        # print(mask)
        #masked_labels = [a*b for a,b in zip(bert_input, mask)]
        #masked_labels = [-1 if m==0 else m for m in masked_labels]
        masked_labels = [tokenizer.convert_tokens_to_ids(biling_dict[t]) if t in biling_dict.keys() else -100 for t in tokens]
        # masked_labels[0]=bert_input[0]
        # masked_labels[-1]=bert_input[-1]
        if all([x == -1 for x in masked_labels]):
            continue
        input_sequence.append(bert_input)
        masked_labels_seq.append(masked_labels)

    return input_sequence, masked_labels_seq

print('Train')
train_input_sequence, train_masked_labels_seq = create_bert_tokens(sentence_list_train)
print('Val')
val_input_sequence, val_masked_labels_seq = create_bert_tokens(sentence_list_val)
print('Test')
test_input_sequence, test_masked_labels_seq = create_bert_tokens(sentence_list_test)

train = (train_input_sequence, train_masked_labels_seq)
val = (val_input_sequence, val_masked_labels_seq)
test = (test_input_sequence, test_masked_labels_seq)

pickle.dump(train, open('data/train_' + args.src + '_' + args.tgt , 'wb'))
pickle.dump(val, open('data/val_' + args.src + '_' + args.tgt , 'wb'))
pickle.dump(test, open('data/test_' + args.src + '_' + args.tgt , 'wb'))

print(train_input_sequence[:20])
print(train_masked_labels_seq[:20])
