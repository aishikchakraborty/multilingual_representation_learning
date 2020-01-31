import numpy as np
import random
import os
import csv
import json
import time
import argparse
import _pickle as pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
import wandb
from transformers import *

# wandb login 75896b1ec51fe81d94a454defeb033dd0180a941

parser = argparse.ArgumentParser(description='Generation Project Main')
parser.add_argument('--data', type=str, default='data/', help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM', help='Model type')
parser.add_argument('--encoder-type', type=str, default='bert', help='type of encoder')
parser.add_argument('--decoder-type', type=str, default='bert', help='type of decoder')
parser.add_argument('--random_seed', type=int, default=13370, help='random seed')
parser.add_argument('--numpy_seed', type=int, default=1337, help='numpy random seed')
parser.add_argument('--torch_seed', type=int, default=133, help='pytorch random seed')
parser.add_argument('--lr', type=float, default=2e-5, help='Learning rate')
parser.add_argument('--grad-clip', type=float, default=0.25, help='Gradient clipping')
parser.add_argument('--batch-size', type=int, default=20, help='Batch Size')
parser.add_argument('--epochs', type=int, default=5, help='Epochs')
parser.add_argument('--hidden-dim', type=int, default=100, help='Hidden Dimensions of the decoder')
parser.add_argument('--num-layers', type=int, default=1, help='Number of layers of the sequence model')
parser.add_argument('--gpu', type=int, default=0, help='GPU id')
parser.add_argument('--embed-dim', type=int, default=300, help='Decoder embedding dimensions')
parser.add_argument('--log-interval', type=int, default=40, help='Print interval')
parser.add_argument('--cuda', action='store_true', help='use CUDA')
parser.add_argument('--num-accumulations', type=int, default=10, help='Number of gradient accumulations')
parser.add_argument('--src', type=str, default='en', help='location of the data corpus')
parser.add_argument('--tgt', type=str, default='de', help='location of the data corpus')


args = parser.parse_args()

print(args)
# tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased', do_lower_case=True)

if args.random_seed is not None:
    random.seed(args.random_seed)
if args.numpy_seed is not None:
    np.random.seed(args.numpy_seed)
if args.torch_seed is not None:
    torch.manual_seed(args.torch_seed)
    # Seed all GPUs with the same seed if available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.torch_seed)

if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

device = torch.device("cuda:" + str(args.gpu) if args.cuda else "cpu")

train = pickle.load(open('data/train_' + args.src + '_' + args.tgt , 'rb'))
val = pickle.load(open('data/val_' + args.src + '_' + args.tgt , 'rb'))
test = pickle.load(open('data/test_' + args.src + '_' + args.tgt , 'rb'))

train_input_seq , train_masked_lm_seq = train[0], train[1]
val_input_seq , val_masked_lm_seq = train[0], train[1]
test_input_seq , test_masked_lm_seq = train[0], train[1]


def pad_sequences(s, pad_token):
    lengths = [len(s1) for s1 in s]
    longest_sent = max(lengths)

    padded_X = np.ones((args.batch_size, longest_sent), dtype=np.int64) * pad_token
    masked_X = np.ones((args.batch_size, longest_sent), dtype=np.int64)
    for i, x_len in enumerate(lengths):
        sequence = s[i]
        padded_X[i, 0:x_len] = sequence[:x_len]
        masked_X[i, 0:x_len] = [0]*x_len   
    # print(padded_X)
    return padded_X, masked_X

vocab_bert = []
f = open(os.path.join(args.data, 'wikitext-103', 'vocab.txt'))
for lines in f:
    vocab_bert.append(lines.strip())
w2idx = {w:idx for idx, w in enumerate(vocab_bert)}

model = BertForMaskedLM.from_pretrained('bert-base-multilingual-uncased')
model.eval()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

def evaluate(data):
    val_loss = 0
    data_input, data_masked = data[0], data[1]
    num_batches = len(data_input)//args.batch_size
    with torch.no_grad():
        for i in tqdm(range(num_batches)):
            batch_input, attn_mask = pad_sequences(data_input[i*args.batch_size:(i+1)*args.batch_size], w2idx['[PAD]'])
            batch_masked_outputs, _ = pad_sequences(data_masked[i*args.batch_size:(i+1)*args.batch_size], w2idx['[PAD]'])

            batch_input = torch.LongTensor(batch_input).to(device)
            attn_mask = torch.LongTensor(attn_mask).to(device)
            batch_masked_outputs = torch.LongTensor(batch_masked_outputs).to(device)

            outputs = model(batch_input, attention_mask=attn_mask, masked_lm_labels=batch_masked_outputs)
            loss, prediction_scores = outputs[:2]
            val_loss += loss.item()
            
    return val_loss/num_batches


for epoch in range(args.epochs):
    train_loss = 0
    best_val_loss = 1000000
    num_batches = len(train_input_seq)//args.batch_size
    for i in tqdm(range(num_batches)):
        batch_input, attn_mask = pad_sequences(train_input_seq[i*args.batch_size:(i+1)*args.batch_size], w2idx['[PAD]'])
        batch_masked_outputs, _ = pad_sequences(train_masked_lm_seq[i*args.batch_size:(i+1)*args.batch_size], w2idx['[PAD]'])

        batch_input = torch.LongTensor(batch_input).to(device)
        attn_mask = torch.LongTensor(attn_mask).to(device)
        batch_masked_outputs = torch.LongTensor(batch_masked_outputs).to(device)

        optimizer.zero_grad()
        outputs = model(batch_input, attention_mask=attn_mask, masked_lm_labels=batch_masked_outputs)
        loss, prediction_scores = outputs[:2]
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

        if i%args.log_interval == 0:
            message = "Train epoch: %d  iter: %d train MLM loss: %1.3f  " % (epoch, i, train_loss/args.log_interval)
            print(message)
            train_loss = 0
        
            val_loss = evaluate((val_input_seq , val_masked_lm_seq))
            message = "Val for Train epoch: %d  iter: %d val MLM loss: %1.3f  " % (epoch, i, val_loss)
            print(message)
            if best_val_loss > val_loss:
                best_val_loss = val_loss
                torch.save(model, 'models/model.pt')
        

model = torch.load('models/model.pt')
test_loss = evaluate((test_input_seq , test_masked_lm_seq))
message = "Test MLM loss: %1.3f  " % (test_loss)
print(message)
