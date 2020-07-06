import torch
from torchtext import data
from torchtext import datasets
import random
import numpy as np
import spacy
import re
import os
import codecs
from torch import nn
import pickle



class SentimentLSTM(nn.Module):
    
    def __init__(self, n_vocab, n_embed, n_hidden, n_output, n_layers, drop_p = 0.5):
        super().__init__()
        # params: "n_" means dimension
        self.n_vocab = n_vocab     # number of unique words in vocabulary
        self.n_layers = n_layers   # number of LSTM layers 
        self.n_hidden = n_hidden   # number of hidden nodes in LSTM
        
        self.embedding = nn.Embedding(n_vocab + 1, n_embed)#.cuda()
        self.lstm = nn.LSTM(n_embed, n_hidden, n_layers, batch_first = True, dropout = drop_p)#.cuda()
        self.dropout = nn.Dropout(drop_p)#.cuda()
        self.fc = nn.Linear(n_hidden, n_output)#.cuda()
        self.sigmoid = nn.Sigmoid()#.cuda()
        
        
    def forward (self, input_words):
                                             # INPUT   :  (batch_size, seq_length)
        embedded_words = self.embedding(input_words)    # (batch_size, seq_length, n_embed)
        lstm_out, h = self.lstm(embedded_words)         # (batch_size, seq_length, n_hidden)
        lstm_out = self.dropout(lstm_out)
        lstm_out = lstm_out.contiguous().view(-1, self.n_hidden) # (batch_size*seq_length, n_hidden)
        fc_out = self.fc(lstm_out)                      # (batch_size*seq_length, n_output)
        sigmoid_out = self.sigmoid(fc_out)              # (batch_size*seq_length, n_output)
        sigmoid_out = sigmoid_out.view(batch_size, -1)  # (batch_size, seq_length*n_output)
        
        # extract the output of ONLY the LAST output of the LAST element of the sequence
        sigmoid_last = sigmoid_out[:, -1]               # (batch_size, 1)
        
        return sigmoid_last, h
    
    
    def init_hidden (self, batch_size):  # initialize hidden weights (h,c) to 0
        
        device = 'cpu' #"cuda" if torch.cuda.is_available() else "cpu"
        weights = next(self.parameters()).data
        h = (weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device),
             weights.new(self.n_layers, batch_size, self.n_hidden).zero_().to(device))
        
        return h
import re
from string import punctuation
_patterns = [r'\"',
             r'<br /><br />',
             r'\;',
             r'\:',
             r'\s+',
             r'\(',
             r'\)']

_replacements = ['',
                 ' ',
                 ' ',
                 ' ',
                 ' ',
                '',
                '']
_patterns_dict = list((re.compile(p), r) for p, r in zip(_patterns, _replacements))

def normalize(line):
    line = line.lower()
    for pattern_re, replaced_str in _patterns_dict:
        line = pattern_re.sub(replaced_str, line)
    return line

TOKEN_RE = re.compile(r'[a-z]+|\d+[.,]\d+|\d+')

def tokenize(txt, min_token_size = 3):
    txt = normalize(txt)
    all_tokens = TOKEN_RE.findall(txt)
    return [token for token in all_tokens if len(token) >= min_token_size]

def tokenize_corpus(texts, tokenizer=tokenize, **tokenizer_kwargs):
    return [tokenizer(text, **tokenizer_kwargs) for text in texts]

def preprocess(text):
    text = [normalize(texti) for texti in text]
    all_reviews = tokenize_corpus(text, min_token_size = 3)
    text = " ".join(text)
    all_words = tokenize(text, min_token_size = 3)
    
    return all_reviews, all_words

def pad_text(encoded_reviews, seq_length):
    
    reviews = []
    
    for review in encoded_reviews:
        if len(review) >= seq_length:
            reviews.append(review[:seq_length])
        else:
            reviews.append([0]*(seq_length-len(review)) + review)
        
    return np.array(reviews)

def predict(net, review, seq_length = 200):
    device = 'cpu' 
    
    words, _ = preprocess([review])
    encoded_words = [vocab_to_int[word] for word in words[0] if word in vocab_to_int.keys()]
    padded_words = pad_text([encoded_words], seq_length)
    padded_words = torch.from_numpy(padded_words).to(device)
    
    if(len(padded_words) == 0):
        "Your review must contain at least 1 word!"
        return None
    
    net.eval()
    h = net.init_hidden(1)
    padded_words = torch.Tensor(padded_words.float()).long()
    output, h = net(padded_words)
    pred = output.squeeze()
    
    return pred

def rating_pred(out):
    if out > .5:
        return round(7 + 6 * (out - 0.5))
    else:
        return round(4 - 6 * (-out + 0.5))


batch_size = 50
n_vocab = 104199
n_embed = 32
n_hidden = 8
n_output = 1
n_layers = 2
model = SentimentLSTM(n_vocab, n_embed, n_hidden, n_output, n_layers)
model.load_state_dict(torch.load('net.pth'))
pkl_file = open('vocab_to_int.pkl', 'rb')
vocab_to_int = pickle.load(pkl_file)
pkl_file.close()

