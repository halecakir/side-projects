#!/usr/bin/env python
# coding: utf-8

# In[28]:


import sys
import os
import csv
import random

import pickle
import scipy
import pandas as pd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
from torch import optim

from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

seed = 10

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class ArgumentParser:
    permission_type = "RECORD_AUDIO"
    saved_data = "/home/huseyinalecakir/Security/data/saved-parameters/saved-data/emdeddings-sentences-w2i.pickle"
    saved_predicted_reviews = "/home/huseyinalecakir/Security/data/saved-parameters/saved-data/predicted-RECORD_AUDIO-reviews.pickle"
    useful_reviews = 5
    outdir = "output.out"

class SentenceReport:
    def __init__(self, id, sentence):
        self.app_id = id
        self.sentence = sentence
        self.permissions = {}
        self.preprocessed_sentence = None
        self.prediction_result = None
        self.index_tensor = None


class Review:
    def __init__(self, sentence, score):
        self.sentence = sentence
        self.preprocessed_sentence = None
        self.score = score
        self.index_tensor = None
        self.prediction_result = None

class TorchOptions:
    hidden_size = 300
    init_weight = 0.08
    output_size = 1
    print_every = 1000
    grad_clip = 5
    dropout = 0
    dropoutrec = 0
    learning_rate_decay = 1  # 0.985
    learning_rate_decay_after = 1

class Data:
    def __init__(self):
        self.w2i = None
        self.entries = None
        self.train_entries = None
        self.test_entries = None
        self.ext_embedding = None
        self.predicted_reviews = None

    def load(self, infile):
        with open(infile, "rb") as target:
            self.ext_embeddings, self.entries, self.w2i = pickle.load(target)

    def load_predicted_reviews(self, infile):
        with open(infile, "rb") as target:
            self.predicted_reviews = pickle.load(target)
        for app_id in self.predicted_reviews.keys():
            self.predicted_reviews[app_id].sort(key=lambda x: x.prediction_result.item(), reverse=True)

class Encoder(nn.Module):
    def __init__(self, opt, w2i):
        super(Encoder, self).__init__()
        self.opt = opt
        self.w2i = w2i

        self.lstm = nn.LSTM(
            self.opt.hidden_size, self.opt.hidden_size, batch_first=True
        )
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        self.embedding = nn.Embedding(len(self.w2i), self.opt.hidden_size)
        self.__initParameters()

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.uniform_(param, -self.opt.init_weight, self.opt.init_weight)

    def initalizedPretrainedEmbeddings(self, embeddings):
        weights_matrix = np.zeros(((len(self.w2i), self.opt.hidden_size)))
        for word in self.w2i:
            weights_matrix[self.w2i[word]] = embeddings[word]
        self.embedding.weight = nn.Parameter(torch.FloatTensor(weights_matrix))

    def forward(self, input_src):
        src_emb = self.embedding(input_src)  # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        outputs, (h, c) = self.lstm(src_emb)
        return outputs, (h, c)

class Classifier(nn.Module):
    def __init__(self, opt):
        super(Classifier, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.linear = nn.Linear(2*self.hidden_size, opt.output_size)

        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.sigmoid = nn.Sigmoid()
        self.__initParameters()

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.uniform_(param, -self.opt.init_weight, self.opt.init_weight)

    def forward(self, prev_h):
        if self.opt.dropout > 0:
            prev_h = self.dropout(prev_h)
        h2y = self.linear(prev_h)
        pred = self.sigmoid(h2y)
        return pred

class Model:
    def __init__(self):
        self.opt = None
        self.encoders = {}
        self.review_encoder = None
        self.classifier = None
        self.optimizer = None
        self.criterion = None

    def create(self, opt, data):
        self.opt = opt
        self.encoders["sentence"] = Encoder(self.opt, data.w2i)
        self.encoders["reviewL1"] = Encoder(self.opt, data.w2i)
        self.encoders["reviewL2"] =  nn.LSTMCell(opt.hidden_size, opt.hidden_size)
        params = []
        for encoder in self.encoders:
            params += list(self.encoders[encoder].parameters())
        self.classifier = Classifier(self.opt)
        params += list(self.classifier.parameters())
        self.optimizer = optim.Adam(params)
        self.criterion = nn.BCELoss()

    def train(self):
        for encoder in self.encoders:
            self.encoders[encoder].train()
        self.classifier.train()

    def eval(self):
        for encoder in self.encoders:
            self.encoders[encoder].eval()
        self.classifier.eval()

    def step(self):
        self.optimizer.step()

    def zero_grad(self):
        self.optimizer.zero_grad()

    def grad_clip(self):
        for encoder in self.encoders:
            torch.nn.utils.clip_grad_value_(self.encoders[encoder].parameters(), self.opt.grad_clip)
            self.encoders[encoder].train()
        torch.nn.utils.clip_grad_value_(self.classifier.parameters(), self.opt.grad_clip)

    def save(self, filename):
        checkpoint = {}
        checkpoint["opt"] = self.opt
        for encoder in self.encoders:
            checkpoint[encoder] = self.encoders[encoder].state_dict()
        checkpoint["classifier"] = self.classifier.state_dict()
        checkpoint["optimizer"] = self.optimizer.state_dict()
        torch.save(checkpoint, filename)

    def load(self, filename, data):
        checkpoint = torch.load(filename)
        opt = checkpoint["opt"]
        self.create(opt, data)
        for encoder in self.encoders:
            self.encoders[encoder].load_state_dict(checkpoint[encoder])
        self.decoder.load_state_dict(checkpoint["classifier"])
        self.optimizer.load_state_dict(checkpoint["optimizer"])

def write_file(filename, string):
    with open(filename, "a") as target:
        target.write("{}\n".format(string))
        target.flush()

def train_item(args, model, sentence, reviews):
    model.zero_grad()
    outputs_s, (hidden_s, cell_s) = model.encoders["sentence"](sentence.index_tensor)

    hidden_r_lst = []
    for review in reviews:
        outputs_r, (hidden_r, cell_r) = model.encoders["reviewL1"](review.index_tensor)
        hidden_r_lst.append(hidden_r)

    hidden_rl2, cell_rl2 = None, None
    for hidden_r in hidden_r_lst:
        hidden_rl2, cell_rl2 = model.encoders["reviewL2"](hidden_r.view(1,-1))

    hidden = torch.cat((hidden_s, hidden_rl2.view(1, 1,-1)), 2)
    pred = model.classifier(hidden)
    loss = model.criterion(
        pred,
        torch.tensor(
            [[[sentence.permissions[args.permission_type]]]], dtype=torch.float
        ),
    )
    loss.backward()

    if model.opt.grad_clip != -1:
        model.grad_clip()
    model.step()
    return loss

def test_item(model, sentence, reviews):
    outputs_s, (hidden_s, cell_s) = model.encoders["sentence"](sentence.index_tensor)

    hidden_r_lst = []
    for review in reviews:
        outputs_r, (hidden_r, cell_r) = model.encoders["reviewL1"](review.index_tensor)
        hidden_r_lst.append(hidden_r)

    hidden_rl2, cell_rl2 = None, None
    for hidden_r in hidden_r_lst:
        hidden_rl2, cell_rl2 = model.encoders["reviewL2"](hidden_r.view(1,-1))

    hidden = torch.cat((hidden_s, hidden_rl2.view(1, 1,-1)), 2)
    pred = model.classifier(hidden)
    return pred

def train_all(args, model, data):
    write_file(args.outdir, "Training...")

    model.train()
    losses = []
    for index, sentence in enumerate(data.train_entries):
        if sentence.app_id in data.predicted_reviews:
            loss = train_item(args, model, sentence, data.predicted_reviews[sentence.app_id][:args.useful_reviews])
        if index != 0:
            if index % model.opt.print_every == 0:
                write_file(args.outdir,
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - model.opt.print_every :])
                    )
                )
        losses.append(loss.item())


def test_all(args, model, data):
    def pr_roc_auc(predictions, gold):
        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        return roc_auc, pr_auc

    write_file(args.outdir, "Predicting..")

    predictions, gold = [], []
    model.eval()
    with torch.no_grad():
        for index, sentence in enumerate(data.test_entries):
            if sentence.app_id in data.predicted_reviews:
                pred = test_item(model, sentence, data.predicted_reviews[sentence.app_id][:args.useful_reviews])
                predictions.append(pred)
                gold.append(sentence.permissions[args.permission_type])
    return pr_roc_auc(predictions, gold)

def kfold_validation(args, opt, data):
    data.entries = np.array(data.entries)
    random.shuffle(data.entries)

    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
    roc_l, pr_l = [], []
    for foldid, (train, test) in enumerate(kfold.split(data.entries)):
        write_file(args.outdir, "Fold {}".format(foldid+1))

        model = Model()
        model.create(opt, data)
        data.train_entries = data.entries[train]
        data.test_entries = data.entries[test]

        train_all(args, model, data)
        roc_auc, pr_auc = test_all(args, model, data)

        write_file(args.outdir, "ROC {} PR {}".format(roc_auc, pr_auc))
        roc_l.append(roc_auc)
        pr_l.append(pr_auc)
    write_file(args.outdir, "Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l)))


# In[24]:

def run(args):
    opt = TorchOptions()

    data = Data()
    data.load(args.saved_data)
    data.load_predicted_reviews(args.saved_predicted_reviews)

    kfold_validation(args, opt, data)
