#!/usr/bin/env python
# coding: utf-8

# In[1]:


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
from common import *

from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import KFold

seed = 10

random.seed(seed)
torch.manual_seed(seed)
np.random.seed(seed)


class ArgumentParser:
    permission_type = sys.argv[1]
    train = "/home/huseyinalecakir/Security/data/acnet-data/ACNET_DATASET.csv"
    train_file_type = "acnet"
    external_embedding = "/home/huseyinalecakir/Security/data/pretrained-embeddings/{}".format(
        "scraped_with_porter_stemming_300.bin"
    )
    external_embedding_type = "word2vec"
    stemmer = "porter"
    saved_parameters_dir = "/home/huseyinalecakir/Security/data/saved-parameters/"
    saved_prevectors = "embeddings.pickle"
    saved_vocab_train = "acnet-vocab.txt"
    saved_all_data = "{}/all_data".format(saved_parameters_dir)
    reviews = "/home/huseyinalecakir/Security/data/reviews/acnet-reviews/acnet_initial/app_reviews_original.csv"
    lower = True
    outdir = "./test/{}".format(permission_type)


class TorchOptions:
    d_rnn_size = 300
    r_rnn_size = 0
    init_weight = 0.08
    decay_rate = 0.985
    learning_rate = 0.0001
    plot_every = 2500
    print_every = 2500
    grad_clip = 5
    dropout = 0
    dropoutrec = 0
    learning_rate_decay = 0.985
    learning_rate_decay_after = 1


class LSTM(nn.Module):
    def __init__(self, opt, hidden_size):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(hidden_size, 4 * hidden_size)
        self.h2h = nn.Linear(hidden_size, 4 * hidden_size)
        if opt.dropoutrec > 0:
            self.dropout = nn.Dropout(opt.dropoutrec)

    def forward(self, x, prev_c, prev_h):
        gates = self.i2h(x) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        if self.opt.dropoutrec > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
        return cy, hy


class Encoder(nn.Module):
    def __init__(self, opt, w2i, ext_embeddings, hidden_size):
        super(Encoder, self).__init__()
        self.opt = opt
        self.w2i = w2i
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(len(w2i), self.hidden_size)
        self.lstm = LSTM(self.opt, hidden_size)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        self.__initParameters()
        self.__initalizedPretrainedEmbeddings(ext_embeddings)

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.uniform_(param, -self.opt.init_weight, self.opt.init_weight)

    def __initalizedPretrainedEmbeddings(self, embeddings):
        weights_matrix = np.zeros(((len(self.w2i), self.hidden_size)))

        for word in self.w2i:
            weights_matrix[self.w2i[word]] = embeddings[word]
        self.embedding.from_pretrained(torch.FloatTensor(weights_matrix))

    def forward(self, input_src, prev_c, prev_h):
        src_emb = self.embedding(input_src)  # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        prev_cy, prev_hy = self.lstm(src_emb, prev_c, prev_h)
        return prev_cy, prev_hy


class Classifier(nn.Module):
    def __init__(self, opt, output_size):
        super(Classifier, self).__init__()
        self.opt = opt
        self.hidden_size = opt.d_rnn_size + opt.r_rnn_size
        self.linear = nn.Linear(self.hidden_size, output_size)

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


def train_item(opt, args, sentence, encoder, classifier, optimizer, criterion):
    optimizer.zero_grad()
    c = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    h = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    for i in range(sentence.index_tensor.size(1)):
        c, h = encoder(sentence.index_tensor[:, i], c, h)

    pred = classifier(h)
    loss = criterion(
        pred,
        torch.tensor([[sentence.permissions[args.permission_type]]], dtype=torch.float),
    )
    loss.backward()
    if opt.grad_clip != -1:
        torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(classifier.parameters(), opt.grad_clip)
    optimizer.step()
    return loss


def predict(opt, sentence, encoder, classifier):
    c = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    h = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)

    for i in range(sentence.index_tensor.size(1)):
        c, h = encoder(sentence.index_tensor[:, i], c, h)
    pred = classifier(h)
    return pred


def train_and_test(opt, args, epoch_num, w2i, train_data, test_data, foldid):
    encoder = Encoder(opt, w2i, ext_embeddings, opt.d_rnn_size)
    classifier = Classifier(opt, 1)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params)
    optim_state = {"learningRate": opt.learning_rate, "alpha": opt.decay_rate}
    criterion = nn.BCELoss()

    pr_scores = []
    roc_scores = []
    losses = []

    for epoch in range(epoch_num):
        print("---Epoch {}---\n".format(epoch + 1))

        print("Training...")
        encoder.train()
        classifier.train()
        for index, sentence in enumerate(train_data):
            loss = train_item(
                opt, args, sentence, encoder, classifier, optimizer, criterion
            )
            if index != 0:
                if index % opt.print_every == 0:
                    print(
                        "Index {} Loss {}".format(
                            index,
                            np.mean(
                                losses[
                                    epoch * len(train_data) + index - opt.print_every :
                                ]
                            ),
                        )
                    )
            losses.append(loss.item())

        # Learning Rate Decay Optimization
        if opt.learning_rate_decay < 1:
            if epoch >= opt.learning_rate_decay_after:
                decay_factor = opt.learning_rate_decay
                optim_state["learningRate"] = optim_state["learningRate"] * decay_factor
                for param_group in optimizer.param_groups:
                    param_group["lr"] = optim_state["learningRate"]

        print("Predicting..")
        encoder.eval()
        classifier.eval()
        predictions = []
        gold = []
        with torch.no_grad():
            for index, sentence in enumerate(test_data):
                pred = predict(opt, sentence, encoder, classifier)
                predictions.append(pred)
                gold.append(sentence.permissions[args.permission_type])

        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        pr_scores.append(pr_auc)
        roc_scores.append(roc_auc)
        print("Scores ROC {} PR {}".format(roc_auc, pr_auc))

        # Save Model
        model_save_dir = os.path.join(
            args.saved_parameters_dir, "models", "model_for_review_prediction.pt"
        )
        if not os.path.exists(os.path.dirname(model_save_dir)):
            os.makedirs(os.path.dirname(model_save_dir))
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "classifier": classifier.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
                "loss": loss,
                "foldid": foldid,
                "pr_auc": pr_auc,
                "roc_auc": roc_auc,
            },
            model_save_dir,
        )
    return roc_scores, pr_scores


def load_model_and_predict_reviews(args, model_path, reviews):
    encoder = Encoder(opt, w2i, ext_embeddings)
    classifier = Classifier(opt, 1)

    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])

    pr_auc = checkpoint["pr_auc"]
    roc_auc = checkpoint["roc_auc"]
    print(pr_auc, roc_auc)
    with torch.no_grad():
        for app_id in reviews:
            for review in reviews[app_id]:
                pred = predict(opt, review, encoder, classifier)
                review.prediction_result = pred


args = ArgumentParser()
opt = TorchOptions()


save_dir = os.path.join(args.saved_all_data, "without_prediction.pickle")
ext_embeddings, reviews, sentences, w2i = load_all_data(save_dir)

documents = np.array(sentences)
random.shuffle(documents)

train_and_test(opt, args, 1, w2i, documents, documents, 0)

model_save_dir = os.path.join(
    args.saved_parameters_dir, "models", "model_for_review_prediction.pt"
)
load_model_and_predict_reviews(args, model_save_dir, reviews)

save_dir = os.path.join(
    args.saved_all_data, args.permission_type, "with_prediction.pickle"
)
save_all_data(save_dir, ext_embeddings, reviews, sentences, w2i)
