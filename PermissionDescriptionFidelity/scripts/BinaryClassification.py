#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sys
import os
import csv
import random

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


# In[2]:


get_ipython().run_line_magic("load_ext", "autoreload")
get_ipython().run_line_magic("autoreload", "2")


class ArgumentParser:
    permission_type = "READ_CONTACTS"
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
    lower = True
    outdir = "./test/{}".format(permission_type)


# In[3]:


class TorchOptions:
    def __init__(self):
        self.rnn_size = 300
        self.init_weight = 0.08
        self.decay_rate = 0.985
        self.learning_rate = 0.0001
        self.plot_every = 2500
        self.print_every = 2500
        self.grad_clip = 5
        self.dropout = 0
        self.dropoutrec = 0
        self.learning_rate_decay = 0.985
        self.learning_rate_decay_after = 1


# In[4]:


class SentenceReport:
    """TODO"""

    def __init__(self, id, sentence, mark):
        self.app_id = id
        self.mark = mark
        self.preprocessed_sentence = None
        self.sentence = sentence
        self.prediction_result = None
        self.index_tensors = None


# In[5]:


def load_row_acnet(infile, gold_permission, stemmer, embeddings):
    print("Loading row {} ".format(infile))
    # read training data
    print("Reading Train Sentences")
    tagged_train_file = pd.read_csv(infile)
    train_sententence_reports = []
    acnet_map = {
        "RECORD_AUDIO": "MICROPHONE",
        "READ_CONTACTS": "CONTACTS",
        "READ_CALENDAR": "CALENDAR",
        "ACCESS_FINE_LOCATION": "LOCATION",
        "CAMERA": "CAMERA",
        "READ_SMS": "SMS",
        "READ_CALL_LOGS": "CALL_LOG",
        "CALL_PHONE": "PHONE",
        "WRITE_SETTINGS": "SETTINGS",
        "GET_TASKS": "TASKS",
    }
    for idx, row in tagged_train_file.iterrows():
        app_id = row["app_id"]
        sentence = row["sentence"]
        mark = row[acnet_map[gold_permission]]
        sentence_report = SentenceReport(app_id, sentence, mark)
        preprocessed = NLPUtils.preprocess_sentence(sentence_report.sentence, stemmer)
        sentence_report.preprocessed_sentence = [
            word for word in preprocessed if word in embeddings
        ]
        train_sententence_reports.append(sentence_report)
    print("Loading completed")
    return train_sententence_reports


# In[6]:


def load_acnet_app_ids(infile):
    app_ids = []
    with open(APPLIST_FILE) as target:
        for line in target:
            app_ids.append(line.strip())


# In[7]:


class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)
        self.h2h = nn.Linear(opt.rnn_size, 4 * opt.rnn_size)
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


# In[8]:


class Encoder(nn.Module):
    def __init__(self, opt, w2i, ext_embeddings):
        super(Encoder, self).__init__()
        self.opt = opt
        self.w2i = w2i
        self.hidden_size = opt.rnn_size

        self.embedding = nn.Embedding(len(w2i), self.hidden_size)
        self.lstm = LSTM(self.opt)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        self.__initParameters()
        self.__initalizedPretrainedEmbeddings(ext_embeddings)

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.uniform_(param, -opt.init_weight, opt.init_weight)

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


# In[9]:


class Classifier(nn.Module):
    def __init__(self, opt, output_size):
        super(Classifier, self).__init__()
        self.opt = opt
        self.hidden_size = opt.rnn_size
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


# In[10]:


def create_index_tensors(data, w2i):
    for sentence_report in data:
        sentence_report.index_tensor = torch.zeros(
            (1, len(sentence_report.preprocessed_sentence)), dtype=torch.long
        )
        for idx, word in enumerate(sentence_report.preprocessed_sentence):
            sentence_report.index_tensor[0][idx] = w2i[word]


# In[11]:


def load_embeddings(options):
    if options.external_embedding is not None:
        if os.path.isfile(
            os.path.join(options.saved_parameters_dir, options.saved_prevectors)
        ):
            ext_embeddings, _ = IOUtils.load_embeddings_file(
                os.path.join(options.saved_parameters_dir, options.saved_prevectors),
                "pickle",
                options.lower,
            )
            return ext_embeddings
        else:
            ext_embeddings, _ = IOUtils.load_embeddings_file(
                options.external_embedding,
                options.external_embedding_type,
                options.lower,
            )
            IOUtils.save_embeddings(
                os.path.join(options.saved_parameters_dir, options.saved_prevectors),
                ext_embeddings,
            )
            return ext_embeddings
    else:
        raise Exception("external_embedding option is None")


# In[12]:


def trainItem(opt, sentence, encoder, classifier, optimizer, criterion):
    optimizer.zero_grad()
    c = torch.zeros((1, opt.rnn_size), dtype=torch.float, requires_grad=True)
    h = torch.zeros((1, opt.rnn_size), dtype=torch.float, requires_grad=True)
    for i in range(sentence.index_tensor.size(1)):
        c, h = encoder(sentence.index_tensor[:, i], c, h)

    pred = classifier(h)
    loss = criterion(pred, torch.tensor([[sentence.mark]], dtype=torch.float))
    loss.backward()
    if opt.grad_clip != -1:
        torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(classifier.parameters(), opt.grad_clip)
    optimizer.step()

    return loss


# In[13]:


def predict(opt, sentence, encoder, classifier):
    c = torch.zeros((1, opt.rnn_size), dtype=torch.float, requires_grad=True)
    h = torch.zeros((1, opt.rnn_size), dtype=torch.float, requires_grad=True)

    for i in range(sentence.index_tensor.size(1)):
        c, h = encoder(sentence.index_tensor[:, i], c, h)

    pred = classifier(h)
    return pred


# In[14]:


args = ArgumentParser()
opt = TorchOptions()


# In[18]:


get_ipython().run_cell_magic("time", "", "ext_embeddings = load_embeddings(args)")


# In[19]:


get_ipython().run_cell_magic(
    "time",
    "",
    "print('Extracting training vocabulary')\nw2i = IOUtils.load_vocab(args.train, args.train_file_type, args.saved_parameters_dir, args.saved_vocab_train,\n                                            args.external_embedding,\n                                            args.external_embedding_type,\n                                            args.stemmer,\n                                            args.lower)",
)


# In[20]:


get_ipython().run_cell_magic(
    "time",
    "",
    "sentences = load_row_acnet(args.train, args.permission_type, args.stemmer, ext_embeddings)",
)


# In[21]:


get_ipython().run_cell_magic("time", "", "create_index_tensors(sentences, w2i)")


# In[22]:


def train_and_test(opt, args, epoch_num, w2i, train_sentences, test_sentences, foldid):
    encoder = Encoder(opt, w2i, ext_embeddings)
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
            loss = trainItem(opt, sentence, encoder, classifier, optimizer, criterion)
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
                gold.append(sentence.mark)

        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        pr_scores.append(pr_auc)
        roc_scores.append(roc_auc)
        print("Scores ROC {} PR {}".format(roc_auc, pr_auc))

        # Save Model
        model_save_dir = os.path.join(
            args.saved_parameters_dir,
            "models",
            "fold{0}.epoch{1}.roc_auc{2:.2f}.prauc{3:.2f}.pt".format(
                foldid, epoch, roc_auc, pr_auc
            ),
        )
        print(model_save_dir)
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


# In[23]:


def load_model_and_predict(model_path, sentences):
    documents = np.array(sentences)
    documents = documents[:100]
    kfold = KFold(n_splits=2, shuffle=True, random_state=seed)
    kfold_splits = kfold.split(documents)

    encoder = Encoder(opt, w2i, ext_embeddings)
    classifier = Classifier(opt, 1)
    optimizer = optim.Adam(params)
    optim_state = {"learningRate": opt.learning_rate, "alpha": opt.decay_rate}
    criterion = nn.BCELoss()

    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    pr_auc = checkpoint["pr_auc"]
    foldid = checkpoint["foldid"]
    roc_auc = checkpoint["roc_auc"]
    epoch = checkpoint["epoch"]

    train_indexes, test_indexes = kfold_splits[foldid]
    test_data = documents[test_indexes]
    with torch.no_grad():
        for index, sentence in enumerate(test_data):
            pred = predict(opt, sentence, encoder, classifier)
            predictions.append(pred)
            gold.append(sentence.mark)
            y_true = np.array(gold)

    y_scores = np.array(predictions)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    pr_scores.append(pr_auc)
    roc_scores.append(roc_auc)
    print("Scores ROC {} PR {}".format(roc_auc, pr_auc))


# In[24]:


get_ipython().run_cell_magic(
    "time",
    "",
    'documents = np.array(sentences)\nkfold = KFold(n_splits=10, shuffle=True, random_state=seed)\nkfold_splits = kfold.split(documents)\n\n\nchunkend_losses = []\nroc_scores = []\npr_scores = [] \nfor foldid, (train, test) in enumerate(kfold_splits):\n    print("\\nFOLD ID {}\\n".format(foldid+1))\n    train_data = documents[train]\n    test_data = documents[test]\n\n    roc, pr = train_and_test(opt, args, 3, w2i, train_data, test_data, foldid)\n    print("Fold Results :\\n")\n    print("ROC : {}\\n".format(roc))\n    print("PR : {}\\n".format(pr))\n    \n    roc_scores.append(roc)\n    pr_scores.append(pr)',
)
