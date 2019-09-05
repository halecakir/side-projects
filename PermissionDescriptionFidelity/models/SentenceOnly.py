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
    embedding_size = 300
    hidden_size = 300
    init_weight = 0.08
    output_size = 1
    print_every = 1000
    grad_clip = 5
    dropout = 0
    dropoutrec = 0
    learning_rate_decay = 1  # 0.985
    learning_rate_decay_after = 1


class Encoder(nn.Module):
    def __init__(self, opt, w2i, embeddings):
        super(Encoder, self).__init__()
        self.opt = opt
        self.w2i = w2i

        self.lstm = nn.LSTM(
            self.opt.embedding_size, self.opt.hidden_size, batch_first=True
        )
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)
        self.__initParameters()
        self.embedding = nn.Embedding.from_pretrained(
            self.__initalizedPretrainedEmbeddings(embeddings)
        )

    def __initParameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                init.uniform_(param, -self.opt.init_weight, self.opt.init_weight)

    def __initalizedPretrainedEmbeddings(self, embeddings):
        weights_matrix = np.zeros(((len(self.w2i), self.opt.hidden_size)))
        for word in self.w2i:
            weights_matrix[self.w2i[word]] = embeddings[word]
        return torch.FloatTensor(weights_matrix)

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
        self.linear = nn.Linear(self.hidden_size, opt.output_size)

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
    outputs, (hidden, cell) = encoder(sentence.index_tensor)

    pred = classifier(hidden)

    loss = criterion(
        pred,
        torch.tensor(
            [[[sentence.permissions[args.permission_type]]]], dtype=torch.float
        ),
    )
    loss.backward()

    if opt.grad_clip != -1:
        torch.nn.utils.clip_grad_value_(encoder.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(classifier.parameters(), opt.grad_clip)
    optimizer.step()
    return loss

def write_file(fd, string):
    fd.write("{}\n".format(string))
    fd.flush()

def predict(opt, sentence, encoder, classifier):
    outputs, (hidden, cell) = encoder(sentence.index_tensor)
    pred = classifier(hidden)
    return pred

def train_and_test(opt, args, w2i, train_data, test_data, ext_embeddings, file_outdir):
    def pr_roc_auc(predictions, gold):
        y_true = np.array(gold)
        y_scores = np.array(predictions)
        roc_auc = roc_auc_score(y_true, y_scores)
        pr_auc = average_precision_score(y_true, y_scores)
        return roc_auc, pr_auc

    encoder = Encoder(opt, w2i, ext_embeddings)
    classifier = Classifier(opt)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params)
    criterion = nn.BCELoss()

    losses = []
    write_file(file_outdir, "Training...")
    encoder.train()
    classifier.train()
    for index, sentence in enumerate(train_data):
        loss = train_item(
            opt, args, sentence, encoder, classifier, optimizer, criterion
        )
        if index != 0:
            if index % opt.print_every == 0:
                write_file(file_outdir,
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - opt.print_every :])
                    )
                )
        losses.append(loss.item())

    write_file(file_outdir, "Predicting..")
    encoder.eval()
    classifier.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for index, sentence in enumerate(test_data):
            pred = predict(opt, sentence, encoder, classifier)
            predictions.append(pred)
            gold.append(sentence.permissions[args.permission_type])
    return pr_roc_auc(predictions, gold)

def kfold_validation(args, opt, ext_embeddings, sentences, w2i, file_outdir):
    documents = np.array(sentences)
    random.shuffle(documents)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    roc_l, pr_l = [], []
    for foldid, (train, test) in enumerate(kfold.split(documents)):
        write_file(file_outdir, "Fold {}".format(foldid))
        train_data = documents[train]
        test_data = documents[test]
        roc, pr = train_and_test(opt, args, w2i, train_data, test_data, ext_embeddings, file_outdir)
        write_file(file_outdir, "ROC {} PR {}".format(roc, pr))
        roc_l.append(roc)
        pr_l.append(pr)
    write_file(file_outdir, "Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l)))

def train_with_all_data(args, opt, ext_embeddings, sentences, w2i, file_outdir, model_file):
    documents = np.array(sentences)
    random.shuffle(documents)
    train_data = documents

    write_file(file_outdir, "train_with_all_data")

    encoder = Encoder(opt, w2i, ext_embeddings)
    classifier = Classifier(opt)

    params = list(encoder.parameters()) + list(classifier.parameters())
    optimizer = optim.Adam(params)
    criterion = nn.BCELoss()

    losses = []
    write_file(file_outdir, "Training...")
    encoder.train()
    classifier.train()
    for index, sentence in enumerate(train_data):
        loss = train_item(
            opt, args, sentence, encoder, classifier, optimizer, criterion
        )
        if index != 0:
            if index % opt.print_every == 0:
                write_file(file_outdir,
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - opt.print_every :])
                    )
                )
        losses.append(loss.item())

    torch.save(
        {
            "encoder": encoder.state_dict(),
            "classifier": classifier.state_dict(),
            "optimizer": optimizer.state_dict()
        },
        model_file,
    )

def load_model_and_predict_reviews(opt, args, model_path, reviews, ext_embeddings, w2i):
    encoder = Encoder(opt, w2i, ext_embeddings)
    classifier = Classifier(opt)

    checkpoint = torch.load(model_path)
    encoder.load_state_dict(checkpoint["encoder"])
    classifier.load_state_dict(checkpoint["classifier"])

    with torch.no_grad():
        for app_id in reviews:
            for review in reviews[app_id]:
                pred = predict(opt, review, encoder, classifier)
                review.prediction_result = pred

def load_data(infile):
    with open(infile, "rb") as target:
        lst_of_objects = pickle.load(target)
        return lst_of_objects

def save_data(outfile, list_of_objects):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))
    with open(outfile, "wb") as target:
        pickle.dump(list_of_objects, target)

def run(args):
    opt = TorchOptions()

    file_outdir = open(args.outdir, "w")

    # File path
    data_file = os.path.join(args.saved_parameters_dir, args.saved_data)
    review_file = os.path.join(args.saved_parameters_dir, args.saved_reviews)
    model_file = os.path.join(args.saved_parameters_dir, args.model_checkpoint)
    predicted_reviews_file = os.path.join(args.saved_parameters_dir, args.saved_predicted_reviews)

    # Load data
    ext_embeddings, sentences, w2i = load_data(data_file)
    reviews = load_data(review_file)

    # Metrics calculation
    kfold_validation(args, opt, ext_embeddings, sentences, w2i, file_outdir)

    # Train with all and save model
    train_with_all_data(args, opt, ext_embeddings, sentences, w2i, file_outdir, model_file)

    # Load model and predict reviews
    load_model_and_predict_reviews(opt, args, model_file, reviews, ext_embeddings, w2i)
    save_data(predicted_reviews_file, reviews)
    file_outdir.close()