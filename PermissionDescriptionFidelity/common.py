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


def load_row_reviews(infile, stemmer, embeddings):
    print("Loading row {} ".format(infile))
    reviews = {}
    tagged_train_file = pd.read_csv(infile)
    for idx, row in tagged_train_file.iterrows():
        if idx != 0 and idx % 1000 == 0:
            print(idx)
        app_id, sentence, score = (
            row["application_id"],
            row["review_sentence"],
            row["score"],
        )
        if app_id and sentence and score:
            preprocessed = NLPUtils.preprocess_sentence(sentence, stemmer)
            if len(preprocessed) != 0:
                review = Review(sentence, score)
                if app_id not in reviews:
                    reviews[app_id] = []
                review.preprocessed_sentence = [
                    word for word in preprocessed if word in embeddings
                ]
                reviews[app_id].append(review)
    return reviews


def load_row_acnet(infile, stemmer, embeddings):
    print("Loading row {} ".format(infile))
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
        sentence_report = SentenceReport(app_id, sentence)

        for permission in acnet_map:
            sentence_report.permissions[permission] = row[acnet_map[permission]]

        preprocessed = NLPUtils.preprocess_sentence(sentence_report.sentence, stemmer)
        sentence_report.preprocessed_sentence = [
            word for word in preprocessed if word in embeddings
        ]

        train_sententence_reports.append(sentence_report)
    print("Loading completed")
    return train_sententence_reports


def reviews_vocab(reviews):
    vocab = set()
    for app_id in reviews:
        for review in reviews[app_id]:
            for token in review.preprocessed_sentence:
                vocab.add(token)
    return vocab


def create_index_tensors(sentences, reviews, w2i):
    def get_tensor(sequence, w2i):
        index_tensor = torch.zeros((1, len(sequence)), dtype=torch.long)
        for idx, word in enumerate(sequence):
            index_tensor[0][idx] = w2i[word]
        return index_tensor

    for sentence in sentences:
        sentence.index_tensor = get_tensor(sentence.preprocessed_sentence, w2i)
    for app_id in reviews:
        for review in reviews[app_id]:
            review.index_tensor = get_tensor(review.preprocessed_sentence, w2i)


def load_data(infile):
    with open(infile, "rb") as target:
        lst_of_objects = pickle.load(target)
        return lst_of_objects


def save_data(outfile, list_of_objects):
    if not os.path.exists(os.path.dirname(outfile)):
        os.makedirs(os.path.dirname(outfile))

    with open(outfile, "wb") as target:
        pickle.dump(list_of_objects, target)
