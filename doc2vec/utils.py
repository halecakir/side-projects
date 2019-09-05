import csv
import os
import re
from collections import Counter

import numpy as np
import xlrd
from gensim.models import KeyedVectors
from gensim.models.wrappers import FastText


from nltk import sent_tokenize, word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk; nltk.download('punkt'); nltk.download('averaged_perceptron_tagger')

#import stanfordnlp
from nltk.parse import CoreNLPParser
from nltk.tree import *
import pprint
from nltk.corpus import stopwords


MODELS_DIR = 'models'
#stanfordnlp.download('en', MODELS_DIR)
#nlp = stanfordnlp.Pipeline(processors='tokenize', models_dir=MODELS_DIR, treebank='en_ewt', use_gpu=True, pos_batch_size=3000)


class Application:
    def __init__(self, app_id, dsc_sentences, description, dsc_vec, related_permission_doc, tags):
        self.app_id = app_id
        self.dsc_sentences = dsc_sentences
        self.description = description
        self.dsc_vec = dsc_vec
        self.related_permission_doc = related_permission_doc
        self.tags = tags

    # def __str__(self):
    #     print(self.app_id)
    #     print(self.description)


class DscSentence:
    def __init__(self, sentence, chunk_list, permission_doc, manual_marked, key_based, whyper_tool):
        self.sentence = sentence
        self.chunk_list = chunk_list
        self.permission_doc = permission_doc
        self.manual_marked = manual_marked
        self.key_based = key_based
        self.whyper_tool = whyper_tool

    # def __str__(self):
    #     print(self.sent)


class Document:
    def __init__(self, doc_id, title, description, permissions, tags=None):
        self.id = doc_id
        self.title = title
        self.permissions = permissions
        self.description = description
        self.tags = tags

    def __str__(self):
        print(self.title)
        print(self.permissions)


class Permission:
    def __init__(self, permission_type, permission_phrase):
        self.ptype = permission_type
        self.pphrase = permission_phrase

    def __str__(self):
        print(self.ptype)
        print(self.pphrase)


class OverallResultRow:
    def __init__(self, permission, SI, TP, FP, FN, TN, precision_percent, recall_percent, FScore_percent, accuracy_percent):
        self.permission = permission
        self.SI = SI
        self.TP = TP
        self.FP = FP
        self.FN = FN
        self.TN = TN
        self.precision_percent=precision_percent
        self.recall_percent = recall_percent
        self.FScore_percent = FScore_percent
        self.accuracy_percent = accuracy_percent

class Utils:

    @staticmethod
    def read_word_vec(filepath, first_index=0):

        #if not os.path.isfile(filepath):
         #   print("Wiki Vectors File path {} does not exist. Exiting...".format(filepath))
          #  sys.exit()

        word_vec = {}
        with open(filepath) as fp:
            cnt = 0
            for line in fp:
                if cnt < first_index:
                    cnt += 1
                else:
                    #print("line {} contents {}".format(cnt, line))
                    Utils.save_word_vec(line.strip().split(' '), word_vec)
                    cnt += 1
                # if cnt == 5000:
                #     break
        print("input vector length : ")
        print(cnt)
        return word_vec

    @staticmethod
    def save_word_vec(words, word_vec):

        index = 0
        key = ""
        value = []

        for word in words:
            if word != '':
                if index == 0:
                    key = word.lower()
                else:
                    value.append(float(word))
            index += 1

        word_vec[key] = value

    @staticmethod
    def read_whyper_data(input_folder, file_path, file_type, lower, chunk_gram, remove_stop_words):
        wordsCount = Counter()
        permissions = []
        distincts_permissions = set()

        applications = []

        if file_type == "excel":
            handtagged_permissions = ["READ_CALENDAR", "READ_CONTACTS", "RECORD_AUDIO"]
            loc = (input_folder + "/" + file_path)
            wb = xlrd.open_workbook(loc)
            sheet = wb.sheet_by_index(0)

            permission_title = file_path.split("/")[-1].split(".")[0]

            app_id = ""
            app_sentences = []
            app_description = ""
            app_tag = ""

            sharp_count = 0

            if file_path == "Read_Contacts.xls":
                for i in range(sheet.nrows):
                    sentence = sheet.cell_value(i, 0)
                    # sentence = str(sentence.encode("utf-8"))
                    print(sentence)
                    if sentence.startswith("#"):
                        if sharp_count != 0:
                            app_dsc_vec = ""
                            applications.append(Application(app_id, app_sentences, app_description, app_dsc_vec, permission_title, app_tag))
                        elif sharp_count == 0:
                            sharp_count = sharp_count + 1
                        app_id = sentence.split("#")[1]
                        app_sentences = []
                        app_description = ""
                        sentence = sheet.cell_value(i, 1)
                        sentence = str(sentence)
                        app_tag = sentence.split("\\")[2]
                        print("########## Application ID  : " + app_id)
                        print("########## Application Tag : " + app_tag + "\n")
                    else:
                        if sharp_count != 0:

                            manual_marked = sheet.cell_value(i, 2)
                            key_based = sheet.cell_value(i, 3)
                            whyper_tool = sheet.cell_value(i, 4)

                            app_sentence = ""
                            sentence = sentence.strip()

                            tokenizer = RegexpTokenizer(r'\w+')
                            for w in tokenizer.tokenize(sentence):
                                w = Utils.remove_hyperlinks(w)
                                wordsCount.update([Utils.to_lower(w, lower)])
                                app_sentence += " " + w

                            app_description = app_description + app_sentence.strip() + ". "
                            sent_chunk_list2 = []
                            app_sentences.append(DscSentence(app_sentence.strip(), sent_chunk_list2, permission_title, manual_marked, key_based, whyper_tool))

            else:
                for i in range(sheet.nrows):
                    sentence = sheet.cell_value(i, 0)
                    sentence = str(sentence)
                    if sentence.startswith("##"):
                        sharp_count += 1
                        if sharp_count % 2 == 1:
                            if sharp_count != 1:
                                app_dsc_vec = ""
                                applications.append(Application(app_id, app_sentences, app_description, app_dsc_vec, permission_title, app_tag) )
                            app_id = sentence.split("##")[1]
                            app_sentences = []
                            app_description = ""
                        else:
                            app_tag = sentence.split("\\")[2]

                    else:
                        if sharp_count != 0 and sharp_count % 2 == 0:

                            manual_marked = sheet.cell_value(i, 1)
                            key_based = sheet.cell_value(i, 2)
                            whyper_tool = sheet.cell_value(i, 3)

                            app_sentence = ""
                            sentence = sentence.strip()

                            tokenizer = RegexpTokenizer(r'\w+')
                            for w in tokenizer.tokenize(sentence):
                                wordsCount.update([Utils.to_lower(w, lower)])
                                app_sentence += " " + w
                            app_description = app_description + app_sentence.strip() + ". "
                            sent_chunk_list2 = []
                            app_sentences.append(DscSentence(app_sentence.strip(), sent_chunk_list2, permission_title, manual_marked, key_based, whyper_tool))

            for p in handtagged_permissions:
                ptype = Utils.to_lower(p, lower)
                if ptype not in distincts_permissions:
                    pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                    perm = Permission(ptype, pphrase)
                    permissions.append(perm)
                    distincts_permissions.add(ptype)
                    for token in p.split("_"):
                        wordsCount.update([Utils.to_lower(token, lower)])
        else:
            raise Exception("Unsupported file type.")
        return wordsCount.keys(), {w: i for i, w in enumerate(list(wordsCount.keys()))}, permissions, applications

    @staticmethod
    def chunker(sentence, chunk_gram, remove_stop_words):

        try:
            words = nltk.word_tokenize(sentence)
            tagged = nltk.pos_tag(words)
            chunk_parser = nltk.RegexpParser(chunk_gram)
            chunked = chunk_parser.parse(tagged)

            # chunked.draw()

            # print("------------------------------*****************")
            # for subtree in chunked.subtrees(filter=lambda t: t.label() == 'Chunk' or t.label() == 'CLAUSE'):
            #     print(subtree)

            noun_phrases_list = [' '.join(leaf[0] for leaf in tree.leaves())
                                  for tree in chunked.subtrees()
                                  if tree.label() == 'Chunk']

        except Exception as e:
            print(str(e))

        return noun_phrases_list

    @staticmethod
    def chunk_with_sp(sentence):

        verb_phrase_list = []
        try:
            parser = CoreNLPParser(url='http://localhost:9000')
            parsed_tree = next(parser.raw_parse(sentence))
            # parsed_tree.pretty_print()

            # VP_tree = list(parsed_tree.subtrees(filter=lambda x: x.label() == 'VP'))
            parsed_phrase_tree = Utils.ExtractPhrases(parsed_tree, 'VP')

            for vp in parsed_phrase_tree:
                verb_phrase_list.append(" ".join(vp.leaves()))

            # print("\nVerb phrases:")
            # print("Verb Phrase List : ", verb_phrase_list)
            # print("--------\n")

        except Exception as e:
            print(str(e))

        return verb_phrase_list

    # Extract phrases from a parsed (chunked) tree
    # Phrase = tag for the string phrase (sub-tree) to extract
    # Returns: List of deep copies;  Recursive
    @staticmethod
    def ExtractPhrases(myTree, phrase):
        myPhrases = []
        if myTree.label() == phrase:
            myPhrases.append(myTree.copy(True))
        for child in myTree:
            if type(child) is Tree:
                list_of_phrases = Utils.ExtractPhrases(child, phrase)
                if len(list_of_phrases) > 0:
                    myPhrases.extend(list_of_phrases)
        return myPhrases

    @staticmethod
    def to_lower(w, lower):
        return w.lower() if lower else w

    @staticmethod
    def cos_similiariy(v1, v2):
        from numpy import dot
        from numpy.linalg import norm
        return dot(v1, v2)/(norm(v1)*norm(v2))

    # @staticmethod
    # def word_tokenization(sentence):
    #     doc = nlp(sentence)
    #     return [token.text for token in doc.sentences[0].tokens]
    #
    # @staticmethod
    # def dependency_parse(sentence):
    #     doc = nlp(sentence)
    #     return [dep for dep in doc.sentences[0].dependencies]

    @staticmethod
    def remove_stopwords(sentence):

        stop_words = set(stopwords.words('english'))
        word_tokens = word_tokenize(sentence)
        # filtered_words = [word for word in sentence if word not in stop_words]
        filtered_words = ""
        for w in word_tokens:
            if w not in stop_words:
                filtered_words = filtered_words + " " + w


        return filtered_words


    @staticmethod
    def remove_hyperlinks(text):
        regex = r"((https?:\/\/)?[^\s]+\.[^\s]+)"
        text = re.sub(regex, '', text)
        return text



    @staticmethod
    def preprocess(text):
        paragrahps = text.split("\n")
        sentences = []
        for p in paragrahps:
            for s in sent_tokenize(p):
                sentences.append(s)
        return sentences

    @staticmethod
    def remove_hyperlinks(text):
        regex = r"((https?:\/\/)?[^\s]+\.[^\s]+)"
        text = re.sub(regex, '', text)
        return text

    @staticmethod
    def vocab(file_path, file_type="csv", lower=True):
        wordsCount = Counter()
        permissions = []
        distincts_permissions = set()
        if file_type == "csv":
            with open(file_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    text = row[1]
                    for sentence in Utils.preprocess(text):
                        sentence = Utils.remove_hyperlinks(sentence)
                        for w in word_tokenize(sentence):
                            wordsCount.update([Utils.to_lower(w, lower)])
                        for p in row[2].strip().split(","):
                            ptype = Utils.to_lower(p, lower)
                            if ptype not in distincts_permissions:
                                pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                                perm = Permission(ptype, pphrase)
                                permissions.append(perm)
                                distincts_permissions.add(ptype)
                            for token in p.split("_"):
                                wordsCount.update([Utils.to_lower(token, lower)])
        elif file_type == "excel":
            handtagged_permissions = ["READ_CALENDAR", "READ_CONTACTS", "RECORD_AUDIO"]
            loc = (file_path)
            wb = xlrd.open_workbook(loc)
            sheet = wb.sheet_by_index(0)
            sharp_count = 0
            apk_title = ""
            for i in range(sheet.nrows):
                sentence = sheet.cell_value(i, 0)
                if sentence.startswith("##"):
                    sharp_count += 1
                    if sharp_count % 2 == 1:
                        apk_title = sentence.split("##")[1]
                else:
                    if sharp_count != 0 and sharp_count % 2 == 0:
                        sentence = sentence.strip()
                        for w in word_tokenize(sentence):
                            wordsCount.update([Utils.to_lower(w, lower)])
                        for p in handtagged_permissions:
                            ptype = Utils.to_lower(p, lower)
                            if ptype not in distincts_permissions:
                                pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                                perm = Permission(ptype, pphrase)
                                permissions.append(perm)
                                distincts_permissions.add(ptype)
                                for token in p.split("_"):
                                    wordsCount.update([Utils.to_lower(token, lower)])
        else:
            raise Exception("Unsupported file type.")
        return wordsCount.keys(), {w: i for i, w in enumerate(list(wordsCount.keys()))}, permissions

    @staticmethod
    def read_file(file_path, w2i, file_type="csv", lower=True):
        data = []
        doc_id = 0
        if file_type == "csv":
            with open(file_path) as f:
                reader = csv.reader(f)
                next(reader)  # skip header
                for row in reader:
                    doc_id += 1
                    title = row[0]
                    description = row[1]
                    permissions = []
                    for p in row[2].strip().split(","):
                        ptype = Utils.to_lower(p, lower)
                        pphrase = [Utils.to_lower(t, lower) for t in p.split("_")]
                        perm = Permission(ptype, pphrase)
                        permissions.append(perm)

                    sentences = []
                    for sentence in Utils.preprocess(description):
                        sentence = Utils.remove_hyperlinks(sentence)
                        sentences.append([Utils.to_lower(w, lower) for w in word_tokenize(sentence)])
                    yield Document(doc_id, title, sentences, permissions)

        elif file_type == "excel":
            permission_title = file_path.split("/")[-1].split(".")[0]
            loc = (file_path)
            wb = xlrd.open_workbook(loc)
            sheet = wb.sheet_by_index(0)
            sharp_count = 0
            title = ""
            permissions = []
            sentences = []
            tags = []
            for i in range(sheet.nrows):
                sentence = sheet.cell_value(i, 0)
                if sentence.startswith("##"):
                    sharp_count += 1
                    if sharp_count % 2 == 1:
                        if doc_id > 0:
                            yield Document(doc_id, title, sentences, permissions, tags)

                        # Document init values
                        title = sentence.split("##")[1]
                        permissions = []
                        sentences = []
                        tags = []
                        doc_id += 1

                        # Permissions for apk
                        ptype = Utils.to_lower(permission_title, lower)
                        pphrase = [Utils.to_lower(t, lower) for t in permission_title.split("_")]
                        perm = Permission(ptype, pphrase)
                        permissions.append(perm)
                else:
                    if sharp_count != 0 and sharp_count % 2 == 0:
                        sentences.append([Utils.to_lower(w, lower) for w in word_tokenize(sentence.strip())])
                        tags.append(int(sheet.cell_value(i, 1)))

            yield Document(doc_id, title, sentences, permissions, tags)
        else:
            raise Exception("Unsupported file type.")

    @staticmethod
    def read_file_window(file_path, w2i, file_type="csv", window_size=2, lower=True):
        for doc in Utils.read_file(file_path, w2i, file_type, lower):
            doc.description = Utils.split_into_windows(doc.description, window_size)
            yield doc

    @staticmethod
    def split_into_windows(sentences, window_size=2):
        splitted_sentences = []
        for sentence in sentences:
            splitted_sentences.append([])
            if len(sentence) < window_size:
                splitted_sentences[-1].append(sentence)
            else:
                for start in range(len(sentence) - window_size + 1):
                    splitted_sentences[-1].append([sentence[i + start] for i in range(window_size)])
        return splitted_sentences

    @staticmethod
    def load_embeddings_file(file_name, embedding_type, lower=True):
        if not os.path.isfile(file_name):
            print(file_name, "does not exist")
            return {}, 0

        if embedding_type == "word2vec":
            model = KeyedVectors.load_word2vec_format(file_name, binary=True, unicode_errors="ignore")
            words = model.index2entity
        elif embedding_type == "fasttext":
            model = FastText.load_fasttext_format(file_name)
            words = [w for w in model.wv.vocab]
        else:
            print("Unknown Type")
            return {}, 0

        if lower:
            vectors = {word.lower(): model[word] for word in words}
        else:
            vectors = {word: model[word] for word in words}

        if "UNK" not in vectors:
            unk = np.mean([vectors[word] for word in vectors.keys()], axis=0)
            vectors["UNK"] = unk

        return vectors, len(vectors["UNK"])
