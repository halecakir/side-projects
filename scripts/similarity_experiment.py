"""TODO"""
import random
import os
import csv

import dynet_config
# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)
# Initialize dynet import using above configuration in the current scope

import scipy
import dynet as dy
import pandas as pd
import numpy as np

from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils


random.seed(33)


class SentenceReport:
    """TODO"""
    def __init__(self, sentence, mark):
        self.mark = mark
        self.preprocessed_sentence = None
        self.sentence = sentence
        self.all_phrases = None
        self.feature_weights = None
        self.max_similarites = None
        self.prediction_result = None


class SimilarityExperiment:
    """TODO"""
    def __init__(self, w2i, options):
        print('Similarity Experiment - init')
        self.options = options
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.w2i = w2i
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims

        self.ext_embeddings = None
        #Model Parameters
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims))

        self.__load_model()

        self.phrase_rnn = [dy.VanillaLSTMBuilder(1, self.wdims, self.ldims, self.model)]
        self.mlp_w = self.model.add_parameters((1, self.ldims))
        self.mlp_b = self.model.add_parameters(1)

    def __load_model(self):
        if self.options.external_embedding is not None:
            if os.path.isfile(os.path.join(self.options.saved_parameters_dir,
                                           self.options.saved_prevectors)):
                self.__load_external_embeddings(os.path.join(self.options.saved_parameters_dir,
                                                             self.options.saved_prevectors),
                                                "pickle")
            else:
                self.__load_external_embeddings(self.options.external_embedding,
                                                self.options.external_embedding_type)
                self.__save_model()

    def __save_model(self):
        IOUtils.save_embeddings(os.path.join(self.options.saved_parameters_dir,
                                             self.options.saved_prevectors),
                                self.ext_embeddings)

    def __load_external_embeddings(self, embedding_file, embedding_file_type):
        ext_embeddings, ext_emb_dim = IOUtils.load_embeddings_file(
            embedding_file,
            embedding_file_type,
            lower=True)
        assert ext_emb_dim == self.wdims
        self.ext_embeddings = {}
        print("Initializing word embeddings by pre-trained vectors")
        count = 0
        for word in self.w2i:
            if word in ext_embeddings:
                count += 1
                self.ext_embeddings[word] = ext_embeddings[word]
                self.wlookup.init_row(self.w2i[word], ext_embeddings[word])
        print("Vocab size: %d; #words having pretrained vectors: %d" % (len(self.w2i), count))

    def __encode_phrase(self, phrase, encode_type, tfidf=None):
        if encode_type == "RNN":
            dy.renew_cg()
            rnn_forward = self.phrase_rnn[0].initial_state()
            for entry in phrase:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            return rnn_forward.output().npvalue()
        elif encode_type == "ADDITION":
            vec = np.sum([self.wlookup[int(self.w2i.get(entry, 0))].npvalue() for entry in phrase], axis=0)
            return vec
        elif encode_type == "TFIDF_ADDITION":
            lst = [self.wlookup[int(self.w2i.get(entry, 0))].npvalue()* tfidf[entry] for entry in phrase if entry in tfidf]
            vec = np.sum(lst, axis=0)
            return vec
        else:
            raise Exception("Undefined encode type")

    def _cos_similarity(self, vec1, vec2):
        from numpy import dot
        from numpy.linalg import norm
        normalized_similarity = (dot(vec1, vec2)/(norm(vec1)*norm(vec2)) + 1)/2
        return normalized_similarity

    def __to_lower(self, phrase):
        return phrase.lower()

    def __preprocess(self, sentence):
        sentence = self.__to_lower(sentence)
        text_wo_link = NLPUtils.remove_hyperlinks(sentence)
        tokens = []
        try:
            tokens = NLPUtils.word_tokenization(text_wo_link)
            tokens = [NLPUtils.punctuation_removal(token) for token in tokens]
            tokens = NLPUtils.stopword_elimination(tokens)
            tokens = NLPUtils.nonalpha_removal(tokens)
        except AssertionError:
            print("Phrase '{}' cannot be preprocessed".format(sentence))
        return " ".join(tokens)

    def __split_into_windows(self, sentence, window_size):
        splitted_sentences = []
        if len(sentence) < window_size:
            splitted_sentences.append(sentence)
        else:
            for start in range(len(sentence) - window_size + 1):
                splitted_sentences.append([sentence[i+start] for i in range(window_size)])
        return splitted_sentences

    def __find_all_possible_phrases(self, sentence, sentence_only=False):
        entries = sentence.split(" ")
        all_phrases = []
        if sentence_only:
            all_phrases.extend(self.__split_into_windows(entries, len(entries)))
        else:
            for windows_size in range(2, len(entries)+1):
                all_phrases.extend(self.__split_into_windows(entries, windows_size))
        return all_phrases

    def __find_max_similarities(self, report):
        def encode_permissions(encode_type):
            permissions = {}
            permissions["READ_CALENDAR"] = self.__encode_phrase(["read", "calendar"], encode_type, tfidf={"read": 0.5, "calendar": 0.5})
            permissions["READ_CONTACTS"] = self.__encode_phrase(["read", "contacts"], encode_type, tfidf={"read": 0.5, "contacts": 0.5})
            permissions["RECORD_AUDIO"] = self.__encode_phrase(["record", "audio"], encode_type, tfidf={"record": 0.5, "audio": 0.5})
            return permissions
        max_similarites = {"ADDITION" : {"READ_CALENDAR" : {"similarity" : 0, "phrase" : ""},
                                                            "READ_CONTACTS" : {"similarity" : 0, "phrase" : ""},
                                                            "RECORD_AUDIO" :  {"similarity" : 0, "phrase" : ""}},
                           "RNN" : {"READ_CALENDAR" : {"similarity" : 0, "phrase" : ""},
                                              "READ_CONTACTS" : {"similarity" : 0, "phrase" : ""},
                                              "RECORD_AUDIO"  :  {"similarity" : 0, "phrase" : ""}}}

        for encode_type in max_similarites:
            encoded_permissions = encode_permissions(encode_type)
            for part in report.all_phrases:
                encoded_phrase = self.__encode_phrase(part, encode_type, tfidf=report.feature_weights)
                if isinstance(encoded_phrase, np.ndarray):
                    for perm in encoded_permissions:
                        similarity_result = self._cos_similarity(encoded_phrase, encoded_permissions[perm])
                        if max_similarites[encode_type][perm]["similarity"] < similarity_result:
                           max_similarites[encode_type][perm]["similarity"] = similarity_result
                           max_similarites[encode_type][perm]["phrase"] = part
        return max_similarites

    def __dump_detailed_analysis(self, reports, file_name, reported_permission):
        with open(file_name, "w") as target:
            for report in reports:
                tag = "POSITIVE" if report.mark else "NEGATIVE"
                target.write("{} Sentence '{}' - Hantagged Permission {}\n".format(tag, report.sentence, reported_permission))
                for composition_type in report.max_similarites:
                    target.write("\t{} composition : \n".format(composition_type))
                    for permission in report.max_similarites[composition_type]:
                        simimarity =    \
                            report.max_similarites[composition_type][permission]["similarity"]
                        phrase =    \
                            report.max_similarites[composition_type][permission]["phrase"]
                        target.write("\t\t{0} : {1:.3f}\t{2}\n".format(permission, simimarity, phrase))
                target.write("\n")

    def __linearized_similarity_values(self, reports):
        values = {"POSITIVE": {}, "NEGATIVE": {}}
        for report in reports:
            report_tag = "POSITIVE" if report.mark else "NEGATIVE"
            for composition_type in report.max_similarites:
                if composition_type not in values[report_tag]:
                    values[report_tag][composition_type] = {}
                for permission in report.max_similarites[composition_type]:
                    if permission not in values[report_tag][composition_type]:
                        values[report_tag][composition_type][permission] = []
                    similarity = report.max_similarites[composition_type][permission]["similarity"]
                    values[report_tag][composition_type][permission].append(similarity)
        return values

    def __compute_all_desriptive_statistics(self, values):
        def compute_descriptive_statistics(array):
            stats = {}
            descriptive_stats = scipy.stats.describe(array)
            stats["count"] = len(array)
            stats["mean"] = descriptive_stats.mean
            stats["minmax"] = descriptive_stats.minmax
            stats["std"] = np.std(array)
            return stats

        stats = {}
        for tag in values:
            if tag not in stats:
                stats[tag] = {}
            for composition_type in values[tag]:
                if composition_type not in stats[tag]:
                    stats[tag][composition_type] = {}
                for permission in values[tag][composition_type]:
                    if permission not in stats[tag][composition_type]:
                        stats[tag][composition_type][permission] = {}
                    linearized_values = values[tag][composition_type][permission]
                    stats[tag][composition_type][permission] = compute_descriptive_statistics(linearized_values)
        return stats

    def __write_all_stats(self, stats, file_name):
        with open(file_name, "w") as target:
            for tag_idx, tag in enumerate(stats):
                target.write("{}. {} Examples\n".format(tag_idx+1, tag))
                for c_type_idx, composition_type in enumerate(stats[tag]):
                    target.write("\t{}.{} {} Composition\n".format(tag_idx+1, c_type_idx+1, composition_type))
                    for perm_idx, permission in enumerate(stats[tag][composition_type]):
                        target.write("\t\t{}.{}.{} {} Permission\n".format(tag_idx+1, c_type_idx+1, perm_idx+1, permission))
                        for stat in stats[tag][composition_type][permission]:
                            val = stats[tag][composition_type][permission][stat]
                            target.write("\t\t\t{} : {}\n".format(stat, val))
                target.write("\n\n")

    def __draw_distribution(self, data, axis_label, file_name):
        from matplotlib import pyplot as plt
        import seaborn as sns
        sns.set_style('darkgrid')
        sns.distplot(data, axlabel=axis_label)
        plt.savefig(file_name)
        plt.clf()

    def __draw_charts(self, outdir, values, gold_permission):
        for tag in values:
            for composition_type in values[tag]:
                for permission in values[tag][composition_type]:
                    img = "{}_{}_{}_(gold-{}).png".format(tag.lower(),
                                                          composition_type.lower(),
                                                          permission.lower(),
                                                          gold_permission.lower())
                    img_dir = os.path.join(outdir, img)
                    self.__draw_distribution(values[tag][composition_type][permission],
                                             "{}_{}_{}".format(tag.lower(),
                                                               composition_type.lower(),
                                                               permission.lower()),
                                                               img_dir)

    def __normalize_similarity_values(self, values):
        normalized_values = {}
        for tag in values:
            if tag not in normalized_values:
                normalized_values[tag] = {}
            for composition_type in values[tag]:
                if composition_type not in normalized_values[tag]:
                    normalized_values[tag][composition_type] = {}
                for permission in values[tag][composition_type]:
                    if permission not in normalized_values[tag][composition_type]:
                        normalized_values[tag][composition_type][permission] = []
                    for sim in values[tag][composition_type][permission]:
                        if sim != 0:
                            normalized_values[tag][composition_type][permission].append(sim)
        return normalized_values


    def __find_optimized_threshold(self, file_name, values, composition_type, gold_permission):
        with open(file_name, "w") as target:
            tp = values["POSITIVE"][composition_type][gold_permission.upper()]
            tn = values["NEGATIVE"][composition_type][gold_permission.upper()]
            fn = []
            for perm in values["POSITIVE"][composition_type]:
                if perm != gold_permission.upper():
                    fn.extend(values["POSITIVE"][composition_type][perm])
            target.write("TP {} - TN {} - FN {}\n".format(len(tp), len(tn), len(fn)))
            target.write("Applying Threshold:\n")
            best_threshold_tp_tn = 0
            best_threshold_tp_fn = 0
            best_threshold_tp_tn_fn = 0
            thresh_tp_tn = 0
            thresh_tp_fn = 0
            thresh_tp_tn_fn = 0

            for threshold in np.arange(0.01, 0.99, 0.01):
                thresholded_tp = len(list(filter(lambda x: x > threshold, tp)))
                thresholded_tn = len(list(filter(lambda x: x < threshold, tn)))
                thresholded_fn = len(list(filter(lambda x: x < threshold, fn)))
                target.write("Threshold : {}\n".format(threshold))
                target.write("\tCount : TP {} - TN {} - FN {}\n".format(thresholded_tp, thresholded_tn, thresholded_fn))
                target.write("\tRatio : TP {} - TN {} - FN {}\n".format(thresholded_tp/len(tp)*100,
                                                                        thresholded_tn/len(tn)*100,
                                                                        thresholded_fn/len(fn)*100))

                if thresholded_tp + thresholded_tn > best_threshold_tp_tn:
                    best_threshold_tp_tn = thresholded_tp + thresholded_tn
                    thresh_tp_tn = threshold
                    target.write("\t\t{} : {}\n".format("thresh_tp_tn", thresh_tp_tn))
                if thresholded_tp + thresholded_fn > best_threshold_tp_fn:
                    best_threshold_tp_fn = thresholded_tp + thresholded_fn
                    thresh_tp_fn = threshold
                    target.write("\t\t{} : {}\n".format("thresh_tp_fn", thresh_tp_fn))
                if thresholded_tp + thresholded_fn + thresholded_tn > best_threshold_tp_tn_fn:
                    best_threshold_tp_tn_fn = thresholded_tp + thresholded_fn + thresholded_tn
                    thresh_tp_tn_fn = threshold
                    target.write("\t\t{} : {}\n".format("thresh_tp_tn_fn", thresh_tp_tn_fn))

            target.write("\n--Best Thresholded Parameters--\n")
            target.write("best_threshold_tp_tn {} : {}\n".format(best_threshold_tp_tn, thresh_tp_tn))
            target.write("best_threshold_tp_fn {} : {}\n".format(best_threshold_tp_fn, thresh_tp_fn))
            target.write("best_threshold_tp_tn_fn {} : {}\n".format(best_threshold_tp_tn_fn, thresh_tp_tn_fn))

    def __compute_tf_idf(self, data):
        from sklearn.feature_extraction.text import TfidfVectorizer
        vectorizer = TfidfVectorizer(max_df=1.0, min_df=0.0)
        X = vectorizer.fit_transform(data)
        feature_names = vectorizer.get_feature_names()
        feat_to_weight = {}
        for doc_id in range(X.shape[0]):
            feat_to_weight[doc_id] = {feature_names[ind] : weight for ind, weight in zip(X[doc_id].indices, X[doc_id].data)}
        return feat_to_weight

    def __cosine_loss(self, pred, gold):
        sn1 = dy.l2_norm(pred)
        sn2 = dy.l2_norm(gold)
        mult = dy.cmult(sn1, sn2)
        dot = dy.dot_product(pred, gold)
        div = dy.cdiv(dot, mult)
        vec_y = dy.scalarInput(2)
        res = dy.cdiv(1-div, vec_y)
        return res

    def __train(self, data):
        def encode_sequence(seq):
            rnn_forward = self.phrase_rnn[0].initial_state()
            for entry in seq:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            return rnn_forward.output()
        tagged_loss = 0
        untagged_loss = 0
        for index, sentence_report in enumerate(data):
            for phrase in sentence_report.all_phrases:
                loss = None
                encoded_phrase = encode_sequence(phrase)
                y_pred = dy.logistic((self.mlp_w*encoded_phrase) + self.mlp_b)

                if sentence_report.mark:
                    loss = dy.binary_log_loss(y_pred, dy.scalarInput(1))
                else:
                    loss = dy.binary_log_loss(y_pred, dy.scalarInput(0))
                if index % 1000 == 0:
                    print("Description : {}".format(index+1))
                    print("Marked {} Prediction Result {} : ".format(sentence_report.mark, y_pred.scalar_value()))
                    print("Tagged loss {} Untagged Loss {} Total loss {}".format(tagged_loss, untagged_loss, tagged_loss+untagged_loss))

                if sentence_report.mark:
                    tagged_loss += loss.scalar_value()/(index+1)
                else:
                    untagged_loss += loss.scalar_value()/(index+1)
                loss.backward()
                self.trainer.update()
                dy.renew_cg()

    def __predict(self, data):
        def encode_sequence(seq):
            rnn_forward = self.phrase_rnn[0].initial_state()
            for entry in seq:
                vec = self.wlookup[int(self.w2i.get(entry, 0))]
                rnn_forward = rnn_forward.add_input(vec)
            return rnn_forward.output()

        for _, sentence_report in enumerate(data):
            for phrase in sentence_report.all_phrases:
                encoded_phrase = encode_sequence(phrase)
                y_pred = dy.logistic((self.mlp_w*encoded_phrase) + self.mlp_b)
                sentence_report.prediction_result = y_pred.scalar_value()
                dy.renew_cg()

    def __read_raw_data(self, file_path, test_permission):
        sentence_reports = []
        with open(file_path) as stream:
            reader = csv.reader(stream)
            header = next(reader)
            for row in reader:
                title = row[0]
                text = row[1]
                permissions = row[2]
                link = row[3]
                sentence = text.replace("%%", " ")

                sentence_report = None
                app_perms = {perm for perm in permissions.split("%%")}
                if test_permission in app_perms:
                    sentence_report = SentenceReport(sentence, mark=True)
                else:
                    sentence_report = SentenceReport(sentence, mark=False)
                sentence_reports.append(sentence_report)
        return sentence_reports



    def __report_confusion_matrix(self, sentence_reports, threshold):
        results = {"TP":0, "TN":0, "FP":0, "FN":0, "Threshold":threshold}
        total = 0
        for report in sentence_reports:
            total += 1
            if report.mark:
                if report.prediction_result >= threshold:
                    results["TP"] += 1
                else:
                    results["FN"] += 1
            else:
                if report.prediction_result >= threshold:
                    results["FP"] += 1
                else:
                    results["TN"] += 1


        try:
            results["precision"] = results["TP"]/(results["TP"]+results["FP"])
            results["recall"] = results["TP"]/(results["TP"]+results["FN"])
            results["f1_score"] = 2*((results["precision"]*results["recall"])/(results["precision"]+results["recall"]))
            results["accuracy"] = (results["TP"]+results["TN"])/(results["TP"]+results["TN"]+results["FP"]+results["FN"])
        except ZeroDivisionError:
            results["precision"] = 0
            results["recall"] = 0
            results["f1_score"] = 0
            results["accuracy"] = 0
        return results

    def run(self):
        """TODO"""
        print('Similarity Experiment - run')
        train_file = self.options.train
        test_file = self.options.test
        outdir = self.options.outdir
        gold_permission = self.options.permission_type

        tagged_test_file = pd.read_csv(test_file)
        test_sentence_reports = []
        import pdb
        #pdb.set_trace()
        #read and preprocess whyper sentences
        print("Reading Test Sentences")
        for _, row in tagged_test_file.iterrows():
            sentence = str(row["Sentences"])
            if not sentence.startswith("#"):
                mark = False if row["Manually Marked"] is 0 else True
                sentence_report = SentenceReport(sentence, mark)
                sentence_report.preprocessed_sentence = self.__preprocess(sentence_report.sentence)
                sentence_report.all_phrases = self.__find_all_possible_phrases(sentence_report.preprocessed_sentence,
                                                                               sentence_only=True)
                test_sentence_reports.append(sentence_report)
        #pdb.set_trace()
        #read training data
        print("Reading Train Sentences")
        tagged_train_file = pd.read_csv(train_file)
        train_sententence_reports = []
        acnet_map = {"RECORD_AUDIO" : "MICROPHONE", "READ_CONTACTS": "CONTACTS", "READ_CALENDAR": "CALENDAR"}
        i = 0
        for _, row in tagged_train_file.iterrows():
            sentence = row["sentence"]
            mark = False if row[acnet_map[gold_permission]] is 0 else True
            #print(i, sentence, mark)
            i = i+1
            sentence_report = SentenceReport(sentence, mark)
            sentence_report.preprocessed_sentence = self.__preprocess(sentence_report.sentence)
            sentence_report.all_phrases = self.__find_all_possible_phrases( sentence_report.preprocessed_sentence,
                                                                            sentence_only=True)
            train_sententence_reports.append(sentence_report)
        #pdb.set_trace()
        #shuffle data
        random.shuffle(test_sentence_reports)
        random.shuffle(train_sententence_reports)

        test_sentences = test_sentence_reports
        train_sentences = train_sententence_reports
        pdb.set_trace()
        print("Training")
        sentence_reports = test_sentences
        self.__train(train_sentences)
        #pdb.set_trace()
        self.__predict(test_sentences)
        #pdb.set_trace()
        #compute metrics
        threshold_metrics = []
        for threshold in np.arange(0.01, 0.99, 0.01):
            m = self.__report_confusion_matrix(test_sentences, threshold)
            threshold_metrics.append(m)
        #pdb.set_trace()
        #print out metrics
        metrics_dir = os.path.join(outdir, "metrics.txt")
        with open(metrics_dir, "a") as target:
            best_f1_score = 0
            best_result = {}
            for result in threshold_metrics:
                if result["f1_score"] > best_f1_score:
                    best_f1_score = result["f1_score"]
                    best_result = result

                for metric in result:
                    target.write("{} : {}\n".format(metric, result[metric]))
                target.write("-----\n\n")
            target.write("Best results : \n")
            for metric in best_result:
                target.write("{} : {}\n".format(metric, result[metric]))
            target.write("-----\n\n")


        """
        #compute feature weights
        documents = [report.preprocessed_sentence for report in sentence_reports]
        feature_to_weights = self.__compute_tf_idf(documents)
        for index, report in enumerate(sentence_reports):
            report.feature_weights = feature_to_weights[index]

        #find max similarities
        for report in sentence_reports:
            report.max_similarites = self.__find_max_similarities(report)

        #linearize similarity values
        values = self.__linearized_similarity_values(sentence_reports)

        # Analysis results
        analysis_file_dir = os.path.join(outdir, "{}_analysis.txt".format(gold_permission))
        self.__dump_detailed_analysis(sentence_reports,
                                      analysis_file_dir,
                                      gold_permission.upper())

        # Stats results
        stats = self.__compute_all_desriptive_statistics(values)
        stats_file_dir = os.path.join(outdir, "{}_stats.txt".format(gold_permission))
        self.__write_all_stats(stats, stats_file_dir)

        # Charts
        self.__draw_charts(outdir, values, gold_permission)

        # Threshold results
        composition_type = "RNN"
        thresholds_file_dir = os.path.join(outdir, "{}_{}_threshold_results.txt".format(gold_permission, composition_type.lower()))
        self.__find_optimized_threshold(thresholds_file_dir, values, composition_type, gold_permission)
        """
