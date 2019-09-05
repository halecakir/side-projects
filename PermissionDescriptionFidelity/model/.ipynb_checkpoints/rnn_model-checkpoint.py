"""TODO"""
import dynet_config

# Declare GPU as the default device type
dynet_config.set_gpu()
# Set some parameters manualy
dynet_config.set(mem=400, random_seed=123456789)
# Initialize dynet import using above configuration in the current scope

import dynet as dy
import numpy as np
from numpy import inf

from utils.io_utils import IOUtils

from .base_model import BaseModel


class RNNModel(BaseModel):
    """TODO"""

    def __init__(self, w2i, permissions, options):
        super().__init__(options)
        self.model = dy.ParameterCollection()
        self.trainer = dy.AdamTrainer(self.model)
        self.w2i = w2i
        self.wdims = options.wembedding_dims
        self.ldims = options.lstm_dims
        self.all_permissions = permissions

        # Model Parameters
        self.wlookup = self.model.add_lookup_parameters((len(w2i), self.wdims))

        # RNNs
        self.sentence_rnn = [dy.SimpleRNNBuilder(1, self.wdims, self.ldims, self.model)]

        if options.external_embedding is not None:
            self.__load_external_embeddings()

    def __load_external_embeddings(self):
        ext_embeddings, ext_emb_dim = IOUtils.load_embeddings_file(
            self.options.external_embedding,
            self.options.external_embedding_type,
            lower=True,
        )
        assert ext_emb_dim == self.wdims
        print("Initializing word embeddings by pre-trained vectors")
        count = 0
        for word in self.w2i:
            if word in ext_embeddings:
                count += 1
                self.wlookup.init_row(self.w2i[word], ext_embeddings[word])
        self.ext_embeddings = ext_embeddings
        print(
            "Vocab size: %d; #words having pretrained vectors: %d"
            % (len(self.w2i), count)
        )

    def __cosine_proximity(self, pred, gold):
        def l2_normalize(vector):
            square_sum = dy.sqrt(
                dy.bmax(
                    dy.sum_elems(dy.square(vector)),
                    np.finfo(float).eps * dy.ones((1))[0],
                )
            )
            return dy.cdiv(vector, square_sum)

        y_true = l2_normalize(pred)
        y_pred = l2_normalize(gold)
        return -dy.sum_elems(dy.cmult(y_true, y_pred))

    def __cosine_loss(self, pred, gold):
        sn1 = dy.l2_norm(pred)
        sn2 = dy.l2_norm(gold)
        mult = dy.cmult(sn1, sn2)
        dot = dy.dot_product(pred, gold)
        div = dy.cdiv(dot, mult)
        vec_y = dy.scalarInput(2)
        res = dy.cdiv(1 - div, vec_y)
        return res

    def __sentence_phrases_permission_sim(self, phrases, perm):
        max_sim = -inf
        max_index = 0
        for index, phrase_enc in enumerate(phrases):
            sim = self._cos_similarity(phrase_enc, perm)
            if max_sim < sim:
                max_sim = sim
                max_index = index
        return max_sim, max_index

    def statistics(self, similarities):
        """TODO"""
        statistics = {}
        for app_id in similarities.keys():
            statistics[app_id] = {"related": {"all": []}, "unrelated": {"all": []}}
            for related_p in similarities[app_id]["related"]:
                statistics[app_id]["related"]["all"].append(related_p[1])

            for unrelated_p in similarities[app_id]["unrelated"]:
                statistics[app_id]["unrelated"]["all"].append(unrelated_p[1])
        return statistics

    def __encode_permissions(self):
        permission_vecs = {}
        # gather all permission encoding of permissions
        for perm in self.all_permissions:
            permission_vecs[perm.ptype] = self.__encode_sequence(perm.pphrase).npvalue()
            dy.renew_cg()
        return permission_vecs

    def __encode_sequence(self, sequence):
        rnn_forward = self.sentence_rnn[0].initial_state()
        for entry in sequence:
            vec = self.wlookup[int(self.w2i.get(entry, 0))]
            rnn_forward = rnn_forward.add_input(vec)
        return rnn_forward.output()

    def train_unsupervised(self, applications):
        """TODO"""
        app_permission_similarity = {}
        permission_vecs = self.__encode_permissions()

        for app in applications:
            if app.description.phrases:
                app_permission_similarity[app.id] = {"related": [], "unrelated": []}

                for sentence in app.description.phrases:
                    sentence_phrases_enc = []
                    for phrase in sentence:
                        phrase_encode = self.__encode_sequence(phrase)
                        if phrase_encode is not None:
                            sentence_phrases_enc.append(phrase_encode.npvalue())
                        dy.renew_cg()

                    for perm in self.all_permissions:
                        max_sim, _ = self.__sentence_phrases_permission_sim(
                            sentence_phrases_enc, permission_vecs[perm.ptype]
                        )
                        if perm in app.permissions:
                            app_permission_similarity[app.id]["related"].append(
                                (perm.ptype, max_sim)
                            )
                        else:
                            app_permission_similarity[app.id]["unrelated"].append(
                                (perm.ptype, max_sim)
                            )
        return app_permission_similarity

    def train_supervised(self, applications):
        """TODO"""
        tagged_loss = 0
        untagged_loss = 0
        for app in applications:
            if app.description.phrases:
                # Sentence encoding
                for sentence, tag in zip(
                    app.description.phrases, app.description.manual_marked
                ):
                    for phrase in sentence:
                        # gather all permission encoding of permissions
                        permission_vecs = {}
                        for perm in self.all_permissions:
                            permission_vecs[perm.ptype] = self.__encode_sequence(
                                perm.pphrase
                            )

                        phrase_expression = self.__encode_sequence(phrase)
                        loss = []
                        for perm in self.all_permissions:
                            if perm in app.permissions:
                                similarity = self.__cosine_loss(
                                    phrase_expression, permission_vecs[perm.ptype]
                                )
                                if tag in (1, 2, 3):
                                    loss.append(1 - similarity)
                                elif tag == 0:
                                    loss.append(similarity)

                        loss = dy.esum(loss)
                        if tag in (1, 2, 3):
                            tagged_loss += loss.scalar_value()
                        elif tag == 0:
                            untagged_loss += loss.scalar_value()
                        else:
                            raise Exception("Unexpected tag!")
                        loss.backward()
                        self.trainer.update()
                        dy.renew_cg()

        total_loss = tagged_loss + untagged_loss
        print(
            "Total loss : {} - Tagged Loss {} - Untagged loss {}".format(
                total_loss, tagged_loss, untagged_loss
            )
        )

    def test(self, applications):
        """TODO"""
        application_permission_similarity = {}
        permission_vecs = self.__encode_permissions()

        for app in applications:
            if app.description.phrases:
                application_permission_similarity[app.id] = {
                    "related": [],
                    "unrelated": [],
                }
                for sentence, tag in zip(app.description.phrases, app.description.tags):
                    sentence_phrases_enc = []
                    if tag in (1, 2, 3):
                        for phrase in sentence:
                            phrase_expression = self.__encode_sequence(phrase)
                            if phrase_expression is not None:
                                sentence_phrases_enc.append(phrase_expression.npvalue())
                            dy.renew_cg()

                        for perm in self.all_permissions:
                            max_sim, _ = self.__sentence_phrases_permission_sim(
                                sentence_phrases_enc, permission_vecs[perm.ptype]
                            )
                            if perm in app.permissions:
                                application_permission_similarity[app.id][
                                    "related"
                                ].append((perm.ptype, max_sim))
                            else:
                                application_permission_similarity[app.id][
                                    "unrelated"
                                ].append((perm.ptype, max_sim))
        return application_permission_similarity
