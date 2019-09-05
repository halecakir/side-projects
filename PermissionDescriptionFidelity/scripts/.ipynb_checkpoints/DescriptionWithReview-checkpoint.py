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


# In[2]:


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
    saved_all_data = "{}/all_data".format(saved_parameters_dir)
    reviews = "/home/huseyinalecakir/Security/data/reviews/acnet-reviews/acnet_initial/app_reviews_original.csv"
    lower = True
    outdir = "./test/{}".format(permission_type)


class TorchOptions:
    d_rnn_size = 300
    r_rnn_size = 300
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


def train_item(
    opt,
    args,
    sentence,
    review,
    encoder_description,
    encoder_review,
    classifier,
    optimizer,
    criterion,
):
    optimizer.zero_grad()
    c_d = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    h_d = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    for i in range(sentence.index_tensor.size(1)):
        c_d, h_d = encoder_description(sentence.index_tensor[:, i], c_d, h_d)

    c_r = torch.zeros((1, opt.r_rnn_size), dtype=torch.float, requires_grad=True)
    h_r = torch.zeros((1, opt.r_rnn_size), dtype=torch.float, requires_grad=True)
    for i in range(review.index_tensor.size(1)):
        c_r, h_r = encoder_review(review.index_tensor[:, i], c_r, h_r)

    h = torch.cat((h_d, c_r), 1)
    pred = classifier(h)
    loss = criterion(
        pred,
        torch.tensor([[sentence.permissions[args.permission_type]]], dtype=torch.float),
    )
    loss.backward()
    if opt.grad_clip != -1:
        torch.nn.utils.clip_grad_value_(encoder_description.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(encoder_review.parameters(), opt.grad_clip)
        torch.nn.utils.clip_grad_value_(classifier.parameters(), opt.grad_clip)
    optimizer.step()
    return loss


def predict(opt, sentence, review, encoder_description, encoder_review, classifier):
    c_d = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    h_d = torch.zeros((1, opt.d_rnn_size), dtype=torch.float, requires_grad=True)
    for i in range(sentence.index_tensor.size(1)):
        c_d, h_d = encoder_description(sentence.index_tensor[:, i], c_d, h_d)

    c_r = torch.zeros((1, opt.r_rnn_size), dtype=torch.float, requires_grad=True)
    h_r = torch.zeros((1, opt.r_rnn_size), dtype=torch.float, requires_grad=True)
    for i in range(review.index_tensor.size(1)):
        c_r, h_r = encoder_review(review.index_tensor[:, i], c_r, h_r)

    h = torch.cat((h_d, c_r), 1)
    pred = classifier(h)
    return pred


def pr_roc_auc(predictions, gold):
    y_true = np.array(gold)
    y_scores = np.array(predictions)
    roc_auc = roc_auc_score(y_true, y_scores)
    pr_auc = average_precision_score(y_true, y_scores)
    return roc_auc, pr_auc


def train_and_test(opt, args, w2i, train_data, test_data, reviews):
    encoder_description = Encoder(opt, w2i, ext_embeddings, opt.d_rnn_size)
    encoder_review = Encoder(opt, w2i, ext_embeddings, opt.r_rnn_size)

    classifier = Classifier(opt, 1)

    params = (
        list(encoder_description.parameters())
        + list(encoder_review.parameters())
        + list(classifier.parameters())
    )
    optimizer = optim.Adam(params)
    criterion = nn.BCELoss()

    losses = []

    print("Training...")
    encoder_description.train()
    encoder_review.train()
    classifier.train()
    for index, sentence in enumerate(train_data):
        loss = train_item(
            opt,
            args,
            sentence,
            reviews[sentence.app_id][0],
            encoder_description,
            encoder_review,
            classifier,
            optimizer,
            criterion,
        )
        if index != 0:
            if index % opt.print_every == 0:
                print(
                    "Index {} Loss {}".format(
                        index, np.mean(losses[index - opt.print_every :])
                    )
                )
        losses.append(loss.item())

    print("Predicting..")
    encoder_description.eval()
    encoder_review.eval()
    classifier.eval()
    predictions = []
    gold = []
    with torch.no_grad():
        for index, sentence in enumerate(test_data):
            pred = predict(
                opt,
                sentence,
                reviews[sentence.app_id][0],
                encoder_description,
                encoder_review,
                classifier,
            )
            predictions.append(pred)
            gold.append(sentence.permissions[args.permission_type])

    return pr_roc_auc(predictions, gold)


if __name__ == "__main__":
    args = ArgumentParser()
    opt = TorchOptions()

    save_dir = os.path.join(args.saved_all_data, "with_prediction.pickle")
    ext_embeddings, reviews, sentences, w2i = load_all_data(save_dir)

    for app_id in reviews.keys():
        reviews[app_id].sort(key=lambda x: x.prediction_result.item(), reverse=True)

    documents = []
    for sentence in sentences:
        if sentence.app_id in reviews:
            documents.append(sentence)

    documents = np.array(documents)
    random.shuffle(documents)

    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    roc_l, pr_l = [], []
    for foldid, (train, test) in enumerate(kfold.split(documents)):
        print("Fold {}".format(foldid))
        train_data = documents[train]
        test_data = documents[test]
        roc, pr = train_and_test(opt, args, w2i, train_data, test_data, reviews)
        print("ROC {} PR {}".format(roc, pr))
        roc_l.append(roc)
        pr_l.append(pr)

    print("Summary : ROC {} PR {}".format(np.mean(roc_l), np.mean(pr_l)))
