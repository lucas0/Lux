# -*- coding: utf-8 -*-
import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_loader
import os
import random
import numpy as np
from sklearn.metrics import f1_score
from statistics import mean

torch.set_num_threads(8)
torch.manual_seed(1)
random.seed(1)

dtype = torch.cuda.FloatTensor

class LSTMClassifier(nn.Module):

    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, batch_size):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.batch_size = batch_size
        # self.lstm = nn.LSTM(embedding_dim, hidden_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, bidirectional=True)
        # self.hidden2label = nn.Linear(hidden_dim, label_size)
        self.hidden2label = nn.Linear(hidden_dim*2, label_size)
        self.hidden = self.init_hidden()

    def init_hidden(self):
        # the first is the hidden h
        # the second is the cell  c
        # return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).type(dtype),
        #         autograd.Variable(torch.zeros(1, 1, self.hidden_dim)).type(dtype))
        return (autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).type(dtype),
                autograd.Variable(torch.zeros(2, 1, self.hidden_dim)).type(dtype))

    def forward(self, sentence):
        embeds = self.word_embeddings(sentence)
        x = embeds.view(len(sentence), self.batch_size , -1)
        lstm_out, self.hidden = self.lstm(x, self.hidden)
        y  = self.hidden2label(lstm_out[-1])
        log_probs = F.log_softmax(y, dim=1)
        return log_probs

def get_accuracy(truth, pred):
     assert len(truth)==len(pred)
     right = 0

     labels = []
     truth_labels = []

     for i in range(len(truth)):
        _,label = pred[i].max(1)
        truth_label = np.argmax(truth[i])
        if truth_label==label.data.cpu().numpy()[0]:
            right += 1.0
        labels.append(label.data.cpu().numpy()[0])
        truth_labels.append(truth_label)

     f1 = f1_score(truth_labels, labels, average=None)
     return (right/len(truth),f1)


def get_frequency(truth, pred):
    truth_list = [np.argmax(e) for e in truth]
    pred_list = [e.max(1)[1].data.cpu().numpy()[0] for e in pred]
    return np.bincount(truth_list),np.bincount(pred_list)

def train(fold, root):
    train_data, dev_data, test_data, word_to_ix, label_to_ix = data_loader.load_LSTM_pos(collapse_classes=True, fold=fold, random_state=root)
    EMBEDDING_DIM = 32
    HIDDEN_DIM = 50
    EPOCH = 10
    model = LSTMClassifier(embedding_dim=EMBEDDING_DIM,hidden_dim=HIDDEN_DIM,
                           vocab_size=len(word_to_ix),label_size=len(label_to_ix), batch_size=1)
    model.cuda()
    loss_function = nn.MultiLabelSoftMarginLoss()
    optimizer = optim.Adam(model.parameters(),lr = 1e-3)
    # optimizer = torch.optim.SGD(model.parameters(), lr = 1e-3)
    no_up = 0
    for i in range(EPOCH):
        random.shuffle(train_data)
        print('fold: %d epoch: %d start!' %(fold, i))
        train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i)
        dev_acc = evaluate(model, dev_data, loss_function, word_to_ix, label_to_ix)
    test_acc,test_f1 = evaluate(model, test_data, loss_function, word_to_ix, label_to_ix,"test")
    return test_acc,test_f1

def evaluate(model, data, loss_function, word_to_ix, label_to_ix, name ='dev'):
    model.eval()

    avg_loss = 0.0
    truth_res = []
    pred_res = []

    for sent, label in data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()
        sent = autograd.Variable(torch.cuda.LongTensor([word_to_ix[w] for w in sent.split(' ')]))
        label = autograd.Variable(torch.cuda.FloatTensor([int(e) for e in label_to_ix[label].tolist()]))
        pred = model(sent)
        pred_res.append(pred)

        loss = loss_function(pred, label.unsqueeze(0))
        avg_loss += loss.data.item()
    avg_loss /= len(data)
    acc,f1 = get_accuracy(truth_res, pred_res)
    print(name + ' avg_loss: %g , acc: %g' % (avg_loss, acc))
    print("F1(per class):",f1)
    dist = get_frequency(truth_res,pred_res)
    print(name + ' dist:',dist[0],dist[1])
    return acc,f1

def train_epoch(model, train_data, loss_function, optimizer, word_to_ix, label_to_ix, i):
    model.train()

    avg_loss = 0.0
    count = 0
    truth_res = []
    pred_res = []
    batch_sent = []

    for sent, label in train_data:
        truth_res.append(label_to_ix[label])
        # detaching it from its history on the last instance.
        model.hidden = model.init_hidden()

        sent = autograd.Variable(torch.cuda.LongTensor([word_to_ix[w] for w in sent.split(' ')]))
        label = autograd.Variable(torch.cuda.FloatTensor([int(e) for e in label_to_ix[label].tolist()]))

        pred = model(sent)
        pred_res.append(pred)
        model.zero_grad()

        loss = loss_function(pred, label.unsqueeze(0))
        avg_loss += loss.data.item()
        count += 1

        loss.backward()
        optimizer.step()
    avg_loss /= len(train_data)
    acc, f1 = get_accuracy(truth_res,pred_res)
    print('epoch: %d done! \ntrain avg_loss: %g, acc: %g'%(i, avg_loss, acc))
    print("F1(per class):",f1)

num_folds = 9
seed = random.randint(0,10090000)
results = [train(e,seed) for e in range(num_folds)]
acc = [e[0] for e in results]
f1 = [e[1] for e in results]

avg_acc = (sum(acc)/float(len(acc)))

avg_f1 = ["{0:.4f}".format(sum(col) / float(len(col))) for col in zip(*f1)]

str_avg = '{0:.4f}'.format(avg_acc)
print(str(num_folds)+"-fold acc:"+str_avg)
print("F1_avg: ",avg_f1)

