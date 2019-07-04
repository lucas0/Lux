import re
import numpy as np
import os
from gensim.models import word2vec
import pandas as pd
import nltk

cwd = os.getcwd()

modelpath = cwd+"/w2vModel"
pos_samples_path = cwd+"/pos_samples.csv"
dataframepath = cwd+"/data1.csv"

def clean_text(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip()

def load_LSTM_W2V():
    print('loading data from',dataframepath)
    w2v_model = word2vec.Word2Vec.load(modelpath)
    data1 = pd.read_csv(dataframepath,sep='\t')
    data1 = data1.sample(frac=1)

    # dev_data = [(sent, 1) for sent in pos_sents[4250:4800]] + [(sent, 0) for sent in neg_sents[4250:4800]]
    train_data = list(zip(data1.iloc[:265]['embeddings'].tolist(),data1.iloc[:265]['target'].tolist()))
    test_data = list(zip(data1.iloc[265:]['embeddings'].tolist(),data1.iloc[265:]['target'].tolist()))
    # test_data = [e[0] for e in list(data1.iterrows())[265:]]
    
    print('train:',len(train_data),'test:',len(test_data))

    word_to_ix = {word: w2v_model[word] for word in w2v_model.wv.vocab}
    labels = ['false', 'mfalse', 'mixture', 'mtrue', 'true', 'unverified']
    labels_idx = [labels.index(label) for label in labels]
    targets = np.eye(len(labels))[labels_idx]
    label_to_ix = {label:targets[labels.index(label)] for label in labels}
    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')
    return train_data,test_data,word_to_ix,label_to_ix

def load_LSTM_data(collapse_classes=False, fold=None, random_state=None):
    print('loading data from',pos_samples_path)
    data1 = pd.read_csv(pos_samples_path,sep='\t')
    non_dup = data1.drop_duplicates()
    data = non_dup.sample(frac=1, random_state=random_state)

    labels = ['false', 'mfalse', 'mixture', 'mtrue', 'true', 'unverified']

    if(collapse_classes):
        data.loc[data['claim_label'] == "mfalse", 'claim_label'] = 'false'
        data.loc[data['claim_label'] == "mtrue", 'claim_label'] = 'true'
        print(data.claim_label.unique())
        labels = ['false', 'mixture', 'true', 'unverified']

    bucket_size = int(len(data.index)/9)

    fold_dev = fold+1
    if fold == 8:
        fold_dev = 0

    dev_data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())[bucket_size*(fold_dev):bucket_size*(fold_dev+1)]]
    test_data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())[bucket_size*(fold):bucket_size*(fold+1)]]
    
    train_data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())]
    
    for e in dev_data+test_data:
        if e in train_data:
            train_data.remove(e)

    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))

    raw_text = " ".join([e[0] for e in train_data+test_data+dev_data])
    vocab = set(raw_text.split(" "))
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    labels_idx = [labels.index(label) for label in labels]
    targets = np.eye(len(labels))[labels_idx]
    label_to_ix = {label:targets[labels.index(label)] for label in labels}

    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')

    return train_data, dev_data, test_data, word_to_ix, label_to_ix

def pos(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagsAndTokes = nltk.pos_tag(tokens)
    justTags = " ".join([e[1] for e in tagsAndTokes])
    return justTags

def load_LSTM_pos(collapse_classes=False, fold=None, random_state=None):
    print('loading data from',pos_samples_path)
    data1 = pd.read_csv(pos_samples_path,sep='\t')
    data = data1.sample(frac=1,random_state=random_state)

    labels = ['false', 'mfalse', 'mixture', 'mtrue', 'true', 'unverified']

    if(collapse_classes):
        data.loc[data['claim_label'] == "mfalse", 'claim_label'] = 'false'
        data.loc[data['claim_label'] == "mtrue", 'claim_label'] = 'true'
        print(data.claim_label.unique())
        labels = ['false', 'mixture', 'true', 'unverified']

    bucket_size = int(len(data.index)/9)

    fold_dev = fold+1
    if fold == 8:
        fold_dev = 0

    dev_data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())[bucket_size*(fold_dev):bucket_size*(fold_dev+1)]]
    test_data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())[bucket_size*(fold):bucket_size*(fold+1)]]
    
    train_data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())]
    
    for e in dev_data+test_data:
        if e in train_data:
            train_data.remove(e)


    # train_data is the whole dataset without the entries contained in dev + test
    # train_data.concat([train_data, dev_data, test_data]).drop_duplicates(keep=False)

    print('train:',len(train_data),'dev:',len(dev_data),'test:',len(test_data))

    # it should be called POS-tag
    raw_text = " ".join([e[0] for e in train_data+test_data+dev_data])
    vocab = set(raw_text.split(" "))
    vocab_size = len(vocab)
    word_to_ix = {word: i for i, word in enumerate(vocab)}

    labels_idx = [labels.index(label) for label in labels]
    targets = np.eye(len(labels))[labels_idx]
    label_to_ix = {label:targets[labels.index(label)] for label in labels}

    print('vocab size:',len(word_to_ix),'label size:',len(label_to_ix))
    print('loading data done!')

    return train_data, dev_data, test_data, word_to_ix, label_to_ix
