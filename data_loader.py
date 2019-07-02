import resource
from sklearn.utils import shuffle
import tensorflow as tf
import subprocess
import json
import seaborn as sns
import matplotlib.pyplot as plt
import re
import hashlib
import numpy as np
import os,sys
import pickle
import gensim.models as w2v
import pandas as pd
import nltk
import generateFeatures as feat
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.FATAL)

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")

modelpath = cwd+"/w2vModel"
pos_samples_path = cwd+"/goldData/pos_samples.csv"
results_path = cwd+"/goldData/results.csv"
dataframepath = cwd+"/data1.csv"
data_dir = cwd+"/data"
bert_dir = cwd+"/res/bert"
hash_path = cwd+"/data/hash.txt"
spec_dir = cwd+"/res/specificity/Domain-Agnostic-Sentence-Specificity-Prediction"

#max num of words in a sentence
MAX_SENT_LEN = 3000
EMB_DIM_SIZE = 300
BERT_DIM = 786
FEATS_DIM = 94

def clean_text(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"’", "\'", string)
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

def pos(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagsAndTokes = nltk.pos_tag(tokens)
    justTags = " ".join([e[1] for e in tagsAndTokes])
    return justTags

def save_p(filename, data):
    with open(filename, "wb") as p:
        pickle.dump(data, p)

def read_p(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def check_hash(df_hash, num_folds):
    print("Checking Hash Path on: ",hash_path)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as h:
            lines = h.readlines()
            old_hash = lines[0][:-1]
            old_folds = lines[1]
            print("Old Hash: ",old_hash)
            print("New Hash: ",df_hash)
            print("Old and new #folds:",old_folds, num_folds)
        if (old_hash == df_hash) and (old_folds == num_folds):
            return True
    return False

def load_data(emb_type='w2v', collapse_classes=False, fold=None, num_folds=1, random_state=None):
    print('Loading data from',pos_samples_path)
    data1 = pd.read_csv(pos_samples_path, sep='\t')
    data_results = pd.read_csv(results_path, sep=',')
    data_results = data_results.loc[data_results.value == 1]
    columns = list(set(data_results.columns) & set(data1.columns))
    data_results = data_results.loc[:,columns]
    data_results = data1.sample(frac=1,random_state=random_state)
    data = data1.sample(frac=1,random_state=random_state)
    data = pd.concat([data,data_results])
    data = data.drop_duplicates(keep='first')
    data = data[data['source_body'].map(len) > 50]
    data = data.reset_index()
    json_data = data.to_json().encode()
    df_hash = hashlib.sha256(json_data).hexdigest()

    data = data.drop_duplicates(subset='claim_source_url', keep='first')
    labels = ['false', 'mfalse', 'mixture', 'mtrue', 'true', 'unverified']
    if(collapse_classes):
        data.loc[data['claim_label'] == "mfalse", 'claim_label'] = 'false'
        data.loc[data['claim_label'] == "mtrue", 'claim_label'] = 'true'
        labels = ['false', 'mixture', 'true', 'unverified']

    labels = ['true', 'false']
    data = data.loc[data.claim_label.isin(labels)]
    print(data.claim_label.unique())

    labels_idx = [labels.index(label) for label in labels]
    labels_one_hot = np.eye(len(labels))[labels_idx]
    label_to_oh = {label:labels_one_hot[labels.index(label)] for label in labels}

    print("MEMORY: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    assert (num_folds > 2), "Needs at least three folds for Dev/Train/Test to be different from each other"
    #generate and save the folds:
    for fold in range(num_folds):
        bucket_size = int(len(data.index)/num_folds)
        fold_dev = fold+1
        if fold == num_folds-1:
            fold_dev = 0

    if not check_hash(df_hash, str(num_folds)):
        print("Processing New Data...")

        data = [(clean_text(pd.Series(e[1])['source_body']),pd.Series(e[1])['claim_label'])for e in list(data.iterrows())]
        df = pd.DataFrame(data, columns=["body","label"])

        lens = pd.Series([len(e.split(" ")) for e in df['body'].values])
        df = df[lens < MAX_SENT_LEN]
        num_entries = len(df)
        lens = np.asarray([len(e.split(" ")) for e in df['body'].values])
        df = df.reset_index(drop=True)
        df.to_csv(data_dir+'/data.csv', index=False)

        #plots the data distribution by number of words
        print("Number of entries: ", num_entries)
        print("Mean and Std of number of words per document: ",np.mean(lens),np.std(lens))
        #sns.distplot(lens)
        #plt.show()

        #Generate BERT embeddings
        with open(cwd+'/res/bert/input.txt', 'w+') as f:
            for i,e in df.iterrows():
                f.write(e['body']+'\n')
        cmd1 = "python3 extract_features.py   --input_file=input.txt   --output_file=output4layers.json   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --layers=-1,-2,-3,-4  --max_seq_length=512   --batch_size=32"
        subprocess.call(cmd1, shell=True, cwd = bert_dir)
        print("BERT Embeddings Saved")

        #Generate the features ndarray and save it to a pickle
        feat.generate_specificity()
        feat.generate_complexity()
        features = []
        for idx,e in df.iterrows():
            print("Generating Features: ",idx+1,"out of ",len(df))
            feature = feat.vectorize(e[0],idx)
            features.append(feature)
        features = np.array(features).astype(np.float)
        print(features.shape)
        with open(data_dir+"/features", "wb") as p:
            pickle.dump(features, p)
        print("Generated Features. Saved to pickle.")

        #features = pickle.load(open(data_dir+"/features", "rb"))

        #normalize features
        features = np.nan_to_num(features)
        features_t = features.T
        for c in range(features_t.shape[0]):
            row = features_t[c]
            features_t[c] = np.interp(row, (np.min(row), np.max(row)), (-2, +2))
        features = features_t.T
        print(features.shape)

        print("MEMORY AFTER FEATURES: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        #generate w2v embeddings from data
        w2v_model = w2v.KeyedVectors.load_word2vec_format(cwd+'/data/GoogleNews-vectors-negative300.bin', binary=True)
        embeddings = []
        for idx, row in df.iterrows():
            words = row['body'].split(" ")
            words = filter(lambda x: x in w2v_model.vocab, words)
            embedding = [w2v_model.wv[word] for word in words]
            masked = np.zeros((MAX_SENT_LEN, EMB_DIM_SIZE))
            mask_len = min(len(embedding),MAX_SENT_LEN)
            masked[MAX_SENT_LEN-mask_len:] = embedding[:mask_len]
            embeddings.append(masked)

        with open(data_dir+"/masked_embeddings", "wb") as p:
            pickle.dump(embeddings, p)
        print("Masked Embeddings Saved")

        print("MEMORY AFTER W2V: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        #concatenate features and w2vmasked_embeddings
        embeddings = np.array(embeddings, dtype=np.float32)
        features_broad = np.array([features] * MAX_SENT_LEN)
        features_broad = np.swapaxes(features_broad, 0,1)
        print(embeddings.shape)
        print(features_broad.shape)
        w2v_post_data = np.concatenate((features_broad,embeddings), axis=2)
        print("W2V+Features shape: ",w2v_post_data.shape)

        #print("MEMORY AFTER CONCAT: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        #concatenate features and bert_embeddings
        with open(bert_dir+"/output4layers.json", "r+") as f:
            bert = [json.loads(l)['features'][0]['layers'][0]['values'] for l in f.readlines()]

        bert_post_data = np.concatenate((features,bert), axis=1)
        print("Bert+Features shape: ",bert_post_data.shape)

        #shuffles (same permutation) the three arrays while splitting them
        labels = [label_to_oh[label].tolist() for label in df['label'].values.tolist()]
        labels, w2v_post_data, bert_post_data = shuffle(labels, w2v_post_data, bert_post_data, random_state=0)
        label_folds = np.array_split(labels, num_folds)
        only_w2v_folds = np.array_split(embeddings, num_folds, axis=0)
        w2v_folds = np.array_split(w2v_post_data, num_folds, axis=0)
        bert_folds = np.array_split(bert_post_data, num_folds)
        only_bert_folds = np.array_split(bert, num_folds)

        print("MEMORY AFTER FOLDS: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        #saves one dumpfile for each fold
        subprocess.call("rm -rf "+data_dir+"/folds/*", shell=True, cwd=data_dir)

        for i in range(num_folds):
            fold_dir = data_dir+"/folds/"+str(i)
            os.mkdir(fold_dir)
            save_p(fold_dir+"/w2v", w2v_folds[i])
            save_p(fold_dir+"/only_w2v", only_w2v_folds[i])
            save_p(fold_dir+"/labels", label_folds[i])
            save_p(fold_dir+"/bert", bert_folds[i])
            save_p(fold_dir+"/only_bert", only_bert_folds[i])

        print("MEMORY AFTER FOLDS SAVING: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        #returns the selected emb type (bert/w2v)
        test_data = read_p(data_dir+"/folds/"+str(fold)+"/"+emb_type)
        test_target = read_p(data_dir+"/folds/"+str(fold)+"/labels")

        dev_data = read_p(data_dir+"/folds/"+str(fold_dev)+"/"+emb_type)
        dev_target = read_p(data_dir+"/folds/"+str(fold_dev)+"/labels")

        train_data_filenames = [data_dir+"/folds/"+str(i)+"/"+emb_type for i in range(num_folds) if i not in [fold,fold_dev]]
        train_data = np.concatenate([read_p(fn) for fn in train_data_filenames], axis=0)
        train_target_filenames = [data_dir+"/folds/"+str(i)+"/labels" for i in range(num_folds) if i not in [fold,fold_dev]]
        train_target = np.concatenate([read_p(fn) for fn in train_target_filenames], axis=0)

        print("Train Shape: %s\nDev Shape: %s\nTest Shape: %s" % (train_data.shape, dev_data.shape, test_data.shape))
        print('Generation of data done!')
        #select what to return

        with open(hash_path, "w") as h:
            h.write(df_hash+"\n")
            h.write(str(num_folds))

        return train_data, train_target, dev_data, dev_target, test_data, test_target, label_to_oh

    else:
        print("Reading already processed data")
        #returns the selected emb type (bert/w2v)
        test_data = read_p(data_dir+"/folds/"+str(fold)+"/"+emb_type)
        test_target = read_p(data_dir+"/folds/"+str(fold)+"/labels")

        dev_data = read_p(data_dir+"/folds/"+str(fold_dev)+"/"+emb_type)
        dev_target = read_p(data_dir+"/folds/"+str(fold_dev)+"/labels")

        train_data_filenames = [data_dir+"/folds/"+str(i)+"/"+emb_type for i in range(num_folds) if i not in [fold,fold_dev]]
        train_data = np.concatenate([read_p(fn) for fn in train_data_filenames], axis=0)
        train_target_filenames = [data_dir+"/folds/"+str(i)+"/labels" for i in range(num_folds) if i not in [fold,fold_dev]]
        train_target = np.concatenate([read_p(fn) for fn in train_target_filenames], axis=0)

        return train_data, train_target, dev_data, dev_target, test_data, test_target, label_to_oh
