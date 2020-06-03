import linecache
import traceback
import resource
import filecmp
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
data_dir = cwd+"/data"
dataset_dir = data_dir+"/datasets"
bert_dir = cwd+"/res/bert"
hash_path = cwd+"/data/hash.txt"
spec_dir = cwd+"/res/specificity/Domain-Agnostic-Sentence-Specificity-Prediction"

#max num of words in a sentence
MAX_SENT_LEN = 3000
MIN_BODY_LEN = 300
EMB_DIM_SIZE = 300
BERT_DIM = 786
FEATS_DIM = 94

stages = ["data","folds","bert","features","complexity","specificity","w2v"]

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

def pos(sentence):
    tokens = nltk.word_tokenize(sentence)
    tagsAndTokes = nltk.pos_tag(tokens)
    justTags = " ".join([e[1] for e in tagsAndTokes])
    return justTags

def save_p(filename, data):
    with open(filename, "wb") as p:
        pickle.dump(data, p, protocol=4)

def read_p(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def check_hash(df_hash, num_folds, stage="data"):
    print(stage.upper(), " CHECK> Checking Hash Path on: ",hash_path)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as h:
            lines = h.readlines()
            old_folds = lines[1].split("folds")[1][2:-1]
            for i,l in enumerate(lines):
                if l.startswith(stage):
                    old_hash = l.split(stage)[1][2:-1]

            print("Old and New Hash: ",old_hash[:5],df_hash[:5]," Same? ", (old_hash == df_hash))
            print("Old and New #folds:",old_folds, num_folds, " Same? ", (int(old_folds)==num_folds), "\n")
        if old_hash != df_hash:
            return False
        if stage == "data" and (int(old_folds) != num_folds):
            return False
        return True
    return False

def savehash(stage, hashcode):
    with open(hash_path, "r") as h:
        data = h.readlines()
        for i,l in enumerate(data):
            if l.startswith(stage):
                data[i] = stage+": "+hashcode+"\n"
    with open(hash_path, "w") as h:
        h.writelines(data)

def reset_hash():
    for s in stages:
        savehash(s,"0")

#run concat+normalize.py inside dataset/ before loading data
def load_data(emb_type='w2v', collapse_classes=False, fold=None, num_folds=1, random_state=None, force_reload=False):
    print('Loading data from',dataset_dir)
    data = pd.read_csv(dataset_dir+"/dataset.csv", sep=',')

    if force_reload: reset_hash()

    print("size of initial \"dataset\":",len(data))
    data = data.drop_duplicates(subset='o_url', keep='first')
    print("after dropping duplicates:",len(data))
    data.o_body = data.o_body.astype('str')
    data.verdict = data.verdict.astype('str')
    data['verdict'] = data['verdict'].str.lower()
    data = data[data['o_body'].map(len) > MIN_BODY_LEN]
    print("after dropping origins with less than "+str(MIN_BODY_LEN)+" chars:",len(data))
    data = data.reset_index()

    if(collapse_classes):
        print("labels before collapse classes:", data.verdict.unique())
        data.loc[data['verdict'] == "mfalse", 'verdict'] = 'false'
        data.loc[data['verdict'] == "mtrue", 'verdict'] = 'true'

    labels = ['true', 'false']
    print(data['verdict'].value_counts())
    data = data.loc[data.verdict.isin(labels)]
    print("considered labels:", data.verdict.unique())
    print("after dropping invalid labels:",len(data))

    #creating hash
    json_data = data.to_json().encode()
    data = data.sample(frac=1, random_state=random_state)
    df_hash = hashlib.sha256(json_data).hexdigest()

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

    if not check_hash(df_hash, num_folds):
        #input("Processing New Data. Press ENTER to start...")

        df = data[['o_body','verdict']].copy()
        df = df.rename(columns={"o_body": "body"})
        df.body.apply(clean_text)
        df.verdict.apply(clean_text)

        lens = np.asarray([len(e.split(" ")) for e in df['body'].values])
        #df = df[lens < MAX_SENT_LEN]
        df.reset_index(drop = True, inplace = True)
        df.to_csv(data_dir+'/data.csv', sep="\t", index=False)
        num_entries = len(df)

        #plots the data distribution by number of words
        print("Number of entries: ", num_entries)
        print("True/False: ",df.groupby('verdict').count())
        print("Mean and Std of number of words per document: ",np.mean(lens),np.std(lens), "\n")
        #sns.distplot(lens)
        #plt.show()

        ##################################
        ########## BERT TRAIN ############
        ##################################

        #check if new bert should be generated
        if not check_hash(df_hash, num_folds, stage="bert"):
            try:
                inputfile = cwd+'/res/bert/input.txt'
                #copy sentences so BERT is generated
                with open(inputfile, 'w+') as f:
                    for i,e in df.iterrows():
                        b = re.sub("\n", " ", e['body'])
                        f.write(b+'\n')

                #Generate BERT embeddings if data is new
                #removes the output file so a new one is generated
                cmd0 = "rm "+cwd+"/res/bert/output4layers.json"
                subprocess.call(cmd0, shell=True, cwd = bert_dir)

                #generates new BERT embeddings
                cmd1 = "python3 extract_features.py   --input_file=input.txt   --output_file=output4layers.json   --vocab_file=$BERT_BASE_DIR/vocab.txt   --bert_config_file=$BERT_BASE_DIR/bert_config.json   --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt   --layers=-1,-2,-3,-4  --max_seq_length=512   --batch_size=32"
                subprocess.call(cmd1, shell=True, cwd = bert_dir)
            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while fine training BERT. Press any key to exit.")
                sys.exit(1)
            print("BERT Embeddings Saved")
            savehash("bert", df_hash)


        ###################################
        ############# FEATURES ############
        ###################################

        #check if new linguistic features should be generated
        if not check_hash(df_hash, num_folds, stage="complexity"):
            #Generate the features ndarray and save it to a pickle
            try:
                feat.generate_complexity()
            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while GENERATING COMPLEXITY. Press any key to exit.")
                sys.exit(1)
            savehash("complexity", hashcode=df_hash)
        if not check_hash(df_hash, num_folds, stage="specificity"):
            try:
                feat.generate_specificity()
            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while GENERATING SPECIFICITY. Press any key to exit.")
                sys.exit(1)
            savehash("specificity", hashcode=df_hash)
        if not check_hash(df_hash, num_folds, stage="features"):
            try:
                features = []
                for idx,e in list(df.iterrows()):
                    print("Generating Features: ",idx+1,"out of ",len(df))
                    feature = feat.vectorize(e[0],idx)
                    features.append(feature)
                features = np.array(features).astype(np.float)

            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while GENERATING FEATURES. Press any key to exit.")
                sys.exit(1)
            save_p(data_dir+"/features", features)
            print("Generated Features. Saved to pickle.")
            print("Features Shape:", features.shape)
            savehash("features", hashcode=df_hash)

        features = read_p(data_dir+"/features")

        #normalize features
        features = np.nan_to_num(features)
        features_t = features.T
        for c in range(features_t.shape[0]):
            row = features_t[c]
            features_t[c] = np.interp(row, (np.min(row), np.max(row)), (-2, +2))
        features = features_t.T

        print("MEMORY AFTER FEATURES: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        ###################################
        ############### W2V ###############
        ###################################

        if not check_hash(df_hash, num_folds, stage="w2v"):
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

            embeddings = np.array(embeddings, dtype=np.float32)
            save_p(data_dir+"/masked_embeddings", embeddings)
            print("Masked Embeddings Saved")
            savehash("w2v", hashcode=df_hash)

        print("MEMORY AFTER W2V: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        #########################################
        ## CONCATENATION, SHUFFLING AND SAVING ##
        #########################################

        #delete folds folders
        subprocess.call("rm -rf "+data_dir+"/folds/*", shell=True, cwd=data_dir)

        #creates the shuffle order
        index_shuf = list(range(len(df)))

        #LABELS
        labels = [label_to_oh[label].tolist() for label in df['verdict'].values.tolist()]
        labels = [labels[i] for i in index_shuf]
        label_folds = np.array_split(labels, num_folds)
        for i in range(num_folds):
            fold_dir = data_dir+"/folds/"+str(i)
            os.mkdir(fold_dir)
            save_p(fold_dir+"/labels", label_folds[i])

        ##W2V
        #embeddings = read_p(data_dir+"/masked_embeddings")
        #features_broad = np.array([features] * MAX_SENT_LEN)
        #features_broad = np.swapaxes(features_broad, 0,1)
        #w2v_post_data = np.concatenate((features_broad,embeddings), axis=2)
        #print("W2V+Features shape: ",w2v_post_data.shape)
        #w2v_post_data = [w2v_post_data[i] for i in index_shuf]
        #only_w2v_folds = np.array_split(embeddings, num_folds, axis=0)
        #w2v_folds = np.array_split(w2v_post_data, num_folds, axis=0)
        #for i in range(num_folds):
        #    fold_dir = data_dir+"/folds/"+str(i)
        #    save_p(fold_dir+"/w2v", w2v_folds[i])
        #    save_p(fold_dir+"/only_w2v", only_w2v_folds[i])

        bert_folds = np.array_split(index_shuf, num_folds)
        bert_folds = [a.tolist() for a in bert_folds]

        fold_idx = [bert_folds.index(list(sl)) for e in index_shuf for sl in bert_folds if e in list(sl)]

        flag = {idx:False for idx in range(len(bert_folds))}
        for fold, idx in zip(fold_idx, index_shuf):
            b_line = linecache.getline(bert_dir+"/output4layers.json", idx+1)
            b_values = json.loads(b_line)['features'][0]['layers'][0]['values']
            entry = np.concatenate((features[idx,:],b_values))
            #print("lenghts:",len(features[idx,:]), len(b_values), len(entry))
            feat_df = pd.DataFrame([entry], columns=['f'+str(e) for e in range(len(entry))])

            feat_df.to_csv(data_dir+"/folds/"+str(fold)+"/features+bert.csv", mode='a', index=False, header=flag[fold])
            flag[fold] = False

        linecache.clearcache()

        for i in range(num_folds):
            fold_dir = data_dir+"/folds/"+str(i)
            bert = np.genfromtxt(fold_dir+"/features+bert.csv", delimiter=',')
            print("saving bert fold ",str(i), bert.shape)
            save_p(fold_dir+"/bert", bert)
            #save_p(fold_dir+"/only_bert", only_bert_folds[i])


        print("MEMORY AFTER FOLDS SAVING: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        print('Generation of data done!')

        savehash("data", hashcode=df_hash)
        savehash("folds", hashcode=str(num_folds))

        return load_data(emb_type=emb_type, collapse_classes=collapse_classes, fold=fold, num_folds=num_folds, random_state=random_state)

    else:
        print("Reading already processed data")
        #returns the selected emb type (bert/w2v)
        test_data = read_p(data_dir+"/folds/"+str(fold)+"/"+emb_type)
        test_target = read_p(data_dir+"/folds/"+str(fold)+"/labels")

        dev_data = read_p(data_dir+"/folds/"+str(fold_dev)+"/"+emb_type)
        #dev_data = np.ndarray(dev_data)
        dev_target = read_p(data_dir+"/folds/"+str(fold_dev)+"/labels")

        train_data_filenames = [data_dir+"/folds/"+str(i)+"/"+emb_type for i in range(num_folds) if i not in [fold,fold_dev]]
        train_data = np.concatenate([read_p(fn) for fn in train_data_filenames], axis=0)
        train_target_filenames = [data_dir+"/folds/"+str(i)+"/labels" for i in range(num_folds) if i not in [fold,fold_dev]]
        train_target = np.concatenate([read_p(fn) for fn in train_target_filenames], axis=0)

        return train_data, train_target, dev_data, dev_target, test_data, test_target, label_to_oh

