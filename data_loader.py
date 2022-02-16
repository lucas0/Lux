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
from bert_serving.server.helper import get_args_parser
from bert_serving.server.helper import get_shutdown_parser
from bert_serving.server import BertServer
from bert_serving.client import BertClient
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
#only used for w2v
MAX_SENT_LEN = 2000
MIN_BODY_LEN = 50
EMB_DIM_SIZE = 300
#EMB_DIM_SIZE = 100

stages = ["concat","data","folds","bert","features","complexity","specificity","w2v"]

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

class LineSeekableFile:
    def __init__(self, seekable):
        self.c = 0
        self.fin = seekable
        self.line_map = list() # Map from line index -> file position.
        self.line_map.append(0)
        while seekable.readline():
            print("Creating LineSeekableFile Object: ",self.c)
            self.c += 1
            self.line_map.append(seekable.tell())

    def __getitem__(self, index):
        # NOTE: This assumes that you're not reading the file sequentially.
        # For that, just use 'for line in file'.
        self.fin.seek(self.line_map[index])
        return self.fin.readline()

def save_p(filename, data):
    with open(filename, "wb") as p:
        pickle.dump(data, p, protocol=4)

def read_p(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)

def check_hash(df_hash, num_folds, drop_feat_idx=None, stage="data"):
    print(stage.upper(), " CHECK> Checking Hash Path on: ",hash_path)
    if os.path.exists(hash_path):
        with open(hash_path, "r") as h:
            lines = h.readlines()
            old_folds = lines[1].split(" ")[1][:-1]
            old_feat_idx = lines[-2].split("drop_feat: ")[1][:-1]
            print("old_feat_idx:", old_feat_idx)
            for i,l in enumerate(lines):
                if l.startswith(stage):
                    old_hash = l.split(" ")[1][:-1]

            print("Old and New Hash: ",old_hash[:5],df_hash[:5]," Same? ", (old_hash == df_hash))
            print("Old and New #folds:",old_folds, num_folds, " Same? ", (int(old_folds)==num_folds))
            print("Old and New Drop_Feat: ",old_feat_idx,drop_feat_idx," Same? ", (old_feat_idx == str(drop_feat_idx)),"\n")

        if old_hash != df_hash:
            return False
        if stage in ["drop_feat", "data"] and (old_feat_idx != str(drop_feat_idx)):
            return False
        if stage == "data" and (int(old_folds) != num_folds):
            return False
        return True
    return False

def savehash(stage, hashcode, drop_feat_idx=None):
    with open(hash_path, "r") as h:
        data = h.readlines()
        for i,l in enumerate(data):
            if l.startswith(stage):
                if stage == "drop_feat":
                    data[i] = stage+": "+str(drop_feat_idx)+"\n"
                else:
                    data[i] = stage+": "+hashcode+"\n"
    with open(hash_path, "w") as h:
        h.writelines(data)

def reset_hash(force_reload):
    if force_reload == "all" :
        for s in stages:
            savehash(s,"0")
    if force_reload == "emb" :
        for s in ["data", "bert", "concat"]:
            savehash(s,"0")
    if force_reload == "feat" :
        for s in ["data", "features", "concat", "complexity", "specificity"]:
            savehash(s,"0")
    if force_reload == "just_reload" :
        for s in ["data", "concat"]:
            savehash(s,"0")

#run concat+normalize.py inside dataset/ before loading data
def load_data(emb_type='w2v', collapse_classes=False, fold_test=None, num_folds=1, random_state=None, force_reload=None, drop_feat_idx=[], only_claims=False, feature_list=["inf","div","qua","aff","sbj","spe","pau","unc","pas"]):
    print('Loading data from',dataset_dir)
    data = pd.read_csv(dataset_dir+"/dataset.csv", sep=',')

    if force_reload is not None: reset_hash(force_reload)

    print("size of initial \"dataset\":",len(data))
    #determines if data will be whole body or only claims
    if only_claims:
        df = data[['claim','verdict']].copy()
        df = df.rename(columns={"claim": "body"})
    else:
        df = data[['o_body','verdict']].copy()
        df = df.rename(columns={"o_body": "body"})
    if 'o_url' in df.columns:
        df = df.drop_duplicates(subset='o_url', keep='first')
    print("after dropping duplicates:",len(df))
    df.body = df.body.astype('str')
    df.verdict = df.verdict.astype('str')
    df['verdict'] = df['verdict'].str.lower()
    df.body.apply(clean_text)
    df = df.reset_index()

    if(collapse_classes):
        print("labels before collapse classes:", df.verdict.unique())
        df.loc[df['verdict'] == "mfalse", 'verdict'] = 'false'
        df.loc[df['verdict'] == "false.", 'verdict'] = 'false'
        df.loc[df['verdict'] == "mtrue", 'verdict'] = 'true'
        df.loc[df['verdict'] == "true.", 'verdict'] = 'true'

    labels = ['true', 'false']
    print(df['verdict'].value_counts())
    df = df.loc[df.verdict.isin(labels)]
    print("considered labels:", df.verdict.unique())
    print("after dropping invalid labels:",len(df))

    #creating hash
    json_data = df.to_json().encode()
    df = df.sample(frac=1, random_state=random_state)
    df_hash = hashlib.sha256(json_data).hexdigest()

    labels_idx = [labels.index(label) for label in labels]
    labels_one_hot = np.eye(len(labels))[labels_idx]
    label_to_oh = {label:labels_one_hot[labels.index(label)] for label in labels}

    print("MEMORY: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

    assert (num_folds > 2), "Needs at least three folds for Train/Dev/Test to be different from each other"
    #determine train_fold and dev_fold:
    bucket_size = int(len(data.index)/num_folds)
    fold_dev = 0 if fold_test == num_folds-1 else fold_test+1

    if not check_hash(df_hash, num_folds, drop_feat_idx=drop_feat_idx):
        lens = np.asarray([len(e.split(" ")) for e in df['body'].values])
        #df = df[lens < MAX_SENT_LEN]
        df = df[df['body'].map(len) > MIN_BODY_LEN]
        print("after dropping origins with less than "+str(MIN_BODY_LEN)+" chars:",len(df))
        df.reset_index(drop = True, inplace = True)
        df.to_csv(data_dir+'/data.csv', sep="\t", index=False)
        num_entries = len(df)

        #plots the data distribution by number of words
        print("Number of entries: ", num_entries)
        print("True/False: ",df.groupby('verdict').count())
        print("Mean and Std of number of words per document: ",np.mean(lens),np.std(lens), "\n")
        #input("Press any key to continue.")

        ###################################
        ############# FEATURES ############
        ###################################
        #check if new linguistic features should be generated
        flag_concat = False
        if not check_hash(df_hash, num_folds, stage="complexity") and "inf" in feature_list:
            flag_concat = True
            #Generate the features ndarray and save it to a pickle
            try:
                feat.generate_complexity()
            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while GENERATING COMPLEXITY. Press any key to exit.")
                sys.exit(1)
            savehash("complexity", hashcode=df_hash)
        if not check_hash(df_hash, num_folds, stage="specificity") and "spe" in feature_list:
            flag_concat = True
            try:
                feat.generate_specificity()
            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while GENERATING SPECIFICITY. Press any key to exit.")
                sys.exit(1)
            savehash("specificity", hashcode=df_hash)

        if not check_hash(df_hash, num_folds, drop_feat_idx=drop_feat_idx, stage="features") or (force_reload == 'just_reload'):
            flag_concat = True
            try:
                if force_reload != "just_reload":
                    features = feat.generateFeats(feature_list)
                else:
                    features = feat.generateFeats(feature_list, just_reload=True)
            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while GENERATING FEATURES. Press any key to exit.")
                sys.exit(1)
            save_p(data_dir+"/features", features)
            print("Generated Features. Saved to pickle.")
            savehash("features", hashcode=df_hash, drop_feat_idx=drop_feat_idx)

        #check if drop_features is NOT the same
        if not check_hash(df_hash, num_folds, drop_feat_idx=drop_feat_idx, stage="drop_feat"):
            flag_concat = True
            savehash("drop_feat", hashcode=df_hash, drop_feat_idx=drop_feat_idx)

        print("MEMORY AFTER FEATURES: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

        ####################################
        ############### BERT ###############
        ####################################
        #check if new bert should be generated
        if not check_hash(df_hash, num_folds, stage="bert"):
            try:
                #creates the shuffle order (not random)
                index_shuf = list(range(len(df)))

                #creates a list of N=folds lists, each inner list contains the index of the elements of each fold
                bert_folds = np.array_split(index_shuf, num_folds)
                bert_folds = [a.tolist() for a in bert_folds]

                #creates an ordered list of N=entries of integers(:folds) indicating the fold idx of each entry
                fold_idx = [bert_folds.index(list(sl)) for e in index_shuf for sl in bert_folds if e in list(sl)]

                #start the bert-as-a-service server
                bert_dir = os.environ.get("BERT_BASE_DIR")
                #args = get_args_parser().parse_args(['-model_dir', bert_dir, '-port', '5555', '-port_out', '5556', '-max_seq_len', '512', '-mask_cls_sep'])
                #server = BertServer(args)
                #server.start()

                #delete the bert.csv files inside the folds
                for i in range(num_folds):
                    filename = data_dir+"/folds/"+str(i)+"/bert.csv"
                    if os.path.exists(filename):
                        subprocess.call("rm -rf "+filename, shell=True, cwd=data_dir)

                #TODO make this process read only one fold at a time
                for fold, idx in zip(fold_idx, index_shuf):
                    #generates the encodings for the texts
                    bc = BertClient(check_version=False)
                    b = bc.encode([df.body[idx]])[0]

                    bert_df = pd.DataFrame([b], columns=['f'+str(e) for e in range(len(b))])
                    bert_df.to_csv(data_dir+"/folds/"+str(fold)+"/bert.csv", mode='a+', index=False, header=False)

                #stops the bert-as-a-service server
                #shut_args = get_shutdown_parser().parse_args(['-ip','localhost','-port','5555','-timeout','5000'])
                #server.shutdown(shut_args)

            except Exception as e:
                print(traceback.format_exc())
                input("Error occured while fine training BERT. Press any key to exit.")
                sys.exit(1)

            print("BERT Embeddings Saved")
            savehash("bert", df_hash)

        #########################################
        ## CONCATENATION, SHUFFLING AND SAVING ##
        #########################################

        #if not check_hash(df_hash, num_folds, stage="concat"):
        if flag_concat:
            features = read_p(data_dir+"/features")
            features = np.delete(features,drop_feat_idx,axis=1)

            #normalize features
            features = np.nan_to_num(features)
            features_t = features.T
            for c in range(features_t.shape[0]):
                row = features_t[c]
                features_t[c] = np.interp(row, (np.min(row), np.max(row)), (-2, +2))
            features = features_t.T
            #delete labels and folds folders
            for i in range(num_folds):
                subprocess.call("rm -rf "+data_dir+"/folds/"+str(i)+"/labels", shell=True, cwd=data_dir)
                subprocess.call("rm -rf "+data_dir+"/folds/"+str(i)+"/features+bert.csv", shell=True, cwd=data_dir)
                subprocess.call("rm -rf "+data_dir+"/folds/"+str(i)+"/bert", shell=True, cwd=data_dir)
                subprocess.call("rm -rf "+data_dir+"/folds/"+str(i)+"/only_bert", shell=True, cwd=data_dir)

            #creates the shuffle order (not random)
            index_shuf = list(range(len(df)))

            #LABELS
            labels = [label_to_oh[label].tolist() for label in df['verdict'].values.tolist()]
            labels = [labels[i] for i in index_shuf]
            label_folds = np.array_split(labels, num_folds)

            for i in range(num_folds):
                fold_dir = data_dir+"/folds/"+str(i)
                if not os.path.exists(fold_dir):
                    os.mkdir(fold_dir)
                save_p(fold_dir+"/labels", label_folds[i])

            #creates a list of N=folds lists, each inner list contains the index of the elements of each fold
            bert_folds = np.array_split(index_shuf, num_folds)
            bert_folds = [a.tolist() for a in bert_folds]

            #creates an ordered list of N=entries of integers(:folds) indicating the fold idx of each entry
            fold_idx = [bert_folds.index(list(sl)) for e in index_shuf for sl in bert_folds if e in list(sl)]

            #TODO make this process read only one fold at a time
            for fold in range(num_folds):
                b_fold_csv = pd.read_csv(data_dir+"/folds/"+str(fold)+"/bert.csv", header=None)
                #gets only the indexes
                count = sum([1 for fidx,_ in zip(fold_idx, index_shuf) if fold == fidx])
                for idx in range(count):
                    b = b_fold_csv.iloc[idx]
                    entry = np.concatenate((features[idx,:],b))

                    feat_df = pd.DataFrame([entry], columns=['f'+str(e) for e in range(len(entry))])
                    feat_df.to_csv(data_dir+"/folds/"+str(fold)+"/features+bert.csv", mode='a+', index=False, header=False)

            for i in range(num_folds):
                fold_dir = data_dir+"/folds/"+str(i)
                bert = np.genfromtxt(fold_dir+"/features+bert.csv", delimiter=',')
                only_bert = np.genfromtxt(fold_dir+"/bert.csv", delimiter=',')
                print("saving bert fold ",str(i), bert.shape)
                save_p(fold_dir+"/bert", bert)
                save_p(fold_dir+"/only_bert", only_bert)

            print("MEMORY AFTER FOLDS SAVING: ",resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)

            savehash("concat", hashcode=df_hash)

        checks = ["bert", "features", "concat"]
        if "inf" in feature_list: checks.append("complexity")
        if "spe" in feature_list: checks.append("specificity")

        for e in checks:
            print(e)
            print(check_hash(df_hash,num_folds,stage=e))
            if not (check_hash(df_hash,num_folds,stage=e, drop_feat_idx=drop_feat_idx)):
                print('Problem at Generation of data!')
                print("Stage: "+e)
                return

        print('Generation of data successfully done!')
        savehash("data", hashcode=df_hash)
        savehash("folds", hashcode=str(num_folds))

        return load_data(emb_type=emb_type, collapse_classes=collapse_classes, fold_test=fold_test, num_folds=num_folds, random_state=random_state, drop_feat_idx=drop_feat_idx, only_claims=only_claims)

    else:
        print("Reading already processed data")
        #returns the selected emb type (bert/w2v)
        test_data = read_p(data_dir+"/folds/"+str(fold_test)+"/"+emb_type)
        test_target = read_p(data_dir+"/folds/"+str(fold_test)+"/labels")

        dev_data = read_p(data_dir+"/folds/"+str(fold_dev)+"/"+emb_type)
        dev_target = read_p(data_dir+"/folds/"+str(fold_dev)+"/labels")

        train_data_filenames = [data_dir+"/folds/"+str(i)+"/"+emb_type for i in range(num_folds) if i not in [fold_test,fold_dev]]
        train_data = np.concatenate([read_p(fn) for fn in train_data_filenames], axis=0)
        train_target_filenames = [data_dir+"/folds/"+str(i)+"/labels" for i in range(num_folds) if i not in [fold_test,fold_dev]]
        train_target = np.concatenate([read_p(fn) for fn in train_target_filenames], axis=0)

        return train_data, train_target, dev_data, dev_target, test_data, test_target, label_to_oh

