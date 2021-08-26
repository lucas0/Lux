#!/home/lucas/Lux/envLux/bin/python3
import warnings
#warnings.filterwarnings("once")
from numpy.random import seed
seed(1)
from tensorflow import set_random_seed
set_random_seed(2)
import traceback
import resource
import os
import itertools
from itertools import groupby
from itertools import chain, combinations

import gc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import sys
import argparse
from distutils import util
from data_loader import load_data

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Bidirectional, TimeDistributed
from tensorflow.keras import losses, regularizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

random.seed(32442)
root = random.randint(0,10090000)
print("ROOT:", root)

#read these from the call:
parser = argparse.ArgumentParser(description='LUX main script.')
parser.add_argument('--train', default=True, type=lambda x: bool(util.strtobool(x)), dest='train_flag', help='boolean that defines if model should be trained or loaded from lux_best_model.')
parser.add_argument('--num_folds', action='store', type=int, default=9, help='number of folds data is split into, 1 fold for val, 1 for test, rest for trainig.')
parser.add_argument('--regenerate_features', default=False, type=lambda x: bool(util.strtobool(x)), dest='force_reload', help='boolean that defines if the data features (including document embeddings) should be re-generated.')
parser.add_argument('--only_claims', default=False, type=lambda x: bool(util.strtobool(x)), help='boolean that defines if model should take only claims into account instead of whole documents.')
args = parser.parse_args()

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")
checkpoint_filepath = cwd+'/lux_best_models/'

model = None
num_epochs = 5
LSTM_DIM = 256
DENSE_DIM = 256
learning_rate = 0.001
batch_size = 32

def svm_model(data_shape, target_len, learning_rate, DENSE_DIM):
    #one suggestion is to determine the size the layers same as the input, instead of hard-coded
    model = Sequential()
    model.add(Dense(DENSE_DIM, activation='relu', input_shape=(data_shape[1:])))
    model.add(Dense(target_len, kernel_regularizer=l2(0.01), activation='softmax'))
    model.summary()
    model.compile(loss='hinge', optimizer='adadelta', metrics=['binary_accuracy'])

    return model

def linear_model(data_shape, target_len, learning_rate, DENSE_DIM):
    #one suggestion is to determine the size the layers same as the input, instead of hard-coded
    model = Sequential()
    layer1 = Dense(DENSE_DIM,
            activation='relu',
            input_shape=(data_shape[1:]),
            #kernel_regularizer=regularizers.l2(1e-2),
            #bias_regularizer=regularizers.l2(1e-2),
            #activity_regularizer=regularizers.l2(1e-3),
            )
    model.add(layer1)
    model.add(Dropout(0.7))
    #model.add(Dense(DENSE_DIM/2, activation='relu'))
    #model.add(Dropout(0.5))
    model.add(Dense(target_len, activation='softmax'))
    model.summary()

    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

    return model

def LSTM_model(data_shape, target_len, learning_rate, LSTM_DIM):
    #one suggestion is to determine the size the layers same as the input, instead of hard-coded
    model = Sequential()
    model.add(LSTM(LSTM_DIM, input_shape=(data_shape[1:])))
    model.add(TimeDistributed(Dense(target_len, activation='softmax')))
    model.summary()

    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

    return model

def BILSTM_model(data_shape, target_len, learning_rate, LSTM_DIM):
    #one suggestion is to determine the size the layers same as the input, instead of hard-coded
    model = Sequential()
    model.add(Bidirectional(LSTM(LSTM_DIM, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), input_shape=(data_shape[1:])))
    model.add(Flatten())
    model.add(Dense(target_len, activation='softmax'))
    model.summary()

    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

    return model

def oh_to_label(l, d):
    for key in d.keys():
        if np.array_equal(l, d[key]):
            return key

input_type = ['only_bert']
learning_rate = [0.001]
num_dims = [64]
epochs = [100]

def drop_features(ran):
    initial_feat = list(range(ran))
    removed_feat = []
    remaining_feat = [[x] for x in initial_feat if x not in removed_feat]
    for x in remaining_feat:
        x.extend(removed_feat)
        x.sort()
        x = set(x)
        if len(x) is not len(removed_feat)+1:
            del(x)

    remaining_feat.append(removed_feat)

    return sorted(remaining_feat)

#drop_features_idx = drop_features(101)
#drop_features_idx = [[17, 23, 81, 20, 69, 8, 11, 3, 89]]
drop_features_idx = [[]]

#for i in list(range(79)):
#    drop_features_idx.remove([i])
#drop_features_idx.remove([])

setup = itertools.product(drop_features_idx, epochs,input_type,learning_rate,num_dims)

for s in setup:
    drop_feat_idx = s[0]
    num_epochs = s[1]
    emb_type = s[2]
    learning_rate = s[3]
    num_dims = s[4]
    res_acc, res_f1 = [],[]
    model_name = "MODEL_e"+str(s[0])+"_"+str(s[1])+"_lr"+str(s[2])+"_d"+str(s[3])
    try:
        for fold in range(args.num_folds):
            train, train_target, dev, dev_target, test, test_target, label_to_oh = load_data(emb_type=emb_type, collapse_classes=False, fold=fold, num_folds=args.num_folds, random_state=root, force_reload=args.force_reload, drop_feat_idx=drop_feat_idx, only_claims=args.only_claims)
            args.force_reload = False
            data_shape = (train.shape)
            target_len = len(label_to_oh)
            test_target = np.array([np.argmax(r) for r in test_target])

            if emb_type in ['bert', 'only_bert']:
                model = linear_model(data_shape, target_len, learning_rate, num_dims)
                #model = svm_model(data_shape, target_len, learning_rate, num_dims)

            else:
                model = BILSTM_model(data_shape, target_len, learning_rate, num_dims)

            target = [values.tolist().index(max(values.tolist())) for values in train_target]
            t_count = {str(value): len(list(freq)) for value, freq in groupby(sorted(target))}
            sum_t = sum(t_count.values())
            inverse_weights = {0:int(t_count['1'])/sum_t, 1:int(t_count['0'])/sum_t}
            model_checkpoint_callback = ModelCheckpoint(
                                        filepath=checkpoint_filepath+"model.{epoch:02d}-{val_loss:.2f}'",
                                        save_weights_only=False,
                                        monitor='val_loss',
                                        mode='min',
                                        save_best_only=True)
            if args.train_flag:
                history = model.fit(train, train_target, epochs=num_epochs, batch_size=batch_size, class_weight=inverse_weights, validation_data=(dev,dev_target), shuffle=False, callbacks=[model_checkpoint_callback])
            else:
                best_model = checkpoint_filepath+"model.67-0.46.h5"
                model = load_model(best_model)

            #makes predicitons for the test
            test_preds = model.predict_classes(test)
            print(test_preds)

            fig = pd.DataFrame(history.history).plot(figsize=(8,5))
            plt.savefig("plots/model"+str(fold))

            #prints out the accuracy based on the right values and what the model predicted
            fold_acc = accuracy_score(test_target, test_preds)
            fold_f1 = f1_score(test_target, test_preds, average='macro')
            print("Test Accuracy on fold "+str(fold)+": ",fold_acc)
            print("Test F1 on fold "+str(fold)+": ",fold_f1)
            res_acc.append(fold_acc)
            res_f1.append(fold_f1)
            del train, train_target, dev, dev_target, test, test_target, label_to_oh
            gc.collect()

        avg_acc = sum(res_acc)/args.num_folds
        avg_f1 = sum(res_f1)/args.num_folds
        acc_var = np.var(res_acc)
        print("\n Averaged Test Accuracy over folds: ",avg_acc)
        print("\n Averaged Test Acc. Variance over folds: ",acc_var)
        print("\n Averaged Test F1 over folds: ",avg_f1)
        #salvar no log
        with open(os.getcwd()+"/results.txt", "a") as f:
            string = ("TrainShape:"+str(data_shape)+" #EPOCH: "+str(s)+" AVG: "+str(avg_acc)+" VAR: "+str(acc_var)+" F1: "+str(avg_f1)+"\n")
            f.write(string)

    except Exception as e:
        print(e)
        print(traceback.format_exc())
        sys.exit(1)
        with open(os.getcwd()+"/results.txt", "a") as f:
            string = (str(s)+": OOM."+str(type(e))+"\n")
            mem = "MEMORY: "+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)+"\n"
            f.write(string+mem)
