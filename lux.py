#!/home/lucas/Lux/envLux/bin/python3
import argparse
from distutils import util
#read these from the call:
parser = argparse.ArgumentParser(description='LUX main script.')
parser.add_argument('--train', default=True, type=lambda x: bool(util.strtobool(x)), dest='train_flag', help='boolean that defines if model should be trained or loaded from lux_best_model.')
parser.add_argument('--tune', default=False, type=lambda x: bool(util.strtobool(x)), dest='tune_flag', help='boolean that defines if model should be tuned with keras optimizer.')
parser.add_argument('--num_folds', type=int, default=9, help='number of folds data is split into, 1 fold for val, 1 for test, rest for trainig.')
parser.add_argument('--regenerate_features', default=None, choices=[None, 'all', 'emb', 'feat', 'just_reload'], dest='force_reload', help='defines if the data features (including document embeddings) should be re-generated (all), only the embeddings should be re-generated (emb), only the features should be re-generated (feat), if the individual features should be just reloaded (just_reload) to be used with --feat_list or None (default)')
parser.add_argument('--only_claims', default=False, type=lambda x: bool(util.strtobool(x)), help='boolean that defines if model should take only claims into account instead of whole documents.')
parser.add_argument('--input_features', default='bert', choices=['bert', 'only_bert'],  help='selection of features to be used in the model.')
parser.add_argument('--env', default='last', choices=['dev', 'deploy'],  help='selection of development(testing) and deployment(running) environments. Basically changes the dataset to be used.')
parser.add_argument('--feat_list', nargs='+', default=["inf","div","qua","aff","sbj","spe","pau","unc","pas"], help='argument that defines which features will be used by the model. Default is All.\nSyntax:--feat_list inf div qua foo bar')
parser.add_argument('--learning_rate', '--lr', default=0.0005, type=float, help='defines the learning_rate variable for the model.')
parser.add_argument('--dense_dim', default=256, type=int, help='the number of dimensions of the dense layer.')
parser.add_argument('--dropout', default=0.5, type=float, help='what is the percentage of nodes that will have their weights updated per training example.')
parser.add_argument('--batch_size', default=32, type=int, help='number of entries that each training step will consider at once.')
parser.add_argument('--num_epochs', default=200, type=int, help='number of epochs to train each model.')
parser.add_argument('--drop_feat_idx', nargs="+", default=[], type=int, help='list of idxs to be dropped in the data loading step.')
args = parser.parse_args()

import random
seed = 27966
#seed = 17382
random.seed(seed)
root = random.randint(0,10090000)
print("ROOT:", root)

import warnings
#warnings.filterwarnings("once")
import numpy as np
np.random.seed(root)
import tensorflow
tensorflow.compat.v1.enable_eager_execution()
#tensorflow.enable_eager_execution()
tensorflow.compat.v1.set_random_seed(root)
#tensorflow.set_random_seed(root)

import traceback
import resource
import itertools
from itertools import groupby, chain, combinations

import gc
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score
import os,sys
#for comparing dev/deploy datasets and copying the right one into dataset.csv
import filecmp, shutil
from data_loader import load_data

from tensorflow.keras import losses, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Flatten, Bidirectional, TimeDistributed, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.initializers import RandomNormal, RandomUniform
import keras_tuner as kt

#from keras import backend as K
#print(K.tensorflow_backend._get_available_gpus())
#exit(1)

filename = sys.argv[0]
cwd = os.path.abspath(filename+"/..")
checkpoint_filepath = cwd+'/lux_best_models/'

DATA_SHAPE = None

#check which dataset will be used (development or deploy)
d_dir = cwd+"/data/datasets"
dev_data = d_dir+"/dataset_test02.csv"
run_data = d_dir+"/dataset_bck.csv"
cur_data = d_dir+"/dataset.csv"
if args.env != "last":
    if args.env == "dev" and not filecmp.cmp(dev_data, cur_data, shallow=True):
        shutil.copy2(dev_data, cur_data)
    if args.env == "deploy" and not filecmp.cmp(run_data, cur_data, shallow=True):
        shutil.copy2(run_data, cur_data)

def build_model(hp):
    hp_units = hp.Int('units', min_value=128, max_value=512, step=32)
    hp_dropout = hp.Float('dropout', min_value=0.3, max_value=0.8, step=0.1, default=0.5)
    hp_lr = hp.Choice('learning_rate', values=[1e-3, 5e-4])

    initializer = RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
    initializer2 = RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

    layer1 = Dense(hp_units,
            activation='relu',
            input_shape=(DATA_SHAPE[1:]),
            #kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
            #bias_regularizer=regularizers.l2(1e-4),
            #activity_regularizer=regularizers.l2(1e-5),
            kernel_initializer = initializer)

    batch_norm = BatchNormalization(axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None, gamma_constraint=None)

    model = Sequential()
    model.add(layer1)
    model.add(batch_norm)
    model.add(Dropout(hp_dropout))
    model.add(Dense(target_len, activation='softmax', kernel_initializer=initializer2))
    model.summary()

    adam = Adam(lr=hp_lr)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

    return model

def linear_model(target_len, learning_rate, DENSE_DIM, DROPOUT):
    #one suggestion is to determine the size the layers same as the input, instead of hard-coded
    initializer1 = RandomUniform(minval=-0.05, maxval=0.05, seed=seed)
    initializer2 = RandomUniform(minval=-0.05, maxval=0.05, seed=seed)

    layer1 = Dense(DENSE_DIM,
        activation='relu',
        input_shape=(DATA_SHAPE[1:]))
       # kernel_regularizer=regularizers.l1_l2(l1=1e-5, l2=1e-4),
       # bias_regularizer=regularizers.l2(1e-4),
       # activity_regularizer=regularizers.l2(1e-5),
       # kernel_initializer = initializer1)

    batch_norm = BatchNormalization(axis=-1,
            momentum=0.99,
            epsilon=0.001,
            center=True,
            scale=True,
            beta_initializer="zeros",
            gamma_initializer="ones",
            moving_mean_initializer="zeros",
            moving_variance_initializer="ones",
            beta_regularizer=None,
            gamma_regularizer=None,
            beta_constraint=None, gamma_constraint=None)

    model = Sequential()
    model.add(layer1)
    #model.add(batch_norm)
    model.add(Dropout(DROPOUT))
    model.add(Dense(target_len, activation='softmax'))
    #kernel_initializer=initializer2))
    model.summary()

    adam = Adam(lr=learning_rate)
    model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['binary_accuracy'])

    return model

def BILSTM_model(target_len, learning_rate, lstm_dim):
    #one suggestion is to determine the size the layers same as the input, instead of hard-coded
    model = Sequential()
    model.add(Bidirectional(LSTM(lstm_dim, return_sequences=True, dropout=0.3, recurrent_dropout=0.3), input_shape=(DATA_SHAPE[1:])))
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

res_acc, res_f1 = [],[]
try:
    for fold_test in range(args.num_folds):
        train, train_target, dev, dev_target, test, test_target, label_to_oh = load_data(emb_type=args.input_features, collapse_classes=False, fold_test=fold_test, num_folds=args.num_folds, random_state=root, force_reload=args.force_reload, drop_feat_idx=args.drop_feat_idx, only_claims=args.only_claims, feature_list=args.feat_list)
        args.force_reload = None
        DATA_SHAPE = (train.shape)
        target_len = len(label_to_oh)
        test_target = np.array([np.argmax(r) for r in test_target])

        if args.input_features in ['bert', 'only_bert']:
            model = linear_model(target_len, args.learning_rate, args.dense_dim, args.dropout)
        else:
            model = BILSTM_model(target_len, args.learning_rate, args.dense_dim)

        target = [values.tolist().index(max(values.tolist())) for values in train_target]
        t_count = {str(value): len(list(freq)) for value, freq in groupby(sorted(target))}
        sum_t = sum(t_count.values())
        inverse_weights = {0:int(t_count['1'])/sum_t, 1:int(t_count['0'])/sum_t}

        model_checkpoint = ModelCheckpoint(filepath=checkpoint_filepath+"best_model.h5", save_weights_only=True, monitor='val_loss', mode='min', save_best_only=True)
        early_stop = EarlyStopping(monitor="val_loss", min_delta=0, patience=10, verbose=0, mode="min", baseline=None, restore_best_weights=True)
        my_callbacks = [model_checkpoint, early_stop]

        if args.tune_flag:
            print("Tunning!")
            tuner = kt.BayesianOptimization(build_model,
                        objective="val_loss",
                        max_trials=100,
                        alpha=1e-4,
                        beta=50,
                        directory=cwd+"/Autotuner",
                        project_name="Lux",
                        overwrite=True)

            tuner.search(x=train, y=train_target,
                             validation_data=(dev, dev_target),
                             batch_size=args.batch_size,
                             callbacks=[early_stop],
                             epochs=200)

            bestHPs = tuner.get_best_hyperparameters(num_trials=3)[:3]
            input(len(bestHPs))
            for best_idx,bestHP in enumerate(bestHPs):
                model = tuner.hypermodel.build(bestHP)
                H = model.fit(x=train, y=train_target,
                                validation_data=(dev, dev_target), batch_size=args.batch_size,
                                    epochs=100, callbacks=[early_stop], verbose=1, use_multiprocessing=False)
                    # evaluate the network
                print("[INFO] evaluating network...")
                predictions = model.predict_classes(test, batch_size=args.batch_size)
                f1_best_tune = f1_score(test_target, predictions, average="macro")
                print(f1_best_tune)
                with open(os.getcwd()+"/results.txt", "a") as f:
                    string = ("Tune: "+str(seed)+" BestHP: "+str(bestHP.values)+" F1: "+str(f1_best_tune)+"\n")
                    f.write(string)
                input(best_idx)

        if args.train_flag:
            with tensorflow.device('/cpu:0'):
                history = model.fit(train, train_target, epochs=args.num_epochs, batch_size=args.batch_size, validation_data=(dev,dev_target), shuffle=False, callbacks=my_callbacks, use_multiprocessing=False)
                model_history = pd.DataFrame(history.history)
                model_history.plot(figsize=(8,5))
                plt.savefig("plots/model"+str(fold_test))
                print(model_history['val_loss'].min())

        else:
            best_model = checkpoint_filepath+"best_model.h5"
            model.load_weights(best_model)
            #print(dir(model))
            #print(model.get_config())
            #input()

        #makes predicitons for the test
        test_preds = model.predict_classes(test)
        print(test_preds)
        print("test:", test)
        #with open(cwd+"/log_test_folds.txt","a+") as f:
        #    f.write("fold"+str(fold_test)+"\n")

        #prints out the accuracy based on the right values and what the model predicted
        fold_acc = accuracy_score(test_target, test_preds)
        fold_f1 = f1_score(test_target, test_preds, average='macro')
        print("Test Accuracy on fold "+str(fold_test)+": ",fold_acc)
        print("Test F1 on fold "+str(fold_test)+": ",fold_f1)
        res_acc.append(fold_acc)
        res_f1.append(fold_f1)
        del train, train_target, dev, dev_target, test, test_target, label_to_oh, model
        gc.collect()

    avg_acc = sum(res_acc)/args.num_folds
    avg_f1 = sum(res_f1)/args.num_folds
    acc_var = np.var(res_acc)
    print("\n Averaged Test Accuracy over folds: ",avg_acc)
    print("\n Averaged Test Acc. Variance over folds: ",acc_var)
    print("\n Averaged Test F1 over folds: ",avg_f1)
    #salvar no log
    with open(os.getcwd()+"/results.txt", "a") as f:
        s = str(args.drop_feat_idx)+", "+str(args.num_epochs)+", "+str(args.input_features)+", "+str(args.learning_rate)+", "+str(args.dense_dim)+", "+str(args.dropout)+", "+str(args.batch_size)
        string = ("TrainShape:"+str(DATA_SHAPE)+" #EPOCH: ("+str(s)+") AVG: "+str(avg_acc)+" VAR: "+str(acc_var)+" F1: "+str(avg_f1)+" SEED: "+str(seed)+"\n")
        f.write(string)

except Exception as e:
    print(e)
    print(traceback.format_exc())
    sys.exit(1)
    with open(os.getcwd()+"/results.txt", "a") as f:
        string = (str(s)+": OOM."+str(type(e))+"\n")
        mem = "MEMORY: "+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)+"\n"
        f.write(string+mem)
