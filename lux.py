import traceback
import resource
import os
import itertools
import gc
import seaborn as sns
import matplotlib.pyplot as plt
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
import numpy as np
import sys
from data_loader import load_data
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import TimeDistributed
from tensorflow.keras import losses
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.regularizers import l2

random.seed(1)
root = random.randint(0,10090000)

#read these from the call:
train_flag = True
num_folds = 9

model = None
model_name = "lux_best_model.pkl"
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
    model.add(Dense(DENSE_DIM, activation='relu', input_shape=(data_shape[1:])))
    model.add(Dropout(0.3))
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

input_type = ['bert', 'only_bert', 'w2v', 'only_w2v']
input_type = ['bert']
learning_rate = [0.0001, 0.001, 0.01]
learning_rate = [0.001]
num_dims = [64,128]
epochs = [20,50,100,200]
setup = itertools.product(epochs,input_type,learning_rate,num_dims)
for s in setup:
    num_epochs = s[0]
    emb_type = s[1]
    learning_rate = s[2]
    num_dims = s[3]
    avg_acc = 0
    avg_f1 = 0
    model_name = "MODEL_e"+str(s[0])+"_"+str(s[1])+"_lr"+str(s[2])+"_d"+str(s[3])
    try:
        for fold in range(num_folds):
            train, train_target, dev, dev_target, test, test_target, label_to_oh = load_data(emb_type=emb_type, collapse_classes=True, fold=fold, num_folds=num_folds, random_state=root)
            data_shape = (train.shape)
            target_len = len(label_to_oh)
            test_target = np.array([np.argmax(r) for r in test_target])
            print(data_shape)

            if emb_type in ['bert', 'only_bert']:
                model = linear_model(data_shape, target_len, learning_rate, num_dims)
                #model = svm_model(data_shape, target_len, learning_rate, num_dims)
            else:
                model = BILSTM_model(data_shape, target_len, learning_rate, num_dims)

            #only trains if train is true
            chk = ModelCheckpoint(model_name, monitor='fmeasure', save_best_only=True, mode='max', verbose=1)
            if train_flag: model.fit(train, train_target, epochs=num_epochs, batch_size=batch_size, callbacks=[chk], class_weight={0: 0.7, 1: 0.3}, validation_data=(dev,dev_target))

            #makes predicitons for the test
            test_preds = model.predict_classes(test)
            print(test_preds)

            #prints out the accuracy based on the right values and what the model predicted
            fold_acc = accuracy_score(test_target, test_preds)
            fold_f1 = f1_score(test_target, test_preds, average='macro')
            print("Test Accuracy on fold "+str(fold)+": ",fold_acc)
            print("Test F1 on fold "+str(fold)+": ",fold_f1)
            avg_acc += fold_acc/num_folds
            avg_f1 += fold_f1/num_folds
            del train, train_target, dev, dev_target, test, test_target, label_to_oh
            gc.collect()

        print("\n Averaged Test Accuracy over folds: ",avg_acc)
        print("\n Averaged Test F1 over folds: ",avg_f1)
        #salvar no log
        with open(os.getcwd()+"/results.txt", "a") as f:
            string = (str(s)+" AVG: "+str(avg_acc)+" F1: "+str(avg_f1)+"\n")
            f.write(string)
    except Exception as e:
        print(e)
        print(traceback.format_exc())
        sys.exit(1)
        with open(os.getcwd()+"/results.txt", "a") as f:
            string = (str(s)+": OOM."+str(type(e))+"\n")
            mem = "MEMORY: "+str(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)+"\n"
            f.write(string+mem)
