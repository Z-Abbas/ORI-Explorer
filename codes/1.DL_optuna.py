#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 19 13:40:17 2021

@author: zeeshan
"""
# *************** For Server *******************
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "4";

# ************ Libraries **********************
import numpy as np
from Bio import SeqIO
from sklearn.model_selection import KFold,StratifiedKFold
from sklearn.model_selection import train_test_split
from keras.models import Model,Sequential
from keras.layers import Input, Conv1D, Activation, Multiply, BatchNormalization, GRU, LSTM, merge, Bidirectional, TimeDistributed, Dropout, concatenate, MaxPooling1D, SpatialDropout1D, Flatten, Dense 
from keras.optimizers import SGD,Adam,Adamax, Nadam, RMSprop, Adadelta
from keras.metrics import binary_accuracy
import matplotlib.pyplot as plt
from keras import regularizers
from sklearn.metrics import confusion_matrix, recall_score, roc_curve, roc_auc_score, auc
import math
import tensorflow as tf
from keras import losses
import random
# import scikitplot as skplt
import seaborn as sn
# import keras
from sklearn.metrics import matthews_corrcoef
import keras
import keras.backend as K

np.random.seed(4)
random.seed(4)

# np.random.seed(seed=21)
# ************ Data Processing ****************

def dataProcessing(path,fileformat):
    all_seq_data = []

    for record in SeqIO.parse(path,fileformat):
        sequences = record.seq # All sequences in dataset
  
        seq_data=[]
       
        for i in range(len(sequences)):
            if sequences[i] == 'A':
                seq_data.append([1,0,0,0])
            if sequences[i] == 'T':
                seq_data.append([0,1,0,0])
            if sequences[i] == 'U':
                seq_data.append([0,1,0,0])                
            if sequences[i] == 'C':
                seq_data.append([0,0,1,0])
            if sequences[i] == 'G':
                seq_data.append([0,0,0,1])
            if sequences[i] == 'N':
                seq_data.append([0,0,0,0])
        all_seq_data.append(seq_data)    
        
    all_seq_data = np.array(all_seq_data);
    
    return all_seq_data

data_io = dataProcessing("/home/zeeshan/ORI/datasets/training/bmark-ES.txt", "fasta") #path,fileformat



# ********** DATA LABELLING *********************

len_data = len(data_io)
pos_lab = np.ones(int(len_data/2));
neg_lab = np.zeros(int(len_data/2));
labels = np.concatenate((pos_lab,neg_lab),axis=0)

length = data_io.shape[0]
width = data_io.shape[1]


# ********* CAlculate Scores ********************

def calculateScore(X, y, model, folds):
    
    score = model.evaluate(X, y) # Gives loss and accuracy
    pred_y = model.predict(X)

    accuracy = score[1];

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
    # print(tempLabel)        
    
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()
  
    # **** Confusion Matrix Plot ****

    confusion_norm = confusion / confusion.astype(np.float).sum(axis=1) # Normalize confusion matrix
    sn.heatmap(confusion_norm, annot=True, cmap='Blues')
    # sn.heatmap(confusion, annot=True, cmap='Blues')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()
        

    accuracy2 = (TP + TN) / float(TP+TN+FP+FN)
    sensitivity = TP / float(TP + FN)
    specificity = TN / float(TN + FP)
    MCC = matthews_corrcoef(y, tempLabel)
 
    F1Score = (2 * TP) / float(2 * TP + FP + FN)
    precision = TP / float(TP + FP)

    pred_y = pred_y.reshape((-1, ))

    ROCArea = roc_auc_score(y, pred_y)
    fpr, tpr, thresholds = roc_curve(y, pred_y)
    lossValue = None;

    print(y.shape)
    print(pred_y.shape)

    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    plt.show() 
    
    lossValue = losses.binary_crossentropy(y_true, y_pred)#.eval()

    return {'acc2': accuracy2,'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea, 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'lossValue' : lossValue}

# ************************ RESULTS *****************************
# ******** Performance Calculation and ROC Curve ***************

def analyze(temp, OutputDir):

    trainning_result, validation_result, testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [trainning_result, validation_result, testing_result]:


        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        index += 1;

        file.write(title +  'results\n')

        for j in ['acc2','sn', 'sp', 'acc', 'MCC', 'AUC', 'precision', 'F1', 'lossValue']:

            total = []

            for val in x:
                total.append(val[j])

            file.write(j + ' : mean : ' + str(np.mean(total)) + ' std : ' + str(np.std(total))  + '\n')

        file.write('\n\n______________________________\n')
    file.close();

# **** ROC Curve ****

    index = 0

    for x in [trainning_result, validation_result, testing_result]:

        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        i = 0

        for val in x:
            tpr = val['tpr']
            fpr = val['fpr']
            tprs.append(np.interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i+1, roc_auc))

            i += 1

        print;

        plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',label='Random', alpha=.8)

        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)
        plt.plot(mean_fpr, mean_tpr, color='b',
                 label=r'Mean ROC (AUC = %0.2f $\pm$ %0.2f)' % (mean_auc, std_auc),
                 lw=2, alpha=.8)

        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='grey', alpha=.2,
                         label=r'$\pm$ 1 std. dev.')

        plt.xlim([-0.05, 1.05])
        plt.ylim([-0.05, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic curve')
        plt.legend(loc="lower right")

        title = ''

        if index == 0:
            title = 'training_'
        if index == 1:
            title = 'validation_'
        if index == 2:
            title = 'testing_'

        plt.savefig( OutputDir + '/' + title +'ROC.png')
        plt.close('all');

        index += 1;


def mcc(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0, 1))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())

# ************** MODEL **************************

import keras.layers.core as core
from keras.regularizers import l2, l1

learning_rate = 0.002
weight_decay = 0.00005
dropoutMerge1 = 0.5
dropoutMerge2 = 0.1
dropoutMerge3 = 0.1

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08

    
from keras.layers.core import Permute, Reshape, Dense, Lambda, RepeatVector, Flatten
from keras import backend as K

SINGLE_ATTENTION_VECTOR = False

def attention_3d_block(inputs):
    input_dim = int(inputs.shape[2])
    a = Permute((2, 1))(inputs)
    a = Reshape((input_dim, 300))(a)
    a = Dense(300, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((2, 1), name='attention_vec')(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def model_cnn(params):
    
    inputs = Input(shape=(300, 4,))
    lstm_units = 64
    filters=64
    
    x1 = Conv1D(params['conv1'], 1, activation='relu',padding="same")(inputs)
    x1=Dropout(0.2)(x1)

    x2 = Conv1D(params['conv2'], 3, activation='relu',padding="same")(inputs) 
    x2=Dropout(0.2)(x2)
    
    x4 = Conv1D(params['conv3'], 5, activation='relu',padding="same")(inputs) 
    x4=Dropout(0.2)(x4)
    
    x3 = Conv1D(params['conv4'], 7, activation='relu',padding="same")(inputs)
    x3=Dropout(0.2)(x3)
    
    x5 = Conv1D(params['conv5'], 9, activation='relu',padding="same")(inputs)
    x5=Dropout(0.2)(x5)
    
    x6 = Conv1D(params['conv6'], 11, activation='relu',padding="same")(inputs)
    x6=Dropout(0.2)(x6)
    
    mergeX = concatenate([x1,x2,x4, x3, x5,x6]) #
    conv_out = Dropout(0.3)(mergeX)


    lstm_out = Bidirectional(GRU(params['units1'], return_sequences=True), merge_mode='mul')(conv_out)
    
    lstm_out3 = Bidirectional(GRU(params['units2'], return_sequences=True), merge_mode='mul')(conv_out)

    lstm_out = concatenate([lstm_out, lstm_out3]) 
    lstm_out = Dropout(0.3)(lstm_out) #0.35
    
    a1 = attention_3d_block(lstm_out)
    
    a1 = Flatten()(a1)

    a1 = Dense(params['dens1'], activation='relu',bias_regularizer = regularizers.l2(1e-3))(a1) # Number of nodes in hidden layer
    
    a2 = Dense(params['dens2'], activation='relu',bias_regularizer = regularizers.l2(1e-3))(a1)
    
    output = Dense(1, activation='sigmoid')(a2)
    model = Model(inputs=[inputs], outputs=output)
    optimization = Nadam(learning_rate=params['learning_rate'])

    model.compile(loss='binary_crossentropy', optimizer=optimization,metrics=['accuracy'])

    return model
    


# ************** Data Shuffle  ********************

c = list(zip(data_io, labels))
# random.shuffle(c)
random.Random(50).shuffle(c)
data_io, labels = zip(*c)
data_io=np.asarray(data_io)
labels=np.asarray(labels)

import optuna
# ************** K_Fold  **************************
def objective(trial):
    folds=10
    kf = KFold(n_splits=folds, shuffle=True, random_state=4)
    
    trainning_result = []
    validation_result = []
    testing_result = []
    
    # for test_index in range(folds):    
    for i, (train_index, test_index) in enumerate(kf.split(data_io,labels)): #New Try
            
            X_train0, X_test = data_io[train_index], data_io[test_index]
            y_train0, y_test = labels[train_index], labels[test_index]
        
            X_train, X_validation, y_train, y_validation = train_test_split(X_train0, y_train0, test_size=0.2, random_state=92, shuffle=True) #92
            
            params = {
               'learning_rate': trial.suggest_loguniform('learning_rate', 1e-5, 1e-1),
              
              'conv1': trial.suggest_categorical("conv1", [8,16,32,64,96,128]),
              'conv2': trial.suggest_categorical("conv2", [8,16,32,64,96,128]),
              'conv3': trial.suggest_categorical("conv3", [8,16,32,64,96,128]),
              'conv4': trial.suggest_categorical("conv4", [8,16,32,64,96,128]),
              'conv5': trial.suggest_categorical("conv5", [8,16,32,64,96,128]),
              'conv6': trial.suggest_categorical("conv6", [8,16,32,64,96,128]),
              'drop': trial.suggest_float("drop", 0, 0.5),
              'drop2': trial.suggest_float("drop2", 0, 0.5),
              'dense': trial.suggest_categorical("dense", [8,16,32,64]),
              'units1': trial.suggest_categorical("units1", [8,16,32,64,96,128]),
              'units2': trial.suggest_categorical("units2", [8,16,32,64,96,128]),
              'dens1': trial.suggest_categorical("dens1", [8,16,32,64,96,128]),
              'dens2': trial.suggest_categorical("dens2", [8,16,32,64,96,128]),

              }
           

            model = model_cnn(params);
            
            callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience= 15, restore_best_weights=True)
        
            history = model.fit(X_train, y_train, batch_size = 128, epochs=60, 
                                validation_data = (X_validation, y_validation),callbacks=[callback]);
           
            #**************** Plot graphs **************
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            
            #-----------------------------------
            X=X_test
            y=y_test
            score = model.evaluate(X, y) # Gives loss and accuracy
            pred_y = model.predict(X)
        
            accuracy = score[1];
        
            tempLabel = np.zeros(shape = y.shape, dtype=np.int32)
        
            for i in range(len(y)):
                if pred_y[i] < 0.5:
                    tempLabel[i] = 0;
                else:
                    tempLabel[i] = 1;
            # print(tempLabel)        
            
            confusion = confusion_matrix(y, tempLabel)
            TN, FP, FN, TP = confusion.ravel()

            accuracy = (TP + TN) / float(TP+TN+FP+FN)
            testing_result.append(accuracy)

            
            print('fold',i)
      
            i+=1
    return np.mean(testing_result)

def print_best_trial_so_far(study, trial):
    print('\nBest trial hyper-parameters:')
    for key, value in study.best_trial.params.items():
        print('{}: {}'.format(key, value))
    print('Value {}\n'.format(study.best_trial.value))            

from optuna.samplers import TPESampler
sampler = TPESampler(seed=10)
study = optuna.create_study(direction="maximize", sampler=sampler)

study.optimize(objective, n_trials=1000, timeout=1000000000,gc_after_trial=True,callbacks=[print_best_trial_so_far])

print("Number of finished trials: {}".format(len(study.trials)))

print("Best trial:")
trial = study.best_trial

print("  Value: {}".format(trial.value))

print("  Params: ")
for key, value in trial.params.items():
    print("    {}: {}".format(key, value))
    





