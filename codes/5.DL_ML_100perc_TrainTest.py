#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  9 11:42:45 2023

@author: zeeshan
"""

            #  Train on 100% data for Independent Testing
            
            
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

data_path = '/home/zeeshan/ORI/datasets/training/bmark-AT.txt'

def CKSNAP(fastas, gap, **kw):
  
    kw = {'order': 'ACGT'}
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#', 'label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings

# ************ Data Processing ****************
import re, os, sys

def read_nucleotide_sequences(file):
    
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]

    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences




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

chemical_property = {
    'A': [1, 1, 1],
    'C': [0, 1, 0],
    'G': [1, 0, 0],
    'T': [0, 0, 1],
    'U': [0, 0, 1],
    '-': [0, 0, 0],
}

data_io = dataProcessing(data_path, "fasta") #path,fileformat

# ********** DATA LABELLING *********************

len_data = len(data_io)
pos_lab = np.ones(int(len_data/2));
neg_lab = np.zeros(int(len_data/2));
labels = np.concatenate((pos_lab,neg_lab),axis=0)

length = data_io.shape[0]
width = data_io.shape[1]

#################################

# ********* CAlculate Scores ********************

def calculateScore(X, y, model, folds):

    pred_y = model.predict(X)

    tempLabel = np.zeros(shape = y.shape, dtype=np.int32)

    for i in range(len(y)):
        if pred_y[i] < 0.5:
            tempLabel[i] = 0;
        else:
            tempLabel[i] = 1;
 
    confusion = confusion_matrix(y, tempLabel)
    TN, FP, FN, TP = confusion.ravel()


    # **** Confusion Matrix Plot ****

    confusion_norm = confusion / confusion.astype(np.float).sum(axis=1) # Normalize confusion matrix
    sn.heatmap(confusion_norm, annot=True, cmap='Blues')
 
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
    y_true = y
    y_pred = pred_y
    accuracy=(TN+TP)/(TN+FP+FN+TP)
    
    print('acc')
    print(accuracy)
    print('Sen')
    print(sensitivity)
    print('spe')
    print(specificity)
    
    y_true = tf.convert_to_tensor(y, np.float32)
    y_pred = tf.convert_to_tensor(pred_y, np.float32)
    
    plt.show() 
    
    lossValue = losses.binary_crossentropy(y_true, y_pred)#.eval()

    return {'acc2': accuracy2,'sn' : sensitivity, 'sp' : specificity, 'acc' : accuracy, 'MCC' : MCC, 'AUC' : ROCArea, 'precision' : precision, 'F1' : F1Score, 'fpr' : fpr, 'tpr' : tpr, 'thresholds' : thresholds, 'lossValue' : lossValue}

# ************************ RESULTS *****************************
# ******** Performance Calculation and ROC Curve ***************

def analyze(temp, OutputDir):

    testing_result = temp;

    file = open(OutputDir + '/performance.txt', 'w')

    index = 0
    for x in [testing_result]:


        title = ''

        if index == 0:
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

    for x in [testing_result]:

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



# ********** DATA LABELLING *********************



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
filters = 64
learning_rate = 0.0003346585347938224
weight_decay = 0.00005
dropoutMerge1 = 0.5
dropoutMerge2 = 0.1
dropoutMerge3 = 0.1

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08

optimization = Nadam(learning_rate=0.0003346585347938224)

from keras.layers import Add
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
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul

def CG(x):
    x1 = Conv1D(64, kernel_size=3, strides=3,
                padding='same', activation='relu',kernel_regularizer = regularizers.l2(1e-3),
                bias_regularizer = regularizers.l2(1e-3))(x)
    x1 = Dropout(0.4)(x1)

    x1 = Flatten()(x1)
    x1 = Dense(64, activation='relu')(x1)
    x2 = Bidirectional(GRU(64, return_sequences=True), merge_mode='mul')(x)
    x = Add()([x1, x2])
    x = Activation('relu')(x)
    x = Reshape((300,64))(x)
    return x

def model_cnn():
    
    inputs = Input(shape=(300, 4,))
    lstm_units = 64
    
    x1 = Conv1D(64, 1, activation='relu',padding="same")(inputs)
    x1=Dropout(0.2)(x1)

    x2 = Conv1D(96, 3, activation='relu',padding="same")(inputs) 
    x2=Dropout(0.2)(x2)
    
    x4 = Conv1D(32, 5, activation='relu',padding="same")(inputs) 
    x4=Dropout(0.2)(x4)
    
    x3 = Conv1D(16, 7, activation='relu',padding="same")(inputs)
    x3=Dropout(0.2)(x3)
    
    x5 = Conv1D(32, 9, activation='relu',padding="same")(inputs)
    x5=Dropout(0.2)(x5)
    
    x6 = Conv1D(96, 11, activation='relu',padding="same")(inputs)
    x6=Dropout(0.2)(x6)
    
    mergeX = concatenate([x1,x2,x4, x3, x5,x6]) #
    conv_out = Dropout(0.3)(mergeX)
    
    lstm_out = Bidirectional(GRU(96, return_sequences=True), merge_mode='mul')(conv_out)
    
    lstm_out3 = Bidirectional(GRU(16, return_sequences=True), merge_mode='mul')(conv_out)

    lstm_out = concatenate([lstm_out, lstm_out3]) 
    lstm_out = Dropout(0.3)(lstm_out) #0.35
    
    a1 = attention_3d_block(lstm_out)
    
    a1 = Flatten()(a1)

    a1 = Dense(32, activation='relu',bias_regularizer = regularizers.l2(1e-3))(a1) # Number of nodes in hidden layer
    
    a2 = Dense(16, activation='relu',bias_regularizer = regularizers.l2(1e-3))(a1)
    

    output = Dense(1, activation='sigmoid')(a2)
    model = Model(inputs=[inputs], outputs=[output,a2])
   
    model.compile(loss='binary_crossentropy', optimizer=optimization,metrics=['accuracy'])

    return model


##############
file=read_nucleotide_sequences(data_path) #ENAC encoded
cks = CKSNAP(file,gap=5) #gap=5 default
cc=np.array(cks)
data_only1 = cc[1:,2:]
data_only1 = data_only1.astype(np.float)



import pandas as pd

pcpsednc = pd.read_csv('/home/zeeshan/ORI/datasets/training/ES_PCPseDNC.csv', header=None)
pcpsednc = np.array(pcpsednc)
pcpsednc = pcpsednc[:,1:]


dcc = pd.read_csv('/home/zeeshan/ORI/datasets/training/ES_DCC.csv', header=None)
dcc = np.array(dcc)
dcc = dcc[:,1:]

# ************** K_Fold  **************************

folds=10  
kf = KFold(n_splits=folds, shuffle=True, random_state=4)

trainning_result = []
validation_result = []
testing_result = []
####################################################
# for test_index in range(folds):    
from mlxtend.classifier import StackingClassifier
import catboost as ctb

from sklearn import metrics
from xgboost import XGBClassifier
import shap
import pandas as pd

X_train = data_io
y_train = labels

model = model_cnn()
# model.summary()
callback = keras.callbacks.EarlyStopping(monitor='val_loss', patience= 15, restore_best_weights=True)

history = model.fit(X_train, y_train, batch_size = 128, epochs=100) 
                  
model.save('/home/zeeshan/ORI/Results/BN/model100per_ES.h5')

out,train_fea=model.predict(X_train)
training_fea=np.concatenate((train_fea,data_only1,pcpsednc,dcc),axis=-1)

training_fea=np.array(training_fea)
y_train=np.array(y_train)




xg = XGBClassifier() # 

xgb_clf = xg.fit(training_fea,y_train)
explainer = shap.TreeExplainer(xgb_clf) # , model_output = 'margin'
shap_values = explainer.shap_values(training_fea)
importance = np.mean(abs(shap_values),axis=0)
zero_elements_indx = [i for i, v in enumerate(importance) if v <= 0.00]
np.save('/home/zeeshan/ORI/Results/BN/zero_elements_indx_ES.npy',zero_elements_indx)


X_traintest = pd.DataFrame(training_fea)  
selected_X_train = X_traintest.drop(zero_elements_indx,axis=1)
selected_X_train = np.array(selected_X_train)

model_CBC = ctb.CatBoostClassifier(learning_rate=0.076275702658, max_depth=9, 
                                boosting_type='Plain', l2_leaf_reg=6, random_strength=1, random_state=14)

model_CBC.fit(selected_X_train, y_train)

predicted_y = model_CBC.predict(selected_X_train)

import pickle

filename = '/home/zeeshan/ORI/Results/BN/finalized_model_ES.sav'
pickle.dump(model_CBC, open(filename, 'wb'))

#%%

#                Independent Testing

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID";
os.environ["CUDA_VISIBLE_DEVICES"] = "5";

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
import pandas as pd

np.random.seed(4)
random.seed(4)

data_path = '/home/zeeshan/ORI/datasets/independent/Ind-ES.txt'

def CKSNAP(fastas, gap, **kw):
  
    kw = {'order': 'ACGT'}
    AA = kw['order'] if kw['order'] != None else 'ACGT'
    encodings = []
    aaPairs = []
    for aa1 in AA:
        for aa2 in AA:
            aaPairs.append(aa1 + aa2)

    header = ['#', 'label']
    for g in range(gap + 1):
        for aa in aaPairs:
            header.append(aa + '.gap' + str(g))
    encodings.append(header)

    for i in fastas:
        name, sequence, label = i[0], i[1], i[2]
        code = [name, label]
        for g in range(gap + 1):
            myDict = {}
            for pair in aaPairs:
                myDict[pair] = 0
            sum = 0
            for index1 in range(len(sequence)):
                index2 = index1 + g + 1
                if index1 < len(sequence) and index2 < len(sequence) and sequence[index1] in AA and sequence[
                    index2] in AA:
                    myDict[sequence[index1] + sequence[index2]] = myDict[sequence[index1] + sequence[index2]] + 1
                    sum = sum + 1
            for pair in aaPairs:
                code.append(myDict[pair] / sum)
        encodings.append(code)
    return encodings



# np.random.seed(seed=21)
# ************ Data Processing ****************
import re, os, sys

def read_nucleotide_sequences(file):
    
    if os.path.exists(file) == False:
        print('Error: file %s does not exist.' % file)
        sys.exit(1)
    with open(file) as f:
        records = f.read()
    if re.search('>', records) == None:
        print('Error: the input file %s seems not in FASTA format!' % file)
        sys.exit(1)
    records = records.split('>')[1:]

    fasta_sequences = []
    for fasta in records:
        array = fasta.split('\n')
        header, sequence = array[0].split()[0], re.sub('[^ACGTU-]', '-', ''.join(array[1:]).upper())
        header_array = header.split('|')
        name = header_array[0]
        label = header_array[1] if len(header_array) >= 2 else '0'
        label_train = header_array[2] if len(header_array) >= 3 else 'training'
        sequence = re.sub('U', 'T', sequence)
        fasta_sequences.append([name, sequence, label, label_train])
    return fasta_sequences




def dataProcessing(path,fileformat):
    all_seq_data = []

    for record in SeqIO.parse(path,fileformat):
        sequences = record.seq # All sequences in dataset
    
        # print(record.seq)
        # print(sequences)
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

data_io = dataProcessing(data_path, "fasta") #path,fileformat

# ********** DATA LABELLING *********************

len_data = len(data_io)
pos_lab = np.ones(int(len_data/2));
neg_lab = np.zeros(int(len_data/2));
labels = np.concatenate((pos_lab,neg_lab),axis=0)

length = data_io.shape[0]
width = data_io.shape[1]

file=read_nucleotide_sequences(data_path) #ENAC encoded
cks = CKSNAP(file,gap=5) #gap=5 default
cc=np.array(cks)
data_only1 = cc[1:,2:]
data_only1 = data_only1.astype(np.float)


pcpsednc = pd.read_csv('/home/zeeshan/ORI/datasets/independent/ES_PCPseDNC_ind.csv', header=None)
pcpsednc = np.array(pcpsednc)
pcpsednc = pcpsednc[:,1:]

dcc = pd.read_csv('/home/zeeshan/ORI/datasets/independent/ES_DCC_ind.csv', header=None)
dcc = np.array(dcc)
dcc = dcc[:,1:]


# ************** MODEL **************************

import keras.layers.core as core
from keras.regularizers import l2, l1
filters = 64
learning_rate = 0.0003346585347938224
weight_decay = 0.00005
dropoutMerge1 = 0.5
dropoutMerge2 = 0.1
dropoutMerge3 = 0.1

beta_1 = 0.9
beta_2 = 0.999
epsilon = 1e-08
optimization = Nadam(learning_rate=0.0003346585347938224)

from keras.layers import Add
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
    a_probs = Permute((2, 1))(a)
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul


def model_cnn():
    
    inputs = Input(shape=(300, 4,))
    lstm_units = 64
    
    
    x1 = Conv1D(64, 1, activation='relu',padding="same")(inputs)
    x1=Dropout(0.2)(x1)

    x2 = Conv1D(96, 3, activation='relu',padding="same")(inputs) 
    x2=Dropout(0.2)(x2)
    
    x4 = Conv1D(32, 5, activation='relu',padding="same")(inputs) 
    x4=Dropout(0.2)(x4)
    
    x3 = Conv1D(16, 7, activation='relu',padding="same")(inputs)
    x3=Dropout(0.2)(x3)
    
    x5 = Conv1D(32, 9, activation='relu',padding="same")(inputs)
    x5=Dropout(0.2)(x5)
    
    x6 = Conv1D(96, 11, activation='relu',padding="same")(inputs)
    x6=Dropout(0.2)(x6)
    
    mergeX = concatenate([x1,x2,x4, x3, x5,x6]) #
    conv_out = Dropout(0.3)(mergeX)
    

    lstm_out = Bidirectional(GRU(96, return_sequences=True), merge_mode='mul')(conv_out)
    
    lstm_out3 = Bidirectional(GRU(16, return_sequences=True), merge_mode='mul')(conv_out)

    lstm_out = concatenate([lstm_out, lstm_out3]) 
    lstm_out = Dropout(0.3)(lstm_out) #0.35
    
    a1 = attention_3d_block(lstm_out)
    
    a1 = Flatten()(a1)

    a1 = Dense(32, activation='relu',bias_regularizer = regularizers.l2(1e-3))(a1) # Number of nodes in hidden layer
    
    a2 = Dense(16, activation='relu',bias_regularizer = regularizers.l2(1e-3))(a1)
    

    output = Dense(1, activation='sigmoid')(a2)
    model = Model(inputs=[inputs], outputs=[output,a2])
  
    model.compile(loss='binary_crossentropy', optimizer=optimization,metrics=['accuracy'])

    return model

import pickle
model = model_cnn()
model.load_weights('/home/zeeshan/ORI/Results/BN/model100per_ES.h5')

out,train_fea=model.predict(data_io)
training_fea=np.concatenate((train_fea,data_only1,pcpsednc,dcc),axis=-1)

training_fea = pd.DataFrame(training_fea)
training_fea = training_fea.T.reset_index(drop=True).T
zero_elements_indx = np.load('/home/zeeshan/ORI/Results/BN/zero_elements_indx_ES.npy')
training_fea = training_fea.drop(zero_elements_indx,axis=1)

X_train=np.array(training_fea)
y_train=np.array(labels)

# # load the model from disk

filename = '/home/zeeshan/ORI/Results/BN/finalized_model_ES.sav'
loaded_model = pickle.load(open(filename, 'rb'))
result = loaded_model.score(X_train, labels)


result_pred = loaded_model.predict(X_train)
# print(result)
conf_matt = confusion_matrix(labels, result_pred)
TN, FP, FN, TP = conf_matt.ravel()

print('**** Results ****')
MCC = matthews_corrcoef(labels, result_pred)
print('MCC: ',MCC)

accuracy2 = (TP + TN) / float(TP+TN+FP+FN)
print('accuracy: ',accuracy2)

sensitivity = TP / float(TP + FN)
print('sensitivity: ',sensitivity)

specificity = TN / float(TN + FP)
print('specificity: ',specificity)

# ROCArea = roc_auc_score(labels, result_pred)
# print('AUC: ',ROCArea)
#AUC
y_pred_prob = loaded_model.predict_proba(X_train) #----
y_probs = y_pred_prob[:,1]
ROCArea = roc_auc_score(labels, y_probs)
print('AUC: ',ROCArea)

F1Score = (2 * TP) / float(2 * TP + FP + FN)
print('F1Score: ',F1Score)

precision = TP / float(TP + FP)
print('precision: ',precision)

recall = TP / float(TP + FN)
print('recall: ',recall)


