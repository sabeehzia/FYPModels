
# this code is used for training of 1D Conv ECG model

# Note: Since this file uses dataset,
# to run this file you need dataset folder in current directory, available at https://physionet.org/content/mitdb/1.0.0/

import pandas as pd 
import numpy as np 
import wfdb
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import  roc_auc_score, accuracy_score,  precision_score, recall_score
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout
from keras.utils import to_categorical
from keras.layers import Conv1D



data_path = 'mit-bih-arrhythmia-database-1.0.0/'

pts = [ '100','101','102','103','104','105','106','107',
        '108','109','111','112','113','114','115','116',
        '117','118','119','121','122','123','124','200',
        '201','202','203','205','207','208','209','210',
        '212','213','214','215','217','219','220','221',
        '222','223','228','230','231','232','233','234']

df = pd.DataFrame()

for pt in pts:
    file = data_path +pt
    annotation = wfdb.rdann(file,'atr')
    sym = annotation.symbol

    values, counts = np.unique(sym, return_counts=True)
    df_sub = pd.DataFrame({'sym':values,'val':counts,'pt':[pt]*len(counts)})
    df = pd.concat([df,df_sub],axis =0)

# print(df.groupby('sym').val.sum().sort_values(ascending=False))

abnormal = ['L','R','V','/','A','f','F','g','a','E','J','e','S']
nonbeat = ['[','!',']','x','(',')','p','t','u','`','\'','^','|','~','+','s','T','*','D','=','"','@','Q','?']

df['cat'] = -1
df.loc[df.sym=='N','cat'] = 0
df.loc[df.sym.isin(abnormal),'cat'] = 1

# print(df.groupby('cat').val.sum())


# file = data_path+'100'
# annot = wfdb.rdann(file,'atr')
# sym=annot.symbol

# record = wfdb.rdrecord(file)
# psig = record.p_signal

# xx = []
# yy =[]
# for i in range(len(psig)):
#     xx.append(psig[i][0])
#     yy.append(psig[i][1])



# # print(sym)
# print(psig[len(psig)-2])
# print(len(psig))


def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file,'atr')
    p_signal = record.p_signal

    assert record.fs == 360,'sample freq not 360'

    atr_sym = annotation.symbol
    atr_sample = annotation.sample
    return p_signal, atr_sym, atr_sample

p_signal, atr_sym, atr_sample = load_ecg(file)

# print(atr_sym)
# print(len(atr_sample))

def make_dataset(pts, num_sec, fs):

    num_cols = 2*num_sec*fs
    X_all = np.zeros((1,num_cols))
    Y_all = np.zeros((1,1))
    sym_all = []
    max_rows= []

    for pt in pts:
        file = data_path + pt
        p_signal, atr_sym, atr_sample = load_ecg(file)
    
        p_signal = p_signal[:,0]
        
        df_ann = pd.DataFrame({'atr_sym':atr_sym,'atr_sample':atr_sample})
        df_ann = df_ann.loc[df_ann.atr_sym.isin(abnormal+['N'])]

        num_rows = len(df_ann)
        X = np.zeros((num_rows,num_cols))
        Y = np.zeros((num_rows,1))
        max_row = 0
        for atr_sample, atr_sym in zip(df_ann.atr_sample.values,df_ann.atr_sym.values):
            left = max([0,(atr_sample - num_sec*fs) ])
            right = min([len(p_signal),(atr_sample + num_sec*fs) ])
            x = p_signal[left: right]
            if len(x) == num_cols:
                X[max_row,:] = x
                Y[max_row,:] = int(atr_sym in abnormal)
                sym_all.append(atr_sym)
                max_row += 1
        X = X[:max_row,:]
        Y = Y[:max_row,:]
        max_rows.append(max_row)
        X_all = np.append(X_all,X,axis = 0)
        Y_all = np.append(Y_all,Y,axis = 0)
    # drop the first zero row
    X_all = X_all[1:,:]
    Y_all = Y_all[1:,:]
    # check sizes make sense
    assert np.sum(max_rows) == X_all.shape[0], 'number of rows messed up'
    assert Y_all.shape[0] == X_all.shape[0], 'number of rows messed up'
    assert Y_all.shape[0] == len(sym_all), 'number of rows messed up'
    return X_all, Y_all, sym_all

# x = [2, 4, 6]
# y = [1, 3, 5]

# X1 = [r for r in range(650000)]
# plt.plot(X1, xx)
# plt.show()

num_sec = 3
fs = 360

X_all, Y_all, sym_all = make_dataset(pts, num_sec, fs)

X_train, X_valid, Y_train, Y_valid =  train_test_split(X_all, Y_all, test_size = 0.33, random_state =42)

X_train_cnn = np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
X_valid_cnn = np.reshape(X_valid, (X_valid.shape[0],X_valid.shape[1],1))

model = Sequential()
model.add(Conv1D(filters= 128, kernel_size = 5, activation = 'relu', input_shape=(2160,1)))
model.add(Dropout(rate = 0.25))
model.add(Flatten())    
model.add(Dense(1,activation = 'sigmoid'))

model.compile(
                loss = 'binary_crossentropy',
                optimizer = 'adam',
                metrics = ['accuracy'])

model.fit(X_train_cnn,Y_train, batch_size = 32, epochs = 2, verbose =1)

def print_report(y_actual, y_predict, thresh):
    accuracy = accuracy_score(y_actual,(y_predict>thresh))
    print(accuracy)
    return accuracy

thresh = (sum(Y_train)/len(Y_train))[0]

Y_train_preds_dense = model.predict_proba(X_train_cnn,verbose = 1)
Y_valid_preds_dense = model.predict_proba(X_valid_cnn,verbose = 1)

model.save('model.h5')

print('Train')
print_report(Y_train, Y_train_preds_dense,thresh)
print("Test")
print_report(Y_valid, Y_valid_preds_dense,thresh)

