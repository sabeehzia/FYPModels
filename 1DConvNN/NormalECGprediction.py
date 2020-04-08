
# this code use CNN model to predict normal ecg from dataset

# Note: Since this file uses dataset,
#  to run this file you need dataset folder in current directory, available at https://physionet.org/content/mitdb/1.0.0/

# Note : it first plots ECG after crossing the matplotlib graph sigmoid probabilities prediction is shown on terminal
# closer to zero means normal


import matplotlib.pyplot as plt
import numpy as np
import wfdb
from keras.models import load_model


data_path = 'mit-bih-arrhythmia-database-1.0.0/'

def load_ecg(file):
    record = wfdb.rdrecord(file)
    annotation = wfdb.rdann(file,'atr')
    p_signal = record.p_signal

    assert record.fs == 360,'sample freq not 360'

    atr_sym = annotation.symbol
    atr_sample = annotation.sample
    return p_signal, atr_sym, atr_sample

p_signal, atr_sym, atr_sample = load_ecg(data_path+"100")

p_signal = p_signal[:,0]

arr = p_signal
print(len(arr))
print(len(atr_sym))


app_len=0
if len(arr)<2160:
    app_len = 2160 - len(arr) 
    for i in range(app_len):
        arr.append(arr[i])

# +600 for normal
elif len(arr)>2160:
    arr2 = [arr[a+600] for a in range(0,2160)]
    arr = arr2

x =  [i for i in range(len(arr))]
y = arr
plt.plot(x, y)
plt.show()


#  ----------prediction --------

model = load_model('models/modelcnn.h5')

arr = np.array(arr)
arr = np.reshape(arr,(1,2160,1))

Y_Normal_Dataset = model.predict_proba(arr,verbose = 1)

print(Y_Normal_Dataset)



