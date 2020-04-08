

# this code utilizes the CNN model and give prediction on Abdullah Aziz ecg
# Note : it first plots ECG after crossing the matplotlib graph sigmoid probabilities prediction is shown on terminal
# closer to zero means normal 

import matplotlib.pyplot as plt
import numpy as np
from keras.models import load_model


file = open('abdecgnew.txt')
arr = []

for f in file:
    f = f.strip()
    if f== '':
        continue
    arr.append(f)


arr = [int(a) for a in arr]


app_len=0
if len(arr)<2160:
    app_len = 2160 - len(arr) 
    for i in range(app_len):
        arr.append(arr[i])

elif len(arr)>2160:
    arr2 = [arr[a] for a in range(0,2160)]
    arr = arr2

# tune more
arr = [(a-2400)/2600 for a in arr]

x =  [i for i in range(len(arr))]

y = arr
plt.plot(x, y)
plt.show()

# ---------- prediction------------


model = load_model('models/modelcnn.h5')

arr = np.array(arr)
arr = np.reshape(arr,(1,2160,1))

Y_abdullah_ECG = model.predict_proba(arr,verbose = 1)

print(Y_abdullah_ECG)

