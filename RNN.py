#-------------------------RNN trial--------------
import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import accuracy_score, mean_squared_error
#conda install scikit-learn 网址：https://stackoverflow.com/questions/46113732/modulenotfounderror-no-module-named-sklearn
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import LSTM,LSTMCell,SimpleRNN
from tensorflow.keras.layers import Dropout,Activation
from tensorflow.math import reduce_prod
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import  StandardScaler
from sklearn.decomposition import PCA
import random # set seed


#608034_409	608102_412	608102_414	608103_416	608103_417
#608103_418	616669_251	619539_256	619539_257	619539_258
#619541_274	619542_254	619542_255
file1 = 'zero_maze/619539_257binned_behavior.mat'
file3 = 'zero_maze/619539_257binned_zscore.mat'


data1 = scipy.io.loadmat(file1)
data3 = scipy.io.loadmat(file3)

binned_behavior = data1['binned_behavior']
binned_zscore = data3['binned_zscore']

t = binned_behavior
f = binned_zscore

position = []
for i in range(0,t.shape[1]):
    if t[0,i]==0 and t[1,i]==0:
        #t[0,i] = t[1,i] = 10
        position.append(i)

#index = t[t[1,:]==0].index
t = pd.DataFrame(t)
f = pd.DataFrame(f)
t.drop(position[:],axis = 1,inplace = True)
f.drop(position[:],axis = 0,inplace = True)



#training_data = f.values
# Normalization
x = StandardScaler().fit_transform(f.values)
#scaler = MinMaxScaler()
#x = scaler.fit_transform(f.values)
#y = t.values



'''PCA part, if unnecessary, delete it '''
# random.seed(2022)
# Apply PCA and renew input x
x_pca = PCA(n_components=20).fit(x)
x = x_pca.transform(x)
'''PCA part end'''
x_pca.explained_variance_ratio_

from matplotlib import pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.preprocessing import scale
from sklearn.decomposition import PCA
from psynlig import pca_explained_variance_pie
plt.style.use('seaborn-talk')
data_set = load_wine()
x_pca = PCA(n_components=20).fit(x)
x = x_pca.transform(x)
fig, axi = pca_explained_variance_pie(x_pca, cmap='Spectral')
axi.set_title('Explained variance by PCA')
plt.show()



random.seed(2022) 
y = t.T.values
x_training,x_test = x[0:int(0.7*x.shape[0]),:],x[int(0.7*x.shape[0])+1:x.shape[0],:]
y_training,y_test = y[0:int(0.7*y.shape[0])],y[int(0.7*y.shape[0])+1:y.shape[0]]
x_training_data,y_training_data = [],[]
x_test_data,y_test_data = [],[]

# reshape data frame for RNN
for i in range(3,len(x_training)):
    x_training_data.append(x_training[i-3:i,:])
    y_training_data.append(y_training[i,1])
for i in range(3,len(x_test)):
    x_test_data.append(x_test[i-3:i,:])
    y_test_data.append(y_test[i,1])



# Change to array
x_training_data = np.array(x_training_data)
y_training_data = np.array(y_training_data)
x_test_data = np.array(x_test_data)
y_test_data = np.array(y_test_data)



# add pca to rnn？




x_training_data_sub = []
x_test_data_sub = []
for sublist in x_training_data:
    list = []
    list = sublist.reshape(1,-1)
    x_training_data_sub.append(list)
for sublist in x_test_data:
    list = []
    list = sublist.reshape(1,-1)
    x_test_data_sub.append(list)
x_training_data_sub = np.array(x_training_data_sub)
x_test_data_sub = np.array(x_test_data_sub)




#x_training_data = x_training_data.reshape(-1,1)
# x_training_data = np.array(x_training)
# y_training_data = np.array(y_training[:,1])
# x_test_data = np.array(x_test)
# y_test_data = np.array(y_test[:,1])



#x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],x_training_data.shape[1],1))
# x_training_data = np.reshape(x_training_data, (x_training_data.shape[0],x_training_data.shape[1],1))
# x_test_data = np.reshape(x_test_data, (x_test_data.shape[0],x_test_data.shape[1],1))
random.seed(2022) 
y_training_data = np.reshape(y_training_data,(y_training_data.shape[0],1,1))
y_test_data = np.reshape(y_test_data, (y_test_data.shape[0],1,1))



random.seed(2022)
# Building my RNN
rnn = Sequential()

#'''It's a normal neural network layer'''
#rnn.add(Dense(units=32,activation='relu', input_shape=(x_training_data_sub.shape[1],x_training_data_sub.shape[2])))

#''''It's rnn layer'''
rnn.add(LSTM(units=x_training_data.shape[2],activation='tanh', input_shape=(x_training_data.shape[1],x_training_data.shape[2])))
#conda install numpy=1.19.2 网址：https://stackoverflow.com/questions/58479556/notimplementederror-cannot-convert-a-symbolic-tensor-2nd-target0-to-a-numpy

#rnn.add(LSTM(64, activation='tanh'))
rnn.add(Dense(units=32,activation='relu'))
rnn.add(Dense(units=16,activation='relu'))
rnn.add(Dropout(0.1))
rnn.add(Dense(1, activation='sigmoid'))



random.seed(2022) 
rnn.compile(optimizer='adam',loss='mean_squared_error')
rnn.fit(x_training_data,y_training_data,epochs = 300,batch_size=32)

predictions = rnn.predict(x_test_data)>0.5
predictions = np.reshape(predictions,(predictions.shape[0],1))

# confusion matrix
y_test_data = np.reshape(y_test_data, (y_test_data.shape[0],1))
y_training_data = np.reshape(y_training_data, (y_training_data.shape[0],1))

cm = confusion_matrix(y_test_data,predictions)
accuracy_score(y_test_data,predictions)
sns.heatmap(cm,cmap="RdBu_r",annot=True,fmt="d")
plt.show(block=True)




