import scipy.io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import lag_plot
from statsmodels.graphics.tsaplots import plot_acf,plot_pacf
from statsmodels.tsa.ar_model import AR
from statsmodels.tsa.ar_model import AutoReg
from sklearn.metrics import accuracy_score, mean_squared_error
import seaborn as sns



---------------------------------------
-----------------------EDA-------------
#file1 = 'C:/Users/liyuyang/Desktop/BU/2022spring/679/mouse proj/OneDrive_1_4-2-2022/binned_behavior.mat'
#file2 = 'C:/Users/liyuyang/Desktop/BU/2022spring/679/mouse proj/OneDrive_1_4-2-2022/binned_behavior-2.mat'
#file3 = 'C:/Users/liyuyang/Desktop/BU/2022spring/679/mouse proj/OneDrive_1_4-2-2022/binned_zscore.mat'
#file4 = 'C:/Users/liyuyang/Desktop/BU/2022spring/679/mouse proj/OneDrive_1_4-2-2022/binned_zscore-2.mat'
file1 = 'zero_maze/binned_behavior.mat'
file2 = 'zero_maze/binned_behavior-2.mat'
file3 = 'zero_maze/binned_zscore.mat'
file4 = 'zero_maze/binned_zscore-2.mat'

data1 = scipy.io.loadmat(file1) # load matlab file
data2 = scipy.io.loadmat(file2)
data3 = scipy.io.loadmat(file3)
data4 = scipy.io.loadmat(file4)

binned_behavior = data1['binned_behavior']
binned_behavior_2 = data2['binned_behavior']
binned_zscore = data3['binned_zscore']
binned_zsocre_2 = data4['binned_zscore']

x = list(range(1,6301))
plt.scatter(x,binned_behavior[0,], color = 'blue',s = 0.01)
plt.scatter(x,binned_behavior[1,],color = 'red',s = 0.1)


plt.plot(binned_behavior_2)
plt.plot(binned_zscore[:,19])
plt.axhline(y= 1, ls='--', c='red') # 添加水平线
plt.show()

plt.plot(binned_zsocre_2)
plt.axhline(y= 1, ls='--', c='red') # 添加水平线
plt.show()

t = binned_behavior
f = binned_zscore
position = []
for i in range(0,6300):
    if t[0,i]==0 and t[1,i]==0:
        #t[0,i] = t[1,i] = 10
        position.append(i)

#index = t[t[1,:]==0].index
t = pd.DataFrame(t)
f = pd.DataFrame(f)
t.drop(position[:],axis = 1,inplace = True)
f.drop(position[:],axis = 0,inplace = True)

        
x = list(range(1,6301))
plt.scatter(x,t[0,], color = 'blue',s = 0.01)
plt.scatter(x,t[1,],color = 'red',s = 0.1)
plt.ylim(-1,2)
plt.show()
-------------------EDA end---------------



--------------------------------------------
--------------------4/14/2022 AR------------
x = pd.DataFrame(binned_zscore)
y = binned_behavior
lag_plot(z.iloc[:,19],lag = 11)
plt.show()

#fig, axes = plt.subplot(2,1)

fig,axes = plt.subplots(2,1)
plot_acf(z.iloc[:,19],ax=axes[0])
plot_pacf(z.iloc[:,19],ax=axes[1])

plt.tight_layout()
plt.show()

#train_size= int(0.7*x.shape[0])
#train_x,test_x = pd.DataFrame(x.iloc[0:train_size,:]).values,x.iloc[train_size+1:x.shape[0],:].values
#train_y,test_y = y[1,0:train_size],y[1,train_size+1:y.shape[1]]
from sklearn.model_selection import train_test_split
train_x,test_x = train_test_split(x,test_size=0.3,random_state=2022)
train_y,test_y = train_test_split(y[1,],test_size=0.3,random_state=2022)



lag = [1,2,3]
model_fit_ar = AutoReg(train,lags=lag).fit()
params = model_fit_ar.params
#p = model_fit_ar.k_ar

history = train[-max(lag):]
history = np.hstack(history).tolist()
test = np.hstack(test).tolist()

prediction = []
for t in range(len(test)):
    late = history[-max(lag):]
    yhat = params[0]
    for i in range(len(lag)):
        yhat += params[i+1] * late[max(lag)-lag[i]]
    prediction.append(yhat)
    history.append(test[t])

print(np.mean(np.array(test) - np.array(prediction))**2)
plt.plot(test,color = 'b')
plt.plot(prediction,color = 'r')
plt.show()
---------------AR end-----------------------



-----------------------------------------------------
-----------------logistic regression part------------

x = pd.DataFrame(binned_zscore)
y = binned_behavior
# split training and test set
from sklearn.model_selection import train_test_split
train_x,test_x = train_test_split(x,test_size=0.3,random_state=2022)
train_y,test_y = train_test_split(y[1,],test_size=0.3,random_state=2022)
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,confusion_matrix

# build model
model_logistic = LogisticRegression(solver='liblinear',
                                    random_state=0).fit(train_x,train_y)
model_logistic.predict(test_x)
confusion_matrix(test_y,model_logistic.predict(test_x))

#accurancy
accuracy_score(test_y,model_logistic.predict(test_x))
print(classification_report(test_y,model_logistic.predict(test_x)))
print("RMSE on test set =",mean_squared_error(test_y,model_logistic.predict(test_x)))

#confusion matrix
cm = confusion_matrix(test_y,model_logistic.predict(test_x))
sns.heatmap(cm,cmap="RdBu_r",annot=True,fmt="d")
plt.show()



----------------another data set for logistic model--------------
# fit model without time points losing observation
train_x,test_x = train_test_split(f,test_size=0.3,random_state=2022)
train_y,test_y = train_test_split(t.iloc[1,],test_size=0.3,random_state=2022)
train_y,test_y = train_test_split(t[1,],test_size=0.3,random_state=2022)

model_logistic = LogisticRegression(solver='liblinear',
                                    random_state=0).fit(train_x,train_y)
model_logistic.predict(test_x)
#confusion_matrix(test_y,model_logistic.predict(test_x))
#model_logistic.summary()
print(model_logistic)
print(model_logistic.intercept_, model_logistic.coef_)

accuracy_score(test_y,model_logistic.predict(test_x))
print(classification_report(test_y,model_logistic.predict(test_x)))
print("RMSE on test set =",mean_squared_error(test_y,model_logistic.predict(test_x)))
#confusion matrix
cm = confusion_matrix(test_y,model_logistic.predict(test_x))
sns.heatmap(cm,cmap="RdBu_r",annot=True,fmt="d")
plt.show()



----------------check----------------

from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(test_y, model_logistic.predict(test_x))
fpr, tpr, thresholds = roc_curve(test_y, model_logistic.predict_proba(test_x)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()









