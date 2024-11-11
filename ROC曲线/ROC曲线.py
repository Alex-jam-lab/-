import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv('telecom_churn.csv')


#获取x和y
X = data[['international plan','voice mail plan','number vmail messages','total day minutes','total day calls','total day charge','total eve minutes',
      'total eve calls','total eve charge','total night minutes','total night calls','total night charge','total intl minutes','total intl calls'
    ,'total intl charge','customer service calls']]

y = data[['churn']]

#print(X,y)

X['international plan'] = np.where(X['international plan'] == 'Yes', 1, 0)
X['voice mail plan'] = np.where(X['voice mail plan'] == 'Yes', 1, 0)
y['churn'] = np.where(y['churn'] == True, 1, 0)
y = y.to_numpy()

min_max_scaler = MinMaxScaler()
X = min_max_scaler.fit_transform(X.to_numpy())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

#定义模型DTC和RFC
dtc = DecisionTreeClassifier()
rfc = RandomForestClassifier()

dtc.fit(X_train, y_train.ravel())
rfc.fit(X_train, y_train.ravel())
print('DTC:',dtc.score(X_test, y_test.ravel()), 'RFC:',rfc.score(X_test, y_test.ravel()))

#评估模型
#ROC曲线

from matplotlib import pyplot as plt
from sklearn import metrics

fpr_test,tpr_test ,th_test = metrics.roc_curve(y_test.ravel(),dtc.predict(X_test))
fpr_train , tpr_train , th_train = metrics.roc_curve(y_train.ravel(),dtc.predict(X_train))

rfc_fpr_test,rfc_tpr_test ,rfc_th_test = metrics.roc_curve(y_test.ravel(),rfc.predict(X_test))
rfc_fpr_train , rfc_tpr_train , rfc_th_train = metrics.roc_curve(y_train.ravel(),rfc.predict(X_train))
#画图

plt.figure(figsize=[6,6])
plt.plot(fpr_test,tpr_test,color='blue',label='fpr_test')
plt.plot(fpr_train,tpr_train,color='red',label='fpr_train')
plt.plot(rfc_fpr_test,rfc_tpr_test,color='green',label='rfc_fpr_test')
plt.plot(rfc_fpr_train,rfc_tpr_train,color='cyan',label='rfc_fpr_train')
plt.legend()
plt.title('ROC curve')
plt.show()
