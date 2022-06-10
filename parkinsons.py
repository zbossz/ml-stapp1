import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import svm
from sklearn.metrics import accuracy_score

parkinsons_Data = pd.read_csv('parkinsons.csv')
print(parkinsons_Data.head())
print(parkinsons_Data.tail())

print(parkinsons_Data.shape)
print(parkinsons_Data.info())
print(parkinsons_Data.isnull().sum())
print(parkinsons_Data.describe())
print(parkinsons_Data['status'].value_counts())
print(parkinsons_Data.groupby('status').mean())

X = parkinsons_Data.drop(columns=['name','status'],axis=1)
Y = parkinsons_Data['status']
print(X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X.shape,'\n',X_train.shape,X_test.shape)
print(Y_train.value_counts(),Y_test.value_counts())

scaler = StandardScaler()
scaler.fit(X_train)
X_train_normalized = scaler.transform(X_train)
X_test_normalized = scaler.transform(X_test)
print(X_train_normalized)
print(X_train_normalized)

model = svm.SVC(kernel='linear')
model.fit(X_train_normalized,Y_train)

X_train_prediction = model.predict(X_train_normalized)
training_Data_accuracy = accuracy_score(X_train_prediction,Y_train)

X_test_prediction = model.predict(X_test_normalized)
testing_Data_accuracy = accuracy_score(X_test_prediction,Y_test)

print('训练数据的准确度 ： ',training_Data_accuracy)
print('测试数据的准确度 ： ',testing_Data_accuracy)

input_Data = (119.99200,157.30200,74.99700,0.00784,0.00007,0.00370,0.00554,0.01109,0.04374,0.42600,0.02182,0.03130,0.02971,0.06545,0.02211,21.03300,0.414783,0.815285,-4.813031,0.266482,2.301442,0.284654)

input_Data_np = np.asarray(input_Data)
input_Data_np_reshaped = input_Data_np.reshape(1,-1)
input_Data_np_reshaped_normalized = scaler.transform(input_Data_np_reshaped)
prediction = model.predict(input_Data_np_reshaped_normalized)

if prediction[0] == 1:
    print("数据人员有帕金森")
else:
    print("数据人员没有帕金森")








