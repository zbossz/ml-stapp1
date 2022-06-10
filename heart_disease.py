import pickle

import  numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

heart_data = pd.read_csv('heart.csv')
print(heart_data.head())
print(heart_data.tail())
print(heart_data.shape)
print(heart_data.isnull().sum())

print(heart_data.describe())
print(heart_data['target'].value_counts())

X = heart_data.drop(columns='target',axis=1)
Y = heart_data['target']

print(X,Y)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)
print(X.shape,X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)

model = LogisticRegression()
model.fit(X_train,Y_train)

X_train_prediction = model.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction,Y_train)
print('测试集的精确度 :',training_data_accuracy)

X_test_prediction = model.predict(X_test)
testing_data_accuracy = accuracy_score(X_test_prediction,Y_test)
print('训练集的精确度 :',testing_data_accuracy)

input_Data = (63,1,3,145,233,1,0,150,0,2.3,0,0,1)
input_Data_pd = np.asarray(input_Data)
input_Data_reshaped = input_Data_pd.reshape(1,-1)
input_Data_pd_predict = model.predict(input_Data_reshaped)
print(input_Data_pd_predict)

if input_Data_pd_predict[0] == 0:
    print("数据人员没有心脏疾病")
else:
    print("数据人员有心脏疾病")
    
filename = 'my_heart_disease_model.sav'
pickle.dump(model,open(filename,'wb'))
