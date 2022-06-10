import numpy
import numpy as np
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

diabetes_dataset = pd.read_csv('diabetes.csv')
print(diabetes_dataset.head(5))
print(diabetes_dataset.shape)
print(diabetes_dataset.describe())
print(diabetes_dataset['Outcome'].value_counts())
print(diabetes_dataset.groupby('Outcome').mean())

X = diabetes_dataset.drop(columns='Outcome',axis=1)
Y = diabetes_dataset['Outcome']
print(X,Y)

scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
# standardized_data = scaler.fit_transform(X)
print(standardized_data)

X = standardized_data

print(X,Y)

X_train,X_test,Y_train,Y_test = \
    train_test_split(X,Y,test_size=0.2,stratify=Y,random_state=2)

print(X_train.shape,X_test.shape,Y_train.shape,Y_test.shape)

classifier = svm.SVC(kernel='linear')

classifier.fit(X_train,Y_train)

X_train_predict = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_predict,Y_train)
print('训练集模型准确度 : ',training_data_accuracy) # 79%

X_test_predict = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_predict,Y_test)
print('测试集模型准确度 : ',test_data_accuracy) # 77%

input_data = (6,148,72,35,0,33.6,0.627,50)

input_data_np = np.asarray(input_data)

input_data_reshape = input_data_np.reshape(1,-1)

input_data_reshape_normalize = scaler.transform(input_data_reshape)

prediction = classifier.predict(input_data_reshape_normalize)
print(prediction)

if prediction[0] == 1:
    print("数据人员有糖尿病")
if prediction[0] == 0:
    print("数据人员没有糖尿病")

filename = 'my_diabetes_model.sav'
pickle.dump(classifier,open(filename,'wb'))

loaded_model = pickle.load(open('my_diabetes_model.sav','rb'))











