import os.path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.layers import Dense , Conv2D,Flatten,MaxPool2D , Dropout , BatchNormalization,MaxPooling2D,GlobalAvgPool2D
from keras.models import Sequential
from keras.preprocessing import image
import keras

import glob
import shutil
import math
import imutils
import cv2
from keras.callbacks import ModelCheckpoint,EarlyStopping
from keras.models import load_model

ROOT_DIR = "./Brain Tumor Data Set"
num_of_images = {}
for dir in os.listdir(ROOT_DIR):
    num_of_images[dir] = len(os.listdir(os.path.join(ROOT_DIR,dir)))

if not os.path.exists("Brain Tumor Data Set/train"):
    os.mkdir("Brain Tumor Data Set/train")
    for dir in os.listdir(ROOT_DIR):
        os.makedirs("Brain Tumor Data Set/train/"+dir)
        for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)),
                                    size=(math.floor(70/100*num_of_images[dir])-5),
                                    replace=False):
            O = os.path.join(ROOT_DIR,dir,img)
            D = os.path.join("Brain Tumor Data Set/train",dir)
            shutil.copy(O,D)
            os.remove(O)
else:
    print("训练文件夹已经存在了")

if not os.path.exists("Brain Tumor Data Set/val"):
    os.mkdir("Brain Tumor Data Set/val")
    for dir in os.listdir(ROOT_DIR):
        os.makedirs("Brain Tumor Data Set/val/"+dir)
        for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)),
                                    size=(math.floor(15/100*num_of_images[dir])-5),
                                    replace=False):
            O = os.path.join(ROOT_DIR,dir,img)
            D = os.path.join("Brain Tumor Data Set/val",dir)
            shutil.copy(O,D)
            os.remove(O)
else:
    print("评估文件夹已经存在了")

if not os.path.exists("Brain Tumor Data Set/test"):
    os.mkdir("Brain Tumor Data Set/test")
    for dir in os.listdir(ROOT_DIR):
        os.makedirs("Brain Tumor Data Set/test/"+dir)
        for img in np.random.choice(a = os.listdir(os.path.join(ROOT_DIR,dir)),
                                    size=(math.floor(15/100*num_of_images[dir])-5),
                                    replace=False):
            O = os.path.join(ROOT_DIR,dir,img)
            D = os.path.join("Brain Tumor Data Set/test",dir)
            shutil.copy(O,D)
            os.remove(O)
else:
    print("测试文件夹已经存在了")

model = Sequential()
model.add(Conv2D(filters=16,kernel_size=(3,3),activation='relu',input_shape=(224,224,3)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=32,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=64,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=128,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=256,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=512,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))
model.add(Conv2D(filters=1024,kernel_size=(3,3),activation='relu'))
model.add(MaxPool2D(pool_size=(2,2)))
model.add(Dropout(rate=0.5))

model.add(Flatten())
model.add(Dense(units=512,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=256,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=128,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=64,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=32,activation='relu'))
model.add(Dropout(rate=0.5))
model.add(Dense(units=1,activation='sigmoid'))

model.compile(loss = keras.losses.binary_crossentropy,optimizer='adam',metrics=['accuracy'])
print(model.summary())

train_datagen = image.ImageDataGenerator(
    zoom_range=0.2,shear_range=0.2,rescale=1./255,horizontal_flip=True
)
val_datagen = image.ImageDataGenerator(rescale=1./255)
test_datagen = image.ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(directory="Brain Tumor Data Set/train",
                                               target_size=(224,224),batch_size=64,
                                               class_mode='binary')
val_data = val_datagen.flow_from_directory(directory="Brain Tumor Data Set/val",
                                               target_size=(224,224),batch_size=64,
                                               class_mode='binary')
test_data = test_datagen.flow_from_directory(directory="Brain Tumor Data Set/test",
                                               target_size=(224,224),batch_size=64,
                                               class_mode='binary')



early_stopping = EarlyStopping(monitor='accuracy',min_delta=0.01,patience=5,verbose=1,mode='auto')
model_check_point = ModelCheckpoint(filepath='best_brain_model1.h5',monitor='accuracy',verbose=1,
                                    save_best_only=True,mode='auto')
call_back = [early_stopping,model_check_point]

hist = model.fit_generator(generator=train_data,
                           steps_per_epoch=16,
                           epochs=1024,verbose=1,
                           validation_data=val_data,
                           validation_steps=16,
                           callbacks=None)


# model = load_model("best_brain_model.h5")
# #
# # acc = model.evaluate_generator(generator=test_data)[1]
# # print(f"模型的精确度是 : {acc*100}%100")
#
# path = r"C:\Users\zzy\PycharmProjects\MultipleDiseasePre\Brain Tumor Data Set\Brain Tumor\Cancer (1).jpg"
# img = image.load_img(path,target_size=(224,224))
#
# i = image.img_to_array(img)/255
# print(i)
# input_arr = np.array([i])
# print(input_arr.shape)
# predict_x=model.predict(input_arr)
# classes_x=np.argmax(predict_x,axis=1)
#
# if classes_x == 0:
#     print("该图片数据为脑肿瘤患者")
# else:
#     print("该图片数据为健康状态的大脑")
