import numpy as np 
import pandas as pd 
import os

!pip install opendatasets
import opendatasets as od
od.download("https://www.kaggle.com/datasets/ahemateja19bec1025/traffic-sign-dataset-classification")

labels=pd.read_csv("/content/traffic-sign-dataset-classification/labels.csv")
labels.head(45)

dic={}
for i in range(labels.shape[0]):
    dic[i]=labels.iloc[i]['Name']

dic

from PIL import Image
im = Image.open('/content/traffic-sign-dataset-classification/traffic_Data/DATA/0/000_0002.png')
im.size
# 

from keras.preprocessing import image
x=[]
y=[]
for dirs in os.listdir('/content/traffic-sign-dataset-classification/traffic_Data/DATA'):
    for files in os.listdir("/content/traffic-sign-dataset-classification/traffic_Data/DATA/"+dirs):
        sign_img = np.array(image.load_img("/content/traffic-sign-dataset-classification/traffic_Data/DATA/"+dirs+"/"+files,target_size = (32,32)))
        x.append(sign_img)
        y.append(int(dirs))

from sklearn.utils import shuffle
x_shuffled, y_shuffled = shuffle(x, y)

import matplotlib.pyplot as plt
def plot_image(x,y, index):
    image = plt.imshow(x[index])
    l=plt.title(dic[y[index]])
    ax.grid(False)
    ax.axis('off')    
    return image, l
fig = plt.figure(figsize=(20, 20))

for i in range(20):
    ax = fig.add_subplot(5, 4, i + 1)
    plot_image(x_shuffled,y_shuffled, i)

plt.show()

x_shuffled[0].shape

len(y_shuffled)

import keras
y_categorical = keras.utils.np_utils.to_categorical(y_shuffled, 58)

from sklearn.model_selection import train_test_split
x_train,x_rest,y_train,y_rest=train_test_split(x_shuffled,y_categorical,test_size=0.2)

x_test,x_val,y_test,y_val=train_test_split(x_rest,y_rest,test_size=0.5)

print(len(x_train),len(y_train))
len(x_test),len(y_test)

from tensorflow.keras.applications.resnet import ResNet50
from keras import backend as k
from keras.preprocessing.image import ImageDataGenerator,load_img, img_to_array
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D,GlobalAveragePooling2D
from keras.layers import Activation, Dropout, BatchNormalization, Flatten, Dense, AvgPool2D,MaxPool2D
from keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
import keras
import tensorflow as tf

x_train_array=np.array(x_train)
y_train_array=np.array(y_train)
x_train_scaled=x_train_array/255.0

x_val_array=np.array(x_val)
y_val_array=np.array(y_val)
x_val_scaled=x_val_array/1.0
x_val_scaled=x_val_scaled/255

x_test_array=np.array(x_test)
x_test_scaled=x_test_array/255.0

y_test_array=np.array(y_test)

resnet_model = ResNet50(weights= 'imagenet', include_top=False, input_shape= (32,32,3))

resnet_model.summary()

x = resnet_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
predictions = Dense(58, activation= 'softmax')(x)
model2 = Model(inputs = resnet_model.input, outputs = predictions)

model2.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

from keras.callbacks import EarlyStopping,ModelCheckpoint
stop = EarlyStopping(
    monitor='val_accuracy', 
    mode='max',
    patience=3
)

checkpoint= ModelCheckpoint(
    filepath='./',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True)


history =  model2.fit(x_train_scaled, y_train_array,validation_data = (x_val_scaled, y_val_array), batch_size =256, epochs =10, verbose = 1, callbacks = [stop, checkpoint])

plt.figure(figsize=(12, 5))

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()
plt.figure(figsize=(12, 5))

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()

import seaborn as sns
q = len(list(history.history['loss']))
plt.figure(figsize=(12, 6))
sns.lineplot(x = range(1, 1+q), y = history.history['accuracy'], label = 'Accuracy')
sns.lineplot(x = range(1, 1+q), y = history.history['loss'], label = 'Loss')
plt.xlabel('#epochs')
plt.ylabel('Training')
plt.legend()''
