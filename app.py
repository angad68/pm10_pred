#!/usr/bin/env python
# coding: utf-8

# In[1]:


# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
# You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session


# In[2]:


import sys

from typing import Dict, Optional, Tuple
from pathlib import Path

import math

import tensorflow as tf
from tensorflow import keras

import seaborn as sns
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator

from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from tensorflow.keras import backend #Keras version 2.1.6
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten, Concatenate, LeakyReLU, Input, Conv2D, MaxPooling2D 

from tensorflow.keras import layers

from tensorflow.keras.optimizers import Adam, SGD
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.preprocessing import image
#from PIL import Image

from sklearn.metrics import r2_score
from sklearn.utils import shuffle

get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


df = pd.read_csv('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/IND_and_Nep_AQI_Dataset.csv')
df.head()


# In[4]:


df = shuffle(df)

df.sample(frac=1).reset_index(drop=True)

number_of_rows = 3000
sub_dfs = [df[i:i + number_of_rows] for i in range(0, df.shape[0], number_of_rows)]
for idx, sub_df in enumerate(sub_dfs):
    sub_df.to_csv(f'frag3000_{idx}.csv', index=False)


# In[5]:


df = pd.read_csv('../working/frag3000_1.csv')
df.head(15)


# In[6]:


def build_x(path):
    train_img = []
    for i in range(df.shape[0]):
        img = image.load_img(path + df['Filename'][i])
        img = image.img_to_array(img)
        img = tf.keras.applications.vgg16.preprocess_input(img)
        #img = img / 255        # with respect to imagenet, no scaling be used
        train_img.append(img)

    x = np.array(train_img)
    return x


# In[7]:


x_origin = build_x('/kaggle/input/air-pollution-image-dataset-from-india-and-nepal/Air Pollution Image Dataset/Air Pollution Image Dataset/Combined_Dataset/All_img/')


# In[8]:


x_origin.shape


# In[9]:


pm10 =pd.DataFrame(df['PM10'])
pm10


# In[11]:


x_origin_train, x_origin_temp, y_train, y_temp = train_test_split(
    x_origin, pm10, train_size=0.8, shuffle=True, random_state=42
)

x_origin_valid, x_origin_test, y_valid, y_test = train_test_split(
    x_origin_temp, y_temp, test_size=0.5, shuffle=True, random_state=42
)


# In[12]:


x_origin_train.shape


# In[13]:


y_train.shape


# In[14]:


x_origin = []
x_origin_temp = []
y_temp = []
y = []


# In[17]:


plt.imshow(x_origin_test[10]/255)


# In[18]:


y_test = y_test.reset_index(drop=True)
y_test


# In[19]:


y_test.head(15)


# **VGG16**

# In[20]:


pre_trained_model  = tf.keras.applications.VGG16(input_shape=(224, 224, 3), include_top=False, weights="imagenet")
for layer in pre_trained_model.layers:
    layer.trainable = False
    print(layer.name)


# In[21]:


x1 = Flatten()(pre_trained_model.output)
fc1 = Dense(512, activation = 'relu')(x1)
fc2 = Dense(512, activation = 'relu')(fc1)
x = Dense(1, activation='linear')(fc2)
model = Model(pre_trained_model.input, x)
    
opt = tf.keras.optimizers.Adam(lr=0.0001)
model.compile(loss='mse', optimizer=opt)
model.summary()


# In[22]:


weight_path="{}_aqi.best.hdf5".format('vgg16')


# In[23]:


callback = [
    EarlyStopping(monitor='val_loss', patience=15, verbose=1, mode='auto'),
    ModelCheckpoint(weight_path, monitor='val_loss', verbose=1, 
                    save_best_only=True, mode='min', save_weights_only = True)]
history = model.fit(x=x_origin_train, y=y_train, validation_data=(x_origin_valid, y_valid), batch_size=16, epochs=150, callbacks=callback)


# In[24]:


# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.show()


# In[25]:


model.load_weights(weight_path)


# In[26]:


loss = model.evaluate(x=x_origin_test, y=y_test, batch_size=16)
print('RMSE is :', loss ** 0.5)


# In[27]:


y_predict = model.predict(x_origin_test)


# In[28]:


from sklearn.metrics import r2_score

r2_score(y_test, y_predict)


# In[35]:


y_predict_pm10 = np.zeros(len(y_predict))

for i in range(len(y_predict)):
    if y_predict[i] <= 54:
        y_predict_pm10[i] = 0
    elif y_predict[i] >= 55 and y_predict[i] <= 154:
        y_predict_pm10[i] = 1
    elif y_predict[i] >= 155 and y_predict[i] <= 254:
        y_predict_pm10[i] = 2
    elif y_predict[i] >= 255 and y_predict[i] <= 354:
        y_predict_pm10[i] = 3
    elif y_predict[i] >= 355 and y_predict[i] <= 424:
        y_predict_pm10[i] = 4
    elif y_predict[i] > 424:
        y_predict_pm10[i] = 5
    else:
        print('Exception Occured!')
    
y_predict_pm10 = y_predict_pm10.astype(int)
    
    
y_predict_pm10


# In[36]:


y_test = y_test.to_numpy().tolist()
y_test


# In[37]:


y_test[1][0]


# In[38]:


#Classify the Ground Truth PM10 concentration to the air quality levels

y_test_pm10 = np.zeros(len(y_test))

for i in range(len(y_test)):
    if int(y_test[i][0])  <= 54:
        y_test_pm10[i] = 0
    elif int(y_test[i][0]) >= 55 and int(y_test[i][0]) <= 154:
        y_test_pm10[i] = 1
    elif int(y_test[i][0]) >= 155 and int(y_test[i][0]) <= 254:
        y_test_pm10[i] = 2
    elif int(y_test[i][0]) >= 255 and int(y_test[i][0]) <= 354:
        y_test_pm10[i] = 3
    elif int(y_test[i][0]) >= 355 and int(y_test[i][0]) <= 424:
        y_test_pm10[i] = 4
    elif int(y_test[i][0]) > 424:
        y_test_pm10[i] = 5
    else:
        print('Exception Occured!')

y_test_pm10 = y_test_pm10.astype(int)
        
        
y_test_pm10


# In[39]:


from sklearn.metrics import balanced_accuracy_score

balanced_accuracy_score(y_test_pm10, y_predict_pm10)


# In[41]:


#---Classification Accuracy for PM10---------

t = 0
n = 0

for i in range(len(y_predict_pm10)):
    if y_predict_pm10[i] == y_test_pm10[i]:
        t = t + 1
    else:
        n = n + 1
        
acc = t / len(y_predict_pm10)

print('Acc: ', acc, ' True: ', t, ' False: ', n)


# In[42]:


from sklearn.metrics import f1_score

f1_score(y_test_pm10, y_predict_pm10, average='macro')


# In[43]:


import seaborn as sns
from sklearn.metrics import confusion_matrix
Y_pred_classes = y_predict_pm10
Y_true = y_test_pm10
confusion_mtx = confusion_matrix(Y_true, Y_pred_classes)
f,ax = plt.subplots(figsize=(8, 8))
sns.heatmap(confusion_mtx, annot=True, linewidths=0.01,cmap="Greens",linecolor="gray", fmt= '.1f',ax=ax)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
import cv2
import numpy as np

def is_blurry(pil_img, threshold=100):
    """
    Detects if the input PIL image is blurry using variance of Laplacian.
    """
    img = np.array(pil_img.convert('L'))  # Convert to grayscale
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var < threshold

plt.title("Confusion Matrix")
plt.show()


# In[ ]:


plt.plot(y_test, label='True Label')
plt.plot(y_predict, label='Estimation Value')

# set the x-axis label
plt.xlabel('Index')

# set the y-axis label
plt.ylabel('Value')

# set the plot title
plt.title('True vs Estimation')

# Adding a legend
plt.legend()

# display the plot
plt.show()


# In[44]:


import cv2
import numpy as np

def is_blurry(pil_img, threshold=100):
    """
    Detects if the input PIL image is blurry using variance of Laplacian.
    """
    img = np.array(pil_img.convert('L'))  # Convert to grayscale
    lap_var = cv2.Laplacian(img, cv2.CV_64F).var()
    return lap_var < threshold


# In[46]:



# In[47]:


# Upload an image
from PIL import Image
import streamlit as st

st.title("PM10 Prediction with Blur Detection")

uploaded_file = st.file_uploader("Upload a sky image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    if is_blurry(image):
        st.warning("⚠️ The uploaded image seems blurry. Please retake a clear photo of the sky.")
    else:
        # Preprocess image (assuming 224x224 input)
        img = image.resize((224, 224))
        img = np.array(img) / 255.0
        img = np.expand_dims(img, axis=0)

        # Predict PM10
        pred = model.predict(img)
        pm10_value = float(pred[0][0])

        # Advisory function
        def pm10_advisory(pm10):
            if pm10 <= 50:
                return "Good air quality. You can go outside freely. No precautions needed."
            elif pm10 <= 100:
                return "Moderate air quality. People with breathing issues should limit outdoor activity."
            elif pm10 <= 250:
                return "Poor air quality. Avoid outdoor exercise. Use a mask if stepping out."
            else:
                return "Hazardous air quality. Stay indoors. Use air purifiers and wear masks if outside."

        # Display output
        st.markdown(f"### 🌫️ Predicted PM10: `{pm10_value:.2f} µg/m³`")
        st.markdown(f"### 🛡️ Advisory: {pm10_advisory(pm10_value)}")


# In[49]:





# In[ ]:




