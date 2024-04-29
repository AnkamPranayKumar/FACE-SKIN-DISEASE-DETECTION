#!/usr/bin/env python
# coding: utf-8

# In[2]:


import warnings
from warnings import simplefilter
simplefilter("ignore")


# In[3]:


import numpy as np
import os
import cv2
import random
import matplotlib.pyplot as plt

from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense


# In[4]:


data_path = 'C:\VS_Project\diseases'


# In[5]:


train_data = []
val_data = []

for folder in os.listdir(data_path):
    folder_path = os.path.join(data_path, folder)
    file = os.listdir(folder_path)
    num_train = int(0.8 * len(file))
    files_train = random.sample(file, num_train)
    files_val = list(set(file) - set(files_train))
    
    for file in files_train:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224,224))
        train_data.append((img, folder))
        
    for file in files_val:
        file_path = os.path.join(folder_path, file)
        img = cv2.imread(file_path)
        img = cv2.resize(img, (224,224))
        val_data.append((img, folder))


# In[6]:


fig, axes = plt.subplots(2, 4, figsize=(10, 5))
plt.suptitle('LABELS OF EACH IMAGE')

for (img, label), ax in zip(random.sample(train_data, 12), axes.flatten()):
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.grid(False)
    ax.set_title(label)
    ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB) )

plt.show()


# In[7]:


base_model = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False


# In[8]:


num_classes = 10
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(512, activation='relu')(x)
predictions = Dense(num_classes, activation='softmax')(x)


# In[9]:


model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])



# In[10]:


from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# train_data = [(preprocess_input(input), label) for input, label in train_data]
# val_data = [(preprocess_input(input), label) for input, label in val_data]

X_train, y_train = zip(*train_data)
X_val, y_val = zip(*val_data)

X_train = preprocess_input(np.array(X_train))
X_val = preprocess_input(np.array(X_val))

le = LabelEncoder()
y_train_encoded = le.fit_transform(y_train)
y_val_encoded = le.transform(y_val)

y_train_one_hot = to_categorical(y_train_encoded, num_classes)
y_val_one_hot = to_categorical(y_val_encoded, num_classes)


# In[11]:


EPOCHS = 6
BATCH_SIZE = 32
history = model.fit(X_train, y_train_one_hot, validation_data=(X_val, y_val_one_hot),
                    epochs=EPOCHS, batch_size=BATCH_SIZE)

# Import pickle library
# import pickle
# def predict_image(model, image_path):
#     img = cv2.imread(image_path)
#     if img is None:
#         print(f"Could not load image at {image_path}")
#         return None
#     img = cv2.resize(img, (224,224))
#     img = preprocess_input(np.array([img]))
#     predictions = model.predict(img)
#     return predictions

# Save the model to a pickle file
#with open('C:\VS_Project\my_models1.pkl', 'wb') as file:
  #  pickle.dump(model, file)
# Save the model to a h5 file
model.save('C:\VS_Project\my_models1.h5')

# Load the model from a h5 file
from tensorflow.keras.models import load_model

model = load_model('C:\VS_Project\my_models1.h5')
# image_path = 'C:\VS_Project\images\acne-cystic-1.jpg'
# predictions = predict_image(model, image_path)
# print(predictions)

# In[ ]:




