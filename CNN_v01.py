#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[3]:


import tensorflow as tf
from tensorflow import keras


# In[4]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[5]:


class_names = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
              "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


# # Data Reshape

# In[6]:


X_train_full = X_train_full.reshape((60000, 28, 28, 1))
X_test = X_test.reshape((10000, 28, 28, 1))


# # Data normalization
# 
# We then normaize the data dimensions so that they are of aproximately the same scale

# In[7]:


X_train_n = X_train_full / 255.
X_test_n = X_test / 255.


# # Split the data into train/validation/test datasets

# In[8]:


X_valid, X_train = X_train_n[:5000], X_train_n[5000:]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test_n


# # Create the model architecture

# In[9]:


np.random.seed(42)
tf.random.set_seed(42)


# In[10]:


model = keras.models.Sequential()
model.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides=1, padding='valid', activation='relu',input_shape=(28, 28, 1)))
model.add(keras.layers.MaxPooling2D((2, 2)))

model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(300, activation='relu'))
model.add(keras.layers.Dense(100, activation='relu'))
model.add(keras.layers.Dense(10, activation='softmax'))


# In[11]:


model.summary()


# In[12]:


model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])


# In[13]:


model_history = model.fit(X_train, y_train, epochs=30, batch_size=64,
                         validation_data=(X_valid, y_valid))


# In[14]:


import pandas as pd

pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[16]:


ev = model.evaluate(X_test_n, y_test)


# In[17]:


ev


# In[18]:


X_new = X_test[:3]


# In[19]:


y_pred = model.predict_classes(X_new)
y_pred


# In[20]:


y_test[:3]


# In[21]:


print(plt.imshow(X_test[0].reshape((28,28))))


# # Pooling vs No Pooling

# In[22]:


model_a = keras.models.Sequential()
model_a.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides=1, padding='valid', activation='relu',input_shape=(28, 28, 1)))
model_a.add(keras.layers.MaxPooling2D((2, 2)))
model_a.add(keras.layers.Flatten())
model_a.add(keras.layers.Dense(300, activation='relu'))
model_a.add(keras.layers.Dense(100, activation='relu'))
model_a.add(keras.layers.Dense(10, activation='softmax'))

model_b = keras.models.Sequential()
model_b.add(keras.layers.Conv2D(filters = 32, kernel_size = (3, 3), strides=1, padding='valid', activation='relu',input_shape=(28, 28, 1)))
model_b.add(keras.layers.Flatten())
model_b.add(keras.layers.Dense(300, activation='relu'))
model_b.add(keras.layers.Dense(100, activation='relu'))
model_b.add(keras.layers.Dense(10, activation='softmax'))


# In[23]:


model_a.summary()


# In[24]:


model_b.summary()


# In[25]:


model_a.compile(loss="sparse_categorical_crossentropy",
               optimizer="sgd",
               metrics=["accuracy"])

model_b.compile(loss="sparse_categorical_crossentropy",
               optimizer="sgd",
               metrics=["accuracy"])


# In[27]:


model_history_a = model_a.fit(X_train, y_train, epochs=3,batch_size=64,
                             validation_data=(X_valid, y_valid))


# In[29]:


model_history_b = model_b.fit(X_train, y_train, epochs=3,batch_size=64,
                             validation_data=(X_valid, y_valid))

