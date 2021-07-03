#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# # Building an image Classifier
# First let´s install and import TensorFlow and Keras
# 
# pip install --upgrade tensorflow==2.0.0-rc1
# 

# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


keras.__version__


# In[4]:


tf.__version__


# # Keras
# Link: https://keras.io/datasets/#fashion-mnist-database-of-fashion-articles
# Dataset of 60.000 28x28 grayscale images of 10 fashion categories, along with a test set of 10.000 images.

# # Usage:
# from keras datasets import fashion_mnist
#    
#    (x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()
#    
# Returns: 2 tuples:
# 
#    1. x_train, y_test: uint8 array of grayscale image data with shape (num_samples, 28, 28).
#    2. y_train, y_test: uint8 array of labels (integers in range 0-9) with shape (num_samples,).

# In[5]:


fashion_mnist = keras.datasets.fashion_mnist
(X_train_full, y_train_full), (X_test, y_test) = fashion_mnist.load_data()


# In[6]:


plt.imshow(X_train_full[10])


# In[7]:


y_train_full[1]


# In[8]:


class_names = ["T-shirt/top","Trouser","Pullover","Dress","Coat","Sandal","Shirt","Sneaker","Bag","Ankle boot"]


# In[9]:


class_names[y_train_full[10]]


# In[10]:


X_train_full[10]


# # Data normalization
# we then normalize the data dimensions so that they are of approximately the same scale.

# In[11]:


X_train_n = X_train_full / 255.
X_test_n = X_test / 255.


# In[12]:


X_valid, X_train = X_train_n[:5000], X_train_n[5000::]
y_valid, y_train = y_train_full[:5000], y_train_full[5000:]
X_test = X_test_n


# In[13]:


X_valid[0]


# In[14]:


np.random.seed(42)
tf.random.set_seed(42)


# In[15]:


model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28,28]))
model.add(keras.layers.Dense(300,activation="relu"))
model.add(keras.layers.Dense(100,activation="relu"))
model.add(keras.layers.Dense(10,activation="softmax"))


# In[16]:


model.summary()


# In[17]:


import pydot
keras.utils.plot_model(model)


# In[18]:


weights, biases = model.layers[1].get_weights()


# In[19]:


weights


# In[20]:


weights.shape


# In[21]:


biases


# In[22]:


biases.shape


# In[23]:


model.compile(loss="sparse_categorical_crossentropy",
             optimizer="sgd",
             metrics=["accuracy"])


# In[24]:


model_history = model.fit(X_train, y_train, epochs=30,
                         validation_data=(X_valid, y_valid))


# In[25]:


model_history.params


# In[26]:


model_history.history


# In[28]:


import pandas as pd

pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)
plt.show()


# In[29]:


model.evaluate(X_test, y_test)


# In[30]:


X_new = X_test[:3]


# In[31]:


y_proba = model.predict(X_new)
y_proba.round(2)


# In[32]:


y_pred = model.predict_classes(X_new)
y_pred


# In[33]:


np.array(class_names)[y_pred]


# In[34]:


print(plt.imshow(X_test[0]))


# In[35]:


print(plt.imshow(X_test[1]))


# In[36]:


print(plt.imshow(X_test[2]))


# In[ ]:




