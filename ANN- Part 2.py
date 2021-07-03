#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mpl
import matplotlib.pyplot as plt


# In[2]:


import tensorflow as tf
from tensorflow import keras


# In[3]:


from sklearn.datasets import fetch_california_housing
housing = fetch_california_housing()


# # Documentation
# https://scikit-learn.org/stable/modules/generated/sklearn.datasets.fetch_california_housing.html#examples-using-sklearn-datasets-fetch-california-housing

# In[6]:


print(housing.feature_names)


# In[7]:


from sklearn.model_selection import train_test_split
X_train_full, X_test, y_train_full, y_test = train_test_split(housing.data, housing.target, random_state=42)
X_train, X_valid, y_train, y_valid = train_test_split(X_train_full, y_train_full, random_state=42)


# In[8]:


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_valid = scaler.transform(X_valid)
X_test = scaler.transform(X_test)


# In[9]:


np.random.seed(42)
tf.random.set_seed(42)


# In[10]:


X_train.shape


# In[12]:


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])


# In[13]:


model.summary()


# In[14]:


model.compile(loss="mean_squared_error",
             optimizer=keras.optimizers.SGD(lr=1e-3),
             metrics=["mae"])


# In[18]:


model_history = model.fit(X_train, y_train, epochs=20,validation_data=(X_valid, y_valid))


# In[19]:


mae_test = model.evaluate(X_test, y_test)


# In[20]:


model_history.history


# In[21]:


pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()


# In[22]:


X_new = X_test[:3]


# In[23]:


y_pred = model.predict(X_new)
print (y_pred)
print (y_test[:3])


# In[24]:


del model


# In[26]:


keras.backend.clear_session()


# In[29]:


input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])


# In[30]:


model.summary()


# In[32]:


model.compile(loss="mean_squared_error",
             optimizer=keras.optimizers.SGD(lr=1e-3),
             metrics=['mae'])


# In[37]:


model_history = model.fit(X_train, y_train, epochs=40,validation_data=(X_valid, y_valid))


# In[38]:


mae_test = model.evaluate(X_test, y_test)


# In[35]:


model_history.history


# In[36]:


pd.DataFrame(model_history.history).plot(figsize=(8, 5))
plt.grid(True)
plt.gca().set_ylim(0, 1)

plt.show()


# # Saving and Restoring

# In[39]:


model.save("my_Func_model.h5")


# In[40]:


get_ipython().run_line_magic('pwd', '')


# In[41]:


del model


# In[42]:


keras.backend.clear_session()


# In[43]:


model = keras.models.load_model("my_Func_model.h5")


# In[44]:


model.summary()


# In[45]:


y_pred = model.predict(X_new)
print (y_pred)


# # Using Callbacks during Training

# In[46]:


keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)


# In[47]:


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])


# In[48]:


model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))


# In[49]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("Model-{epoch:02d}.h5")


# In[50]:


history = model.fit(X_train, y_train, epochs=10,
                   validation_data=(X_valid, y_valid),
                   callbacks=[checkpoint_cb])


# In[51]:


del model
keras.backend.clear_session()


# In[53]:


model = keras.models.load_model("my_Func_model.h5")


# In[54]:


mse_test = model.evaluate(X_test, y_test)


# # Best Model Only

# In[55]:


del model
keras.backend.clear_session()


# In[56]:


model = keras.models.Sequential([
    keras.layers.Dense(30, activation="relu", input_shape=[8]),
    keras.layers.Dense(30, activation="relu"),
    keras.layers.Dense(1)
])


# In[57]:


model.compile(loss="mse", optimizer=keras.optimizers.SGD(lr=1e-3))


# In[58]:


checkpoint_cb = keras.callbacks.ModelCheckpoint("Best_Model.h5", save_best_only=True)


# In[59]:


history = model.fit(X_train, y_train, epochs=10,
                   validation_data=(X_valid, y_valid),
                   callbacks=[checkpoint_cb])


# In[60]:


model = keras.models.load_model("Best_Model.h5")
mse_test = model.evaluate(X_test, y_test)


# In[ ]:




