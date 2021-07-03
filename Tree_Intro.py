#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# In[16]:


df = pd.read_csv("Movie_regression.csv", header=0)


# In[10]:


df.info()


# In[ ]:





# ### Missing Value Imputation

# In[17]:


df["Time_taken"].fillna(value=df["Time_taken"].mean(), inplace = True)


# In[ ]:





# In[ ]:





# ### Dummy variable creation

# In[18]:


df = pd.get_dummies(df,columns = ["3D_available","Genre"],drop_first = True)


# In[20]:


df.head()


# ### X-y split

# In[21]:


X = df.loc[:,df.columns!="Collection"]
type(X)


# In[ ]:





# In[ ]:





# In[22]:


y = df["Collection"]
type(y)


# In[ ]:





# In[ ]:





# ### Test-Train Split

# In[23]:


from sklearn.model_selection import train_test_split


# In[24]:


X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.2,random_state=0)


# In[ ]:





# ### Traing Regression Tree

# In[25]:


from sklearn import tree
regtree = tree.DecisionTreeRegressor(max_depth = 3)


# In[26]:


regtree.fit(X_train, y_train)


# ### Predict values using trained model

# In[27]:


y_train_pred = regtree.predict(X_train)
y_test_pred = regtree.predict(X_test)


# ### Model Performance

# In[28]:


from sklearn.metrics import mean_squared_error, r2_score


# In[29]:


mean_squared_error(y_test, y_test_pred)


# In[30]:


r2_score(y_train, y_train_pred)


# In[31]:


r2_score(y_test, y_test_pred)


# ### Plotting a decision tree

# In[33]:


dot_data = tree.export_graphviz(regtree, out_file=None)


# In[34]:


from IPython.display import Image


# In[35]:


import pydotplus


# In[36]:


graph = pydotplus.graph_from_dot_data(dot_data)
Image(graph.create_png())

