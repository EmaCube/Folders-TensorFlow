#!/usr/bin/env python
# coding: utf-8

# # Loading Data from CSV file
# The pandas library in Phyton provides excellent, built-in support for time series data.
# Pandas represents time series datasets as a Series.
# A Series is a one-dimensional array with a time label for each row.
# A Dataframe is a collection of series.

# In[1]:


import pandas as pd


# In[2]:


# Download csv file from resources and put it in working directory
dataframe = pd.read_csv('C:/Users/Usuario/Documents/Data Files/7. ST Academy - Time Series resource files/daily-total-female-births-CA.csv', header=0)


# # First five records

# In[3]:


dataframe.head()


# # Data Type

# In[4]:


dataframe['date'].dtype


# # Loading data with parse_dates

# In[5]:


df2 = pd.read_csv('C:/Users/Usuario/Documents/Data Files/7. ST Academy - Time Series resource files/daily-total-female-births-CA.csv', header=0, parse_dates=[0])


# # First five records

# In[6]:


df2.head()


# # Data Type

# In[7]:


df2['date'].dtype


# In[8]:


series = pd.read_csv('C:/Users/Usuario/Documents/Data Files/7. ST Academy - Time Series resource files/daily-total-female-births-CA.csv', header=0, parse_dates=[0], index_col=0, squeeze=True)


# # First 5 records

# In[9]:


series.head()


# # Exploring Time Series Data

# In[10]:


series.shape


# In[11]:


df2.shape


# # Querying by time

# In[12]:


print(series['1959-01'])


# In[13]:


df2[(df2['date'] > '1959-01-01') & (df2['date'] <= '1959-01-21')]


# # Descriptive Statistics

# In[14]:


series.describe()


# In[15]:


df2.describe()


# # time series - data visualization

# In[16]:


from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[17]:


Dataviz_df = df2.copy()


# In[21]:


Dataviz_df.head(10)


# In[19]:


Dataviz_df['births'].plot()


# In[20]:


Dataviz_df.index = Dataviz_df['date']


# In[22]:


Dataviz_df['births'].plot()


# # Zooming in

# In[23]:


Dataviz_df2 = Dataviz_df[(Dataviz_df['date'] > '1959-01-01')&(Dataviz_df['date']<='1959-01-10')].copy()


# In[24]:


Dataviz_df2


# In[25]:


Dataviz_df2['births'].plot()


# # Trendline

# In[26]:


import seaborn as sns


# In[28]:


sns.regplot(x= df2.index.values, y=df2['births'])


# In[29]:


sns.regplot(x= df2.index.values, y=df2['births'], order = 2)


# In[ ]:




