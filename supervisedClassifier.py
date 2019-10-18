#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Numpy for matrix operations
import numpy as np

# Pandas files input operations
import pandas as pd

#sklearn imports
# Fitting Random Forest Regression to the dataset 

from sklearn.ensemble import RandomForestClassifier 

# Using Skicit-learn to split data into training and testing sets
from sklearn.model_selection import train_test_split


# In[2]:


kMeansOut = pd.read_csv("dataSet.csv")
kMeansOut = kMeansOut.drop(['Unkown','SESSION_ID','PLAN_ID','SQL_TEXT_TYPE','newline'],axis=1)
kMeansOut.head()


# In[3]:


X = pd.get_dummies(kMeansOut[kMeansOut.columns[0:7]])   
X.head()


# In[4]:


Y = kMeansOut[['Clusters']]
Y.head()


# In[5]:


# Split the data into training and testing sets
train_features, test_features, train_labels, test_labels = train_test_split(X, Y, test_size = 0.25, random_state = 42)

test_labels = np.array(test_labels)


# In[6]:


print('Training Features Shape:', train_features.shape)
print('Training Labels Shape:', train_labels.shape)
print('Testing Features Shape:', test_features.shape)
print('Testing Labels Shape:', test_labels.shape)


# In[8]:


# Instantiate model with 10000 decision trees
rf = RandomForestClassifier(n_estimators=10000, max_depth=100, random_state=42)
#rf
# Train the model on training data
rf.fit(train_features, train_labels)


# In[9]:


# Use the forest's predict method on the test data
predictions = rf.predict(test_features)
predictions = np.array(predictions)


# In[10]:


accuracy_test = 100 - np.mean(np.abs(predictions - test_labels)) * 100

print("test accuracy: {} %".format(accuracy_test))


# In[12]:


from sklearn.externals import joblib

# Save the model as a pickle in a file 
joblib.dump(rf, 'randomForest.pkl') 
   




