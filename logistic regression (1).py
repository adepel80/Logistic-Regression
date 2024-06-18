#!/usr/bin/env python
# coding: utf-8

# In[6]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import chart_studio.plotly as py
import plotly.express as px
#import plotly.subplots import make_subplots
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import make_classification
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score,r2_score
plt.style.use ("dark_background")

import os


# In[11]:


x, y = make_classification(
    n_samples=100,
    n_features=1,
    n_classes=2,
    n_clusters_per_class=1,
    flip_y=0.03,
    n_informative=1,
    n_redundant=0,
    n_repeated=0
)
print(y)


# In[12]:


#scater plot
plt.scatter(x, y, c=y,cmap='rainbow')
plt.title('Scatter Plot of Logistic Regression')
plt.show()


# In[13]:


#split the dataset into training and test dataset
x_train, x_test, y_train, y_test = train_test_split(x, y, random_state = 1)


# In[14]:


#perform logistic regression
x_train.shape


# In[17]:


#fit regression object
log_reg = LogisticRegression()
log_reg.fit(x_train, y_train)


# In[18]:


#make predictions using models using the test dataset
y_pred = log_reg.predict(x_test)


# In[19]:


#display the confusion matrix
confusion_matrix(y_test, y_pred)


# In[ ]:


#We are good becos the FN & FP is low (false posi and false negative)

