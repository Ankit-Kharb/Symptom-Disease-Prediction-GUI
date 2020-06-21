#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
#reading the data
data=pd.read_csv("train-dis.csv")
data.head()

#extracting symptoms and diseases name
sym=list(data.columns)
sym.pop()
dis=list(data['prognosis'])
dis=set(dis)
dis=list(dis)

test=pd.read_csv("test-dis.csv")
print("No. of symptoms = " ,len(sym))
print("No. of disease = ", l
      -en(dis))
data.info()


# In[2]:


#creating a dict 
disnum={}
for i in range(len(dis)):
    disnum[dis[i]]=i
numdis={}
for i in range(len(dis)):
    numdis[i]=dis[i]
numsym={}
for i in range(len(sym) ):
    numsym[i]=sym[i]

data.replace({'prognosis':disnum},inplace=True)
test.replace({'prognosis':disnum},inplace=True)


# In[3]:


#preapring the data to feed for model
X_train=data[sym]
Y_train=data['prognosis']


X_test=test.drop(['prognosis'],axis=1)
Y_test= test['prognosis']

print(X_train.shape)
print(Y_train.shape)


# In[4]:


#training the data
from sklearn.linear_model import LogisticRegression
logisticRegr = LogisticRegression()
logisticRegr.fit(X_train, np.ravel(Y_train))


# In[7]:


#pickle the model
import pickle
filename = 'treemodel.sav'
pickle.dump(logisticRegr, open(filename, 'wb'))


# In[9]:


#creating a validation test
df1 = data.iloc[:160, :]
df1.sample(frac=1)
X_t=df1[sym]
Y_t=df1['prognosis']
print(len(Y_t)) 


# In[10]:


#analysing the model
pred1=logisticRegr.predict(X_t)
def accuracy_score(l1,l2):
    c=0
    for i in range(len(l1)):
        if l1[i]==l2[i]:
            c+=1
    return(c/len(l1))
print("Accuracy of LOGISTIC_REGRESSOR = ",accuracy_score(Y_t,pred1) )

