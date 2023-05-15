#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[14]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[15]:


data=pd.read_excel(r"C:\Users\rince\Downloads\PRO1.xlsx")
data


# In[16]:


data.dtypes


# In[ ]:





# In[17]:


data.fillna(0)


# In[18]:


data['AMONE PRIME'] = data['AMONE PRIME'].astype(str).astype(int)


# In[19]:


data.head()
data


# In[20]:


data=data.drop(["NAME","CURRENT STATUS","GENDER"],axis=1)
data


# In[21]:


data.dtypes


# # cheking unfilled data

# In[22]:


sns.heatmap(data.isnull())
sns


# In[23]:


data


# # USING LABEL ENCODER

# In[24]:


label_encoder=preprocessing.LabelEncoder()
data["ADDICTED"]=label_encoder.fit_transform(data["ADDICTED"])
data


# # TRAIN TEST SPLIT

# In[25]:


x=data[["WHATSAAP","FACEBOOK","INSTAGARAME","YOUTUBE\n","NETFLIX","AMONE PRIME","TOTAL"]]
y=data["ADDICTED"]
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.8,random_state=0)


# In[26]:


xtest.info()


# In[ ]:





# In[27]:


print(xtrain)


# In[28]:


print(xtest)


# In[29]:


print(ytrain)


# In[30]:


print(ytest)


# In[31]:


model=LogisticRegression()
model.fit(xtrain,ytrain)
print(model.predict(xtest))


# In[32]:


model.score(xtest,ytest)


# # DECISION TREE CLASIFIER

# In[33]:


from sklearn.tree import DecisionTreeClassifier
model=DecisionTreeClassifier()


# In[34]:


model.fit(xtrain,ytrain)


# In[35]:


xtrain


# In[36]:


model.score(xtest,ytest)


# In[37]:


model.predict(xtest)


# 

# #   GAUSSIAN NAIVEBAYES

# In[38]:


from sklearn.naive_bayes import GaussianNB
model2=GaussianNB()


# In[39]:


model2.fit(xtrain,ytrain)


# In[40]:


model2.score(xtest,ytest)


# # RANDOM FOREST CLASSIFIRE

# In[41]:


from sklearn.ensemble import RandomForestClassifier


# In[42]:


model3=RandomForestClassifier(n_estimators=100)
model3.fit(xtrain,ytrain)


# In[43]:


model3.score(xtest,ytest)


# #            SUPPORT VECTOR CLASSIFIER

# In[44]:


from sklearn.svm import SVC
model4=SVC()


# In[45]:


model4.fit(xtrain,ytrain)


# In[46]:


model4.score(xtest,ytest)


# 

# # KNearest Neighbour Classifier

# In[47]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(xtrain,ytrain)


# In[48]:


ypred=classifier.predict(xtest)


# In[49]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))


# # POLYNIMIAL KERNAL

# In[50]:


model5=SVC(kernel="poly",degree=8)
model5.fit(xtrain,ytrain)


# In[51]:


model5.score(xtest,ytest)


# In[52]:


ypred=model5.predict(xtest)
print(ypred)


# # KNeighborsClassifier

# In[53]:


from sklearn.preprocessing import StandardScaler
st_x=StandardScaler()
xtrain=st_x.fit_transform(xtrain)
xtest=st_x.transform(xtest)


# In[54]:


from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5)
classifier.fit(xtrain,ytrain)


# In[55]:


ypred=classifier.predict(xtest)


# In[56]:


from sklearn.metrics import accuracy_score
print(accuracy_score(ytest,ypred))


# # Adaptive Boosting

# In[57]:


from sklearn.metrics import confusion_matrix
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import AdaBoostRegressor,RandomForestRegressor


# In[58]:


adaboost=AdaBoostClassifier(n_estimators=50,base_estimator=None,learning_rate=0.3,random_state=1)


# In[59]:


adaboost.fit(xtrain,ytrain)


# In[60]:


adaboost.score(xtest,ytest)


# In[61]:


ypred=adaboost.predict(xtest)
print(ypred)


# # Kernal

# In[62]:


model6=SVC(kernel="linear")
model6.fit(xtrain,ytrain)


# In[63]:


ypred=model6.predict(xtest)
print(ypred)


# In[64]:


from sklearn.metrics import accuracy_score
print("accuracy of our model is",accuracy_score(ytest,ypred))


# In[65]:


model.predict([[300,60,600,120,240,10,1330]])


# In[66]:


model.predict([[0,0,0,0,0,0,0,]])


# In[67]:


model.predict([[0,0,1440,0,0,0,1440]])


# In[68]:


model6.predict([[300,60,600,120,240,10,1330]])


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




