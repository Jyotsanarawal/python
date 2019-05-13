#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import pandas as pd
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[ ]:


data_dir='E:\ML Course\Tree Based Models\Data'


# In[ ]:


os.chdir(data_dir)


# In[ ]:


hr_data=pd.read_csv('hr.csv')


# In[ ]:


hr_data.head()


# In[ ]:


hr_data.isnull().sum()


# In[ ]:


hr_data.dtypes


# In[ ]:


hr_data['sales'].unique().tolist()


# In[ ]:


hr_data.rename(columns={'sales':'dept'},inplace=True)


# In[ ]:


hr_data['salary'].head()


# In[ ]:


X=hr_data.drop('left',axis=1)
y=hr_data['left']


# In[ ]:


X.head()


# In[ ]:


X=pd.get_dummies(X)


# In[ ]:


X.head()


# In[ ]:


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.20,random_state=400)


# In[ ]:


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier


# In[ ]:


clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=20,random_state=400,
                      base_estimator=DecisionTreeClassifier())


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


clf.oob_score_


# In[ ]:


clf.score(X_test,y_test)


# In[ ]:


for w in range(10,300,20):
    clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=w,random_state=400,
                          base_estimator=DecisionTreeClassifier())
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print 'For n_estimators = '+str(w)
    print 'OOB score is '+str(oob)
    print '************************'


# In[ ]:


#Finalizing on a tree model with 150 trees
clf=BaggingClassifier(oob_score=True,n_jobs=-1,n_estimators=150,random_state=400,
                      base_estimator=DecisionTreeClassifier())
clf.fit(X_train,y_train)


# In[ ]:


# Feature Importance
clf.estimators_


# In[ ]:


print clf.estimators_[0]


# In[ ]:


print clf.estimators_[0].feature_importances_


# In[ ]:


# We can extract feature importance from each tree then take a mean for all trees
imp=[]
for i in clf.estimators_:
    imp.append(i.feature_importances_)
imp=np.mean(imp,axis=0)


# In[ ]:


feature_importance=pd.Series(imp,index=X.columns.tolist())


# In[ ]:


feature_importance.sort_values(ascending=False)


# In[ ]:


feature_importance.sort_values(ascending=False).plot(kind='bar')


# ### Random Forest Classifier

# In[ ]:


from sklearn.ensemble import RandomForestClassifier


# In[ ]:


clf=RandomForestClassifier(n_estimators=80,oob_score=True,n_jobs=-1,random_state=400)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


clf.oob_score_


# In[ ]:


for w in range(10,300,20):
    clf=RandomForestClassifier(n_estimators=w,oob_score=True,n_jobs=-1,random_state=400)
    clf.fit(X_train,y_train)
    oob=clf.oob_score_
    print 'For n_estimators = '+str(w)
    print 'OOB score is '+str(oob)
    print '************************'


# In[ ]:


#Finalize 190 trees
clf=RandomForestClassifier(n_estimators=190,oob_score=True,n_jobs=-1,random_state=400)


# In[ ]:


clf.fit(X_train,y_train)


# In[ ]:


clf.oob_score_


# In[ ]:


clf.feature_importances_


# In[ ]:


imp_feat=pd.Series(clf.feature_importances_,index=X.columns.tolist())


# In[ ]:


imp_feat.sort_values(ascending=False)


# In[ ]:


imp_feat.sort_values(ascending=False).plot(kind='bar')


# In[ ]:




