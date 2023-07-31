#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.tree import DecisionTreeRegressor,DecisionTreeClassifier
import matplotlib.pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')
#import train and test data.
train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
name=train.Name
train.head()


# In[2]:


train.head(5)


# In[3]:


train.shape


# In[4]:


train.isnull().sum()


# In[6]:


import numpy as np
import pandas as pd
from sklearn.preprocessing import Imputer
Imp=Imputer(missing_values='NaN',strategy='median',axis=1)
new=Imp.fit_transform(train.Age.values.reshape(1,-1))
train['Age2']=new.T


# In[7]:


train.drop('Age',axis=1,inplace=True)


# In[8]:


train.isnull().sum()


# In[11]:


train.shape


# In[12]:


train.Survived.value_counts()/len(train)*100


# In[13]:


train.describe()


# In[14]:


train.groupby('Survived').mean()


# In[15]:


train.groupby('Sex_male').mean()


# In[16]:


train.corr()


# In[17]:


plt.subplots(figsize = (15,8))
sns.heatmap(train.corr(), annot=True,cmap="PiYG")
plt.title("Correlations Among Features", fontsize = 20);


# In[19]:


plt.subplots(figsize = (11,8))
sns.barplot(x = "Sex_male", y = "Survived", data=train, edgecolor=(0,0,0), linewidth=2)
plt.title("Survived/Non-Survived Passenger Gender Distribution", fontsize = 20)
labels = ['Female', 'Male']
plt.ylabel("% of passenger survived", fontsize = 11)
plt.xlabel("Gender",fontsize = 11)
plt.xticks(sorted(train.Sex_male.unique()), labels)


# In[21]:


sns.set(style='darkgrid')
plt.subplots(figsize = (11,8))
ax=sns.countplot(x='Sex_male',data=train,hue='Survived',edgecolor=(0,0,0),linewidth=2)
train.shape
## Fixing title, xlabel and ylabel
plt.title('Passenger distribution of survived vs not-survived',fontsize=25)
plt.xlabel('Gender',fontsize=15)
plt.ylabel("# of Passenger Survived", fontsize = 10)
labels = ['Female', 'Male']
#Fixing xticks.
plt.xticks(sorted(train.Survived.unique()),labels)
## Fixing legends
leg = ax.get_legend()
leg.set_title('Survived')
legs=leg.texts
legs[0].set_text('No')
legs[1].set_text('Yes')


# In[29]:


train.head(7)


# In[23]:


plt.subplots(figsize = (10,10))
ax=sns.countplot(x='Pclass',hue='Survived',data=train)
plt.title("Passenger Class Distribution - Survived vs Non-Survived", fontsize = 25)
leg=ax.get_legend()
leg.set_title('Survival')
legs=leg.texts

legs[0].set_text('No')
legs[1].set_text("yes")


# In[26]:


plt.subplots(figsize=(10,8))
sns.kdeplot(train.loc[(train['Survived'] == 0),'Pclass'],shade=True,color='r',label='Not Survived')
ax=sns.kdeplot(train.loc[(train['Survived'] == 1),'Pclass'],shade=True,color='b',label='Survived' )

labels = ['First', 'Second', 'Third']
plt.xticks(sorted(train.Pclass.unique()),labels)


# In[31]:


X=train.drop('Survived',axis=1)
y=train['Survived'].astype(int)


# In[ ]:


log['Classifier']=acc_dict.keys()
log['Accuracy']=acc_dict.values()
log.set_index([[0,1,2,3,4,5,6,7,8,9,10]])
get_ipython().run_line_magic('matplotlib', 'inline')
sns.set_color_codes("muted")
ax=plt.subplots(figsize=(10,8))
ax=sns.barplot(y='Classifier',x='Accuracy',data=log,color='b')
ax.set_xlabel('Accuracy',fontsize=20)
plt.ylabel('Classifier',fontsize=20)
plt.grid(color='r', linestyle='-', linewidth=0.5)
plt.title('Classifier Accuracy',fontsize=20)


# In[33]:


train.head()


# In[ ]:




