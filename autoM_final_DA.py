#!/usr/bin/env python
# coding: utf-8

# In[538]:


import pandas as pd
import numpy as np
df1=pd.read_csv("C://Users//Yasiru//OneDrive//Desktop//python//automobile2.csv")
df1.head(40)


# In[539]:


df1.info()


# In[540]:


df1.dropna(inplace=True)
df1.info()


# In[541]:


from sklearn.model_selection import train_test_split

X = df1.drop(['price'], axis=1)
Y=df1['price']


# In[542]:


X


# In[543]:


Y


# In[544]:


X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=0)


# In[545]:


df2 = X_train.join(Y_train)


# In[546]:


df2


# In[547]:


import matplotlib.pyplot as plt
import seaborn as sns

df2.hist(figsize=(15,8))


# In[548]:


plt.figure(figsize=(15,8))
sns.heatmap(df2.corr(), annot=True, cmap='YlGnBu')


# In[549]:


df2 = df2.join(pd.get_dummies(df1['body_style'])).drop(['body_style'], axis=1)


# In[550]:


df2


# In[551]:


df2['door_cnt'].replace({'two': 2, 'four': 4}, inplace=True)
df2


# In[552]:


df2=df2.join(pd.get_dummies(df2['eng_loc']))


# In[553]:


df2


# In[554]:


df2=df2.join(pd.get_dummies(df2['fuel_sys']))


# In[555]:


df2


# In[556]:


df2.drop(['make', 'fuel', 'asp', 'drive_wheels', 'eng_loc', 'fuel_sys'], axis=1, inplace=True) 


# In[557]:


df2


# In[558]:


df2.drop(['eng_type'], axis=1, inplace=True)


# In[559]:


df2


# In[560]:


df2['no of cylndrs'].replace({'three':3, 'four': 4, 'six': 6, 'five': 5, 'eight':8, 'two':2, 'twelve':12}, inplace=True)
df2


# In[561]:


df2


# In[562]:


df2


# In[563]:


X_train1 = df2.drop(['price'], axis = 1)
Y_train1 = df2['price']


# In[564]:


from sklearn.preprocessing import StandardScaler

scale = StandardScaler()
X_train1_s = scale.fit_transform(X_train1)

from sklearn.linear_model import LinearRegression

lin = LinearRegression()
lin.fit(X_train1_s, Y_train1)


# In[565]:


test_data = X_test.join(Y_test)


# In[566]:


test_data = test_data.join(pd.get_dummies(df1['body_style'])).drop(['body_style'], axis=1)
test_data['door_cnt'].replace({'two': 2, 'four': 4}, inplace=True)
test_data=test_data.join(pd.get_dummies(test_data['eng_loc']))
test_data=test_data.join(pd.get_dummies(test_data['fuel_sys']))
test_data.drop(['make', 'fuel', 'asp', 'drive_wheels', 'eng_loc', 'fuel_sys'], axis=1, inplace=True) 
test_data.drop(['eng_type'], axis=1, inplace=True)
test_data['no of cylndrs'].replace({'three': 3, 'four': 4, 'six': 6, 'five': 5, 'eight':8, 'two':2, 'twelve':12}, inplace=True)


# In[567]:


X_test1 = test_data.drop(['price'], axis=1)
Y_test1 = test_data['price']


# In[568]:


X_test1_s = scale.fit_transform(X_test1)


# In[569]:


lin.score(X_test1_s, Y_test1)


# In[570]:


#hyper_parameter_tuning
from sklearn.ensemble import RandomForestRegressor

ranfrst = RandomForestRegressor()
ranfrst.fit(X_train1_s, Y_train1)


# In[571]:


Y_train1_np = Y_train1.to_numpy()


# In[572]:


ranfrst.score(X_test1_s, Y_test1)


# In[573]:


Y_test1_np = Y_test1.to_numpy()
Y_test1_np.shape, X_test1_s.shape


# In[574]:


yhat = ranfrst.predict(X_test1_s)
yhat


# In[575]:


from sklearn.metrics import mean_absolute_error, mean_absolute_error, mean_squared_error

mean_absolute_error(yhat, Y_test1_np), mean_squared_error(yhat, Y_test1_np)


# In[576]:


from sklearn.model_selection import cross_val_score

scores = cross_val_score(ranfrst, X_test1_s, Y_test1)
np.mean(scores)


# In[577]:


from sklearn.linear_model import Ridge

Ridge_model = Ridge(alpha=1.6)
Ridge_model.fit(X_train1_s, Y_train1)
Ridge_model.score(X_test1_s, Y_test1)


# In[578]:


from tqdm import tqdm

Rsqu_test = []
Rsqu_train = []
dummy1 = []
Alpha = 10 * np.array(range(0,1000))
pbar = tqdm(Alpha)

for alpha in pbar:
    RigeModel = Ridge(alpha=alpha) 
    RigeModel.fit(X_train1_s, Y_train1)
    test_score, train_score = RigeModel.score(X_test1_s, Y_test1), RigeModel.score(X_train1_s, Y_train1)
    
    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})

    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)


# In[579]:


width = 12
height = 10
plt.figure(figsize=(width, height))

plt.plot(Alpha,Rsqu_test, label='validation data  ')
plt.plot(Alpha,Rsqu_train, 'r', label='training Data ')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.legend()

