#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().run_line_magic('matplotlib', 'inline')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


# # Training Data

# In[3]:


df = pd.read_csv('../elena/barros_2011_training.csv')
print(df.shape)
print(df.columns)
df.head(5)


# In[4]:


df.describe()


# In[5]:


df.describe()


# # Drop and Reverse
# - Electricity generated (kWh)
# - Drop redundant columns

# In[6]:


df['kWh'] = (df['Area'] / df['Area / Electricity'])
df = df.drop(['Longitude', 'Latitude','CO2 (g/kWh)', 'CH4 (g/kWh)', 
              'Area / Electricity', 'Volume / Area', 'Unnamed: 0'], axis=1)
print(df.shape)
df.head(5)


# # Distribution of target variables

# In[7]:


df["CH4 (mg C m-2 d-1)"].plot.hist()
plt.xlabel("CH4 (mg C m-2 d-1)")
plt.ylabel("count")
plt.title("CH4", fontsize = 20)
df["CH4 (mg C m-2 d-1)"].describe()


# In[8]:


df["CO2 (mg C m¯² d¯¹)"].plot.hist()
plt.xlabel("CO2 (mg C m¯² d¯¹)")
plt.ylabel("count")
plt.title("CO2", fontsize = 20)
df["CO2 (mg C m¯² d¯¹)"].describe()


# In[9]:


columns = ["CH4 (mg C m-2 d-1)", "Area", "Age", "Erosion", "Tmax", "Tmean", "Tmin", "Volume", "orgC", "NPP", "kWh"]
for i in range(len(columns)):
    df.plot(kind="scatter", x="CO2 (mg C m¯² d¯¹)", y=columns[i]) 


# # Inspect Outliers

# In[10]:


corr = df.corr(method='pearson').drop(['CH4 (mg C m-2 d-1)']).sort_values('CH4 (mg C m-2 d-1)', ascending=False)['CH4 (mg C m-2 d-1)']
corr 


# In[11]:


sns.heatmap(df.corr())


# In[12]:


corr = df.corr(method='pearson').drop(["CO2 (mg C m¯² d¯¹)"]).sort_values("CO2 (mg C m¯² d¯¹)", ascending=False)["CO2 (mg C m¯² d¯¹)"]
corr 


# # Missing Ratio

# In[13]:


all_na = (df.isnull().sum() / len(df)) * 100
all_na = all_na.drop(all_na[all_na == 0].index).sort_values(ascending=False)
missing_data = pd.DataFrame({'Missing Ratio' :all_na})
missing_data


# In[14]:


f, ax = plt.subplots(figsize=(15, 12))
plt.xticks(rotation='90')
sns.barplot(x=all_na.index, y=all_na)
plt.xlabel('Features', fontsize=15)
plt.ylabel('Percent of missing values', fontsize=15)
plt.title('Percent missing data by feature', fontsize=15)


# # EDA

# In[15]:


ch4 = df.drop(['Name','CO2 (mg C m¯² d¯¹)'], axis=1)
ch4 = ch4[ch4['CH4 (mg C m-2 d-1)'].notna()]


# In[16]:


ch4.columns


# In[17]:


for i in range(len(ch4.columns)):
    df.plot(kind="scatter", x=columns[i], y="CH4 (mg C m-2 d-1)") 


# In[18]:


# Drop the outliers (>250 CH4, >2500 CO2)
df = df[df['CH4 (mg C m-2 d-1)']<250]
df = df[df['CO2 (mg C m¯² d¯¹)']<2500]


# # Imputing Data

# In[19]:


from sklearn.impute import KNNImputer
model_impute = KNNImputer(n_neighbors=int(np.sqrt(ch4.shape[0])))
ch4_imputed = model_impute.fit_transform(ch4)


# In[20]:


ch4_imputed = pd.DataFrame(columns=ch4.columns, data=ch4_imputed)


# In[21]:


for column in ch4_imputed:
    print(column, ':',  ch4_imputed[column].isna().sum())


# # Standardize data

# In[22]:


from sklearn.preprocessing import StandardScaler
data = ch4_imputed.drop(['CH4 (mg C m-2 d-1)'], axis=1).copy()
scaler = StandardScaler()
ch4_scaled = pd.DataFrame(scaler.fit_transform(data),columns=ch4.columns[1:])


# In[23]:


ch4_scaled.head()


# # Using Multiple Linear Regression to Predict CH4 Emissions

# In[31]:


X = ch4_scaled
y = pd.Series(ch4['CH4 (mg C m-2 d-1)'])


# In[34]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
from sklearn.metrics import mean_squared_error


# In[35]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)


# # Baseline Model

# In[36]:


from sklearn.dummy import DummyRegressor

dummy_regr = DummyRegressor(strategy="mean")
dummy_regr.fit(X_train, y_train)
dummy_regr.predict(X_train)
baseline = dummy_regr.score(X_train, y_train)
print("Baseline R^2: %f" %baseline)


# # Multiple Linear Regression

# In[37]:


ols = linear_model.LinearRegression()
ols.fit(X_train, y_train)
print("Coefficients: %s" % ols.coef_)
print("Intercept: %f" % ols.intercept_)
y_test_prediction = ols.predict(X_test)
ols.score(X_train, y_train)


# In[40]:


import statsmodels.api as sm

ols = sm.OLS(list(y_train), X_train)
results = ols.fit()
print(results.summary())


# In[ ]:




