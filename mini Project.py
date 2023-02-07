#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
                        


# In[29]:


path = "C:/Users/himan/OneDrive/Desktop/nyc-rolling-sales.csv"
df = pd.read_csv(path)


# In[30]:


df


# In[31]:


df.sample(10)


# In[32]:


### Dropping col which are empty
del df['EASE-MENT']
### Dropping col looks like iterator
del df['Unnamed: 0']

del df['SALE DATE']


# In[33]:


df.head()


# In[34]:


# removing the duplicates

df.duplicated().sum()


# In[35]:


# Deleting the duplicates 

df = df.drop_duplicates(df.columns,keep='last')

df.duplicated().sum()


# ## Data inspection and Visualization

# In[36]:


df.shape


# In[37]:


df.info()


# In[38]:


## conerting the data type

df['TAX CLASS AT PRESENT'] = df['TAX CLASS AT PRESENT'].astype('category')
df['TAX CLASS AT TIME OF SALE'] = df['TAX CLASS AT TIME OF SALE'].astype('category')
df['LAND SQUARE FEET'] = pd.to_numeric(df['LAND SQUARE FEET'],errors='coerce')
df['GROSS SQUARE FEET'] = pd.to_numeric(df['GROSS SQUARE FEET'],errors='coerce')


# In[39]:


df.info()


# In[40]:


df['SALE PRICE'] = pd.to_numeric(df['SALE PRICE'],errors = 'coerce')
#df['BOROUGH'] = df['BOROUGH'].astype('category')


# In[41]:


df['BOROUGH'] = df['BOROUGH'].astype('category')


# In[42]:


df.info()


# In[43]:


#checking missing values
#df.isnull().sum()
df.columns[df.isnull().any()]


# ###   INFO --These are the null generated while changing the data types.

# In[51]:



missing = df.isnull().sum()/len(df)
missing = missing[missing>0]
missing.sort_values(inplace = True)
missing


# In[52]:


missing=missing.to_frame()
missing.columns=['count']
missing.index.names=['Name']
missing['Name']=missing.index
missing


# In[20]:


## plotingthe missing values


# In[54]:


sns.set(style='whitegrid',color_codes=True)
sns.barplot(x='Name', y='count',data=missing)
plt.xticks(rotation=90)
sns


# ##### We can drop the rows with missing values or we can fill them up with their mean, median or any other relation.

# In[22]:


df['LAND SQUARE FEET'] = df['LAND SQUARE FEET'].fillna(df['LAND SQUARE FEET'].mean())
df['GROSS SQUARE FEET'] = df['GROSS SQUARE FEET'].fillna(df['GROSS SQUARE FEET'].mean())


# In[23]:


df.isnull().sum()


# In[64]:


# splitting dataset

test = df[df['SALE PRICE'].isna()]
train = df[~df['SALE PRICE'].isna()]


# In[67]:


test = test.drop(columns = 'SALE PRICE')


# In[68]:


test.sample(5)


# In[69]:


test.shape


# In[70]:


train.shape


# In[73]:


train.head()


# #### finding the correlation between features

# In[75]:


corr = train.corr()
sns.heatmap(corr)


# ### LAST ROW REPRESENT THE CORRELATION OF DIFFERENT FEATURES WITH SALE PRICE

# .

# In[77]:


##FINDING THE NUMARIC CORRELATION


# In[80]:


corr['SALE PRICE'].sort_values(ascending=False)


# here we can understand that the relation between 'GROSS SQUARE FEET' and 'SALE PRICE' are very high, also with 'TOTAL UNITS','RESIDENTIAL UNITS'

# In[82]:


numaric_data= train.select_dtypes(include = [np.number])
numaric_data.describe()


# .

# .

# SALES PRICE

# In[86]:


plt.figure(figsize=(15,6))

sns.boxplot(x='SALE PRICE',data=train)
plt.ticklabel_format(style='plain',axis='x')
plt.title('Boxplot of SAL PRICE in USD')
plt.show()


# In[89]:


sns.distplot(train['SALE PRICE'])


# In[ ]:





# In[91]:


train=train[(train['SALE PRICE'] > 100000) & (train['SALE PRICE']<5000000)]


# In[92]:


sns.distplot(train['SALE PRICE'])


# In[93]:


#skewness of SALEPRICE
train['SALE PRICE'].skew()


# In[ ]:


##  SALE PRICE is highly right skewed. So, we will log transform it so that it give better results


# In[94]:


sales=np.log(train['SALE PRICE'])
print(sales.skew())
sns.distplot(sales)


# Well now we can see the symmetry and thus it is normalised.

# In[ ]:




