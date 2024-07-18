#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import math
import statistics
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats


# In[2]:


df= pd.read_csv('sales_data_with_discounts.csv')


# In[3]:


df


# In[4]:


df.head()


# In[5]:


df.tail()


# In[6]:


df.size


# In[7]:


df.shape


# In[8]:


num_col= df.select_dtypes(include='number').columns.tolist()   ##select only Numerical columns
                                                               ##.columns- extract the columns names from the selected columns
                                                            ##.tolist()-convert the columns names into a python list 


# In[9]:


num_col


# In[10]:


df.dtypes


# In[11]:


#calculate the mean,median,mode & std dev
df.describe()


# In[12]:


df['Volume'].median()


# In[13]:


stats.mode(df["Total Sales Value"])


# In[14]:


df['Volume'].mode()


# In[15]:


plt.hist(num_col) ## plot histogram for numerical column


# ## Data Visulization

# In[33]:


import matplotlib.pyplot as plt 
for col in num_col:
    plt.hist(df[col])
    plt.title(f'Histogram of {col}')
    plt.show()


# In[34]:


for col in num_col :
    plt.boxplot(df[col])
    plt.title(f'Boxplot of {col}')
    plt.show()                       ### Plot the box plot of Numerical columns


# In[19]:


## Bar chart analysis for categorical column.#Identify categroical column in the dataset
cat_col= df.select_dtypes(include='object').columns.tolist()
cat_col


# In[35]:


for col in cat_col:
    df[col].value_counts().plot(kind='bar')
    plt.title(f'Bar chart of {col}')
    plt.show()


# In[30]:


num_col


# In[31]:


##Standardize the numerical columns using the formula: z=x-mu/sigma
for col in num_col:
    mean = df[col].mean()
    median = df[col].median()
    mode = df[col].mode()[0]
    std_dev = df[col].std()
    print(f'column: {col}\nMean: {mean}\nMedian: {median}\nMode: {mode}\nStrandard Deviation: {std_dev}\n')


# Interpretation
# 
# Volume: The average volume of sales is around 5.07 units, with a median of 4 units, indicating that most of the sales volumes are around 4 to 5 units. The mode is 3, which means that 3 units is the most common sales volume. The standard deviation is 4.23, indicating a moderate spread around the mean.
# 
# Avg Price: The average price is approximately 10453.43 units, but the median is only 1450 units. This suggests that there are some very high prices pulling the mean up, but most prices are closer to 1450 units. The mode is 400, which is the most common price. The standard deviation is quite high (18079.90), indicating a wide spread in prices.
# 
# Total Sales Value: The average total sales value is around 33812.84 units, with a median of 5700 units. This suggests that there are some very high sales values pulling the mean up, but most sales values are closer to 5700 units. The mode is 24300, which is the most common sales value. The standard deviation is quite high (50535.07), indicating a wide spread in sales values.
# 
# Discount Rate (%): The average discount rate is 15.16%, with a median of 16.58%. This suggests that most of the discount rates are around 15% to 16%. The mode is 5.01%, which means that 5.01% is the most common discount rate. The standard deviation is 4.22%, indicating a moderate spread around the mean.
# 
# Discount Amount: The average discount amount is around 3346.50 units, with a median of 988.93 units. This suggests that there are some very high discount amounts pulling the mean up, but most discount amounts are closer to 988.93 units. The mode is 69.18, which is the most common discount amount. The standard deviation is quite high (4509.90), indicating a wide spread in discount amounts.
# 
# Net Sales Value: The average net sales value is around 30466.34 units, with a median of 4677.79 units. This suggests that there are some very high net sales values pulling the mean up, but most net sales values are closer to 4677.79 units. The mode is 326.97, which is the most common net sales value. The standard deviation is quite high (46358.66), indicating a wide spread in net sales values.

# In[36]:


Q1 = df[num_col].quantile(0.25)   ##	Create the interquartile range.
Q3 = df[num_col].quantile(0.75)
IQR = Q3 -Q1
outliers = df[(df[num_col] < (Q1 - 1.5*IQR)) | (df[num_col] > (Q3 + 1.5*IQR))]
print('Outliers:')
print(outliers)


# Standardization or Z-score normalization is a technique that rescales data to have a mean of 0 and standard deviation of 1. It’s done using the formula
# 
# Z=σx−μ where
# 
# x is a value
# 
# μ is the mean
# 
# σ is the standard deviation
# 
# This process makes different datasets directly comparable and is often used in data analysis and machine learning.
# 

# In[37]:


from sklearn.preprocessing import StandardScaler


# In[38]:


scaler = StandardScaler()
df[num_col] = scaler.fit_transform(df[num_col])


# In[39]:


df[num_col].describe()


# In[40]:


plt.figure(figsize=(12,8))
sns.histplot(df[num_col],color = 'blue',kde = True)
plt.title('sales and discount Data')
plt.show()


# In[41]:


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
data_to_scale = df['Volume'].values.reshape(-1,1)
df['column1_standardized']= scaler.fit_transform(data_to_scale)


# In[43]:


plt.figure(figsize = (12,8))
sns.histplot(df['column1_standardized'], color = 'Blue', kde = True)
plt.title('Standardized Data')
plt.show()


# In[44]:


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler()
data_to_scale = df['Avg Price'].values.reshape(-1,1)
df['column2_standardized']= scaler.fit_transform(data_to_scale)


# In[46]:


plt.figure(figsize = (12,8))
sns.histplot(df['column2_standardized'], color = 'Blue', kde = True)
plt.title('Standardized Data')
plt.show()


# # Conversion of Categorical Data into Dummy Variables
# Categorical data is converted into dummy variables (one-hot encoding) because machine learning algorithms require numerical input. This conversion allows these algorithms to perform mathematical operations on the data. It also ensures that all categories are given equal importance, which can lead to better model performance. However, it can increase data dimensionality if the categorical feature has many unique categories

# In[47]:


print(df.columns)


# In[48]:


df['Day'].value_counts()


# In[49]:


from sklearn import preprocessing
trans = preprocessing.OneHotEncoder()
trans.fit(df)


# In[50]:


df_encoded = pd.get_dummies(df, columns=['Day'])


# In[51]:


print("\nTransformed DataFrame with One-Hot Encoding:")
print(df_encoded)


# # conclusion
# 
# Data preprocessing, including steps like standardization and one-hot encoding, is essential in data analysis and machine learning for several key reasons:
# 
# Enhances Model Performance: Standardization helps normalize data scales, improving the effectiveness of algorithms sensitive to feature scale, while one-hot encoding ensures that categorical data is correctly interpreted by the models, avoiding biases related to ordinal assumptions.
# 
# Increases Algorithm Efficiency: Properly scaled and encoded data can speed up learning and convergence of algorithms, particularly those using optimization techniques like gradient descent.
# 
# Improves Accuracy: Clean and well-prepared data leads to more accurate models and reduced prediction errors by minimizing noise and ensuring consistent representation across datasets.
# 
# In short, effective data preprocessing is crucial for optimizing machine learning models' performance, accuracy, and efficiency
# 

# In[ ]:




