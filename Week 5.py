#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd 
df = pd.DataFrame({'Date':['10/2/2011', '11/2/2011', '12/2/2011', '13/2/2011'], 
                   'Event':['Music', 'Poetry', 'Theatre', 'Comedy'], 
                   'Cost':[10000, 5000, 15000, 2000]}) 
print(df)


# In[2]:


df['Discounted_Price'] = df.apply(lambda row: row.Cost * 0.9, axis = 1)
print(df)


# In[3]:


df = pd.DataFrame({'Name':['John','Ted','Dove','Brad','Rex'], 
                   'Salary':[44000, 35000, 75000, 20000,6000]}) 
print(df)


# In[4]:


def salary_stats(value):
    if value < 10000: 
        return "very low" 
    elif 10000 <= value < 25000: 
        return "low" 
    elif 25000 <= value < 40000: 
        return "average" 
    elif 40000 <= value < 50000: 
        return "better" 
    elif value >= 50000: 
        return "very good" 
    
df['salary_stats'] = df['Salary'].map(salary_stats) 
df


# In[9]:


import pandas as pd
data = pd.DataFrame({
 'Name' : ['A', 'B', 'C', 'D','E', 'F'], 
'Education' : ['High School', 'Masters', 'Doctorate', 'Bachelors','Masters', 'High School']})
data


# # Binary Encoding

# In[10]:


education_data = pd.get_dummies(data.Education)
print(education_data)


# # Ranking Transformation

# In[11]:


education_map = {
 'High School' : 1,
 'Bachelors' : 2,
 'Masters': 3,
 'Doctorate': 4
 }
education_data = data['Education'].map(education_map)
data['Education'] = education_data
data


# In[12]:


education_map = {
    'High School' : 12,
    'Bachelors' : 16,
    'Masters': 18,
    'Doctorate': 21
 }
education_data = data['Education'].map(education_map)
data['Education'] = education_data
data


# # Adding data objects- rows

# In[13]:


df.loc[len(df.index)]=['Hruthvik', 15000, 'low']
df


# # Combining two data frames

# In[15]:


import pandas as pd
d1 = {'Name': ['Pankaj', 'Meghna', 'Lisa'], 'Country': ['India', 'India', 'USA']}
df1 = pd.DataFrame(d1)
print('DataFrame 1:\n', df1,'\n')
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Pankaj', 'Anupam', 'Amit']})
print('DataFrame 2:\n', df2,'\n')
df3 = pd.DataFrame({'Name': ['Priya'], 'Country': ['India'], 'Role': ['COO']})
print('DataFrame 3:\n', df3,'\n')


# In[16]:


same_cols_df = pd.concat([df1,df3],ignore_index=True)
same_cols_df


# In[17]:


a_df=df1.append(df2, ignore_index=True)
a_df


# In[18]:


c_df = pd.concat([df1,df2],ignore_index=True)
c_df


# # Defaut Mergeing - inner join

# In[19]:


df_merged = df1.merge(df2)
print('Result:\n', df_merged)


# # Mergeing DataFrames with left, Right and outer join

# In[20]:


print('Result Left Join:\n', df1.merge(df2, how='left'))
print('Result Right Join:\n', df1.merge(df2, how='right'))
print('Result Outer Join:\n', df1.merge(df2, how='outer'))


# # Mergeing dataframes with specific columns

# In[30]:


import pandas as pd

dict1 = {
    'ID': [1, 2, 3],
    'Name': ['Pankaj', 'Meghna', 'Lisa'],
    'Country': ['India', 'India', 'India'],
    'Role': ['CEO', 'CTO', 'CTO']
}

df1 = pd.DataFrame(dict1)
df2 = pd.DataFrame({'ID': [1, 2, 3], 'Name': ['Pankaj', 'Anupam', 'Amit']})

print(df1.merge(df2, on='ID'))
print('\n', df1.merge(df2, on='Name'))


# # Titanic CSV

# In[41]:


import pandas as pd
import missingno as msno
get_ipython().run_line_magic('matplotlib', 'inline')


# In[34]:


titanic_df = pd.read_csv("titanic.csv")
titanic_df 


# In[35]:


titanic_df.info()


# In[36]:


titanic_df.isnull()


# In[42]:


msno.bar(titanic_df)


# In[43]:


msno.matrix(titanic_df)


# In[44]:


titanic_df.isnull().sum()


# In[45]:


df = titanic_df.dropna(axis=0)
df.isnull().sum()


# In[46]:


df.info()


# In[47]:


titanic_df.columns


# In[50]:


df = titanic_df.drop(['Pclass'],axis=1)
df.isnull().sum()


# In[51]:


titanic_df['Pclass'].unique()


# In[53]:


titanic_df['Pclass'] = titanic_df['Pclass'].fillna('C')


# In[54]:


titanic_df['Pclass'].isnull().sum()


# In[55]:


titanic_df


# In[57]:


mean = titanic_df['Age'].mean()
print(mean)
#Replace the missing values for numerical columns with mean
titanic_df['Age'] = titanic_df['Age'].fillna(mean)
titanic_df['Age'] 


# In[58]:


titanic_df = pd.read_csv("titanic.csv")
 #Replace the missing values for categorical columns with mode
mode = titanic_df['Pclass'].mode()[0]
print(mode)
titanic_df['Pclass'] = titanic_df['Pclass'].fillna(mode)


# In[59]:


titanic_df['Pclass']


# In[61]:


titanic_df['Age']= titanic_df['Age'].fillna(titanic_df['Age'].median())
titanic_df['Age']


# In[62]:


titanic_df = pd.read_csv("titanic.csv")
titanic_df


# In[63]:


new_df = titanic_df.fillna(method="ffill")
new_df


# In[64]:


new_df = titanic_df.fillna(method="ffill",limit=1)
new_df


# In[65]:


new_df = titanic_df.fillna(method="bfill")
new_df


# #  Numerosity Data Reduction

# In[66]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# In[67]:


customer_df = pd.read_csv('customer_churn.csv')
print(customer_df.shape)
print(customer_df.Churn.value_counts())


# In[69]:


import pandas as pd

# Example DataFrame creation (replace with your actual DataFrame)
customer_df = pd.DataFrame({
    'Churn': [0, 1] * 500,  # Example data
    'Feature1': range(1000),  # Example feature
    'Feature2': range(1000)   # Example feature
})

# Ensure the sample size is appropriate
sample_size = min(1000, customer_df.shape[0])

# Sample the DataFrame
customer_df_rs = customer_df.sample(sample_size, random_state=1)

# Separate features and target
y = customer_df_rs['Churn']
Xs = customer_df_rs.drop(columns=['Churn'])

print(customer_df_rs.shape)


# In[70]:


customer_df_rs


# In[71]:


print(customer_df_rs.Churn.value_counts())


# # Stratified Sampling

# In[72]:


n,s=len(customer_df),1000
print(n,s)
r = s/n
print('Ratio of each Churn class in sample:',r)
sample_df = customer_df.groupby('Churn').apply(lambda sdf: sdf.sample(round(len(sdf))))
print(sample_df.Churn.value_counts())


# In[73]:


customer_df.Churn.value_counts().plot.bar()


# In[74]:


sample_df.Churn.value_counts().plot.bar()


# # Random Over/Under sampling

# In[75]:


n,s=len(customer_df),500
sample_df = customer_df.groupby('Churn').apply(lambda sdf: sdf.sample(250))
print(sample_df.Churn.value_counts())


# In[76]:


sample_df.Churn.value_counts().plot.bar()


# In[77]:


sample_df


# # Outliners Detection

# In[78]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm


# In[79]:


titanic_df = pd.read_csv("titanic.csv")
titanic_df


# # Scatter plot to detect outliners

# In[81]:


fig,ax = plt.subplots(figsize=(10,4))
ax.scatter(titanic_df['Age'],titanic_df['Fare'])
ax.set_xlabel('Age')
ax.set_ylabel('Fare')
plt.title("Scatter plot")
plt.show()


# # Box plot to detect outliners

# In[83]:


titanic_df['Age'].plot(kind='box')


# In[85]:


q1 = titanic_df["Age"].quantile(0.25)
# finding the 3rd quartile
q3 = titanic_df['Age'].quantile(0.75)
# finding the iqr region
iqr = q3-q1
# finding upper and lower whiskers
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)


# In[86]:


age_arr = titanic_df["Age"]
outliers = age_arr[(age_arr <= lower_bound) | (age_arr >= upper_bound)]
print('The following are the outliers in the boxplot of age:\n',outliers)


# # Histogram to detect outliners

# In[88]:


titanic_df['Fare'].plot(kind='hist')


# # Remove Data Objects with outliners

# In[90]:


upperIndex = titanic_df[titanic_df['Age']>upper_bound].index
titanic_df.drop(upperIndex,inplace=True)
lowerIndex = titanic_df[titanic_df['Age']<lower_bound].index
titanic_df.drop(lowerIndex,inplace=True)
titanic_df.info()


# # Replcing Outliners with upper/lower caps

# In[91]:


titanic_df = pd.read_csv("titanic.csv")


# In[93]:


fare_arr = titanic_df["Fare"]
upper_cap = np.percentile(fare_arr,1)
lower_cap = np.percentile(fare_arr,99)
outliers = fare_arr[(fare_arr < upper_cap) | (fare_arr > lower_cap)]
print('The following are the outliers in the boxplot of fare:\n',outliers)


# In[96]:


for i in titanic_df['Fare']:
    if i<lower_bound :
        titanic_df['Fare'] = titanic_df['Fare'].replace(i,lower_cap)
    elif i>upper_bound :
        titanic_df['Fare'] = titanic_df['Fare'].replace(i,upper_cap)


# In[97]:


titanic_df.info()


# # replacing outliners with mean

# In[98]:


titanic_df = pd.read_csv("titanic.csv")


# In[101]:


m = np.mean(titanic_df['Age'])
print('mean:',m)
for i in titanic_df['Age']:
    if i<lower_bound or i>upper_bound :
        titanic_df['Age'] = titanic_df['Age'].replace(i,m)


# # Replacing Outliners with median

# In[2]:


import pandas as pd
titanic_df = pd.read_csv("titanic.csv")


# In[5]:


q1 = titanic_df["Age"].quantile(0.25)
 # finding the 3rd quartile
q3 = titanic_df['Age'].quantile(0.75)
 # finding the iqr region
iqr = q3-q1
 # finding upper and lower whiskers
upper_bound = q3+(1.5*iqr)
lower_bound = q1-(1.5*iqr)


# In[8]:


m = titanic_df['Age'].median()
print(m)
for i in titanic_df['Age']:
    if i<lower_bound or i>upper_bound :
        titanic_df['Age'] = titanic_df['Age'].replace(i,m)


# In[ ]:




