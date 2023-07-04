import numpy as np
import pandas as pd
import sklearn
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

df = pd.read_csv("churn.csv")
print(df)
df

# to show the row and column
df.shape

#to see all column names
df.columns.values

#to check for NA or missing value
dt=df.isna().sum()
print(dt)

#to show some statistics
ds=df.describe()
print(ds)

#get customer churn count
dc=df['Churn'].value_counts()
print(dc)

#visualize the count of customer churn
#convert the churn column to categorical data type
df['Churn'] = df['Churn'].astype('category')
#plot the countplot
sns.countplot(data=df, x='Churn')

#display the plot
plt.show()

#to see the percentage of customers that are leaving
numRetained = df[df.Churn == 'No'].shape[0]
numChurned = df[df.Churn == 'Yes'].shape[0]

# print the percentage of customers that stayed
print('\n% of customers stayed in the company',numRetained/(numRetained + numChurned) * 100)
# print the percentage of customers that left
print( '\n% of customers left with the company', numChurned/(numRetained + numChurned) * 100,"\n")


#Visual the churn count for both males and females
sns.countplot(x ='gender', hue='Churn', data=df)

#display the plot
plt.show()

#visualize the churn count for the internet service 
sns.countplot(x='InternetService', hue='Churn', data=df)

#display the plot
plt.show()

#to visualize numeric data
numericFeatures = ['tenure', 'MonthlyCharges']
fig, ax = plt.subplots(1,2, figsize=(28, 8))
df[df.Churn == "No"][numericFeatures].hist(bins=20, color='blue', alpha=0.5, ax=ax)
df[df.Churn == "Yes"][numericFeatures].hist(bins=20, color='orange', alpha=0.5, ax=ax)

#display the plot
plt.show()

#to remove unnecessary column
cleanDF = df.drop('customerID', axis=1)


#convert all the non-numerical column to numeric 

for column in cleanDF.columns:
  numeric_columns = cleanDF.select_dtypes(include=[np.number]).columns
cleanDF[numeric_columns] = cleanDF[numeric_columns].astype(float)

#to show datatypes
cleanDF.dtypes