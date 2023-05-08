# Ex-07-Feature-Selection
## AIM
To Perform the various feature selection techniques on a dataset and save the data to a file. 

# Explanation
Feature selection is to find the best set of features that allows one to build useful models.
Selecting the best features helps the model to perform well. 

# ALGORITHM
### STEP 1
Read the given Data
### STEP 2
Clean the Data Set using Data Cleaning Process
### STEP 3
Apply Feature selection techniques to all the features of the data set
### STEP 4
Save the data to the file


# CODE
```
import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

df=pd.read_csv('/content/titanic_dataset.csv')

df.head()

df.isnull().sum()

df.drop('Cabin',axis=1,inplace=True)

df.drop('Name',axis=1,inplace=True)

df.drop('Ticket',axis=1,inplace=True)

df.drop('PassengerId',axis=1,inplace=True)

df.drop('Parch',axis=1,inplace=True)

df

df['Age']=df['Age'].fillna(df['Age'].median())

df['Embarked']=df['Embarked'].fillna(df['Embarked'].mode()[0])

df.isnull().sum()

plt.title("Dataset with outliers")

df.boxplot()

plt.show()

cols = ['Age','SibSp','Fare']

Q1 = df[cols].quantile(0.25)

Q3 = df[cols].quantile(0.75)

IQR = Q3 - Q1

df = df[~((df[cols] < (Q1 - 1.5 * IQR)) |(df[cols] > (Q3 + 1.5 * IQR))).any(axis=1)]

plt.title("Dataset after removing outliers")

df.boxplot()

plt.show()

from sklearn.preprocessing import OrdinalEncoder

climate = ['C','S','Q']

en= OrdinalEncoder(categories = [climate])

df['Embarked']=en.fit_transform(df[["Embarked"]])

df

climate = ['male','female']

en= OrdinalEncoder(categories = [climate])

df['Sex']=en.fit_transform(df[["Sex"]])

df

from sklearn.preprocessing import RobustScaler

sc=RobustScaler()

df=pd.DataFrame(sc.fit_transform(df),columns=['Survived','Pclass','Sex','Age','SibSp','Fare','Embarked'])

df

import statsmodels.api as sm

import numpy as np

import scipy.stats as stats

from sklearn.preprocessing import QuantileTransformer

qt=QuantileTransformer(output_distribution='normal',n_quantiles=692)

df1=pd.DataFrame()

df1["Survived"]=np.sqrt(df["Survived"])

df1["Pclass"],parameters=stats.yeojohnson(df["Pclass"])

df1["Sex"]=np.sqrt(df["Sex"])

df1["Age"]=df["Age"]

df1["SibSp"],parameters=stats.yeojohnson(df["SibSp"])

df1["Fare"],parameters=stats.yeojohnson(df["Fare"])

df1["Embarked"]=df["Embarked"]

df1.skew()

import matplotlib

import seaborn as sns

import statsmodels.api as sm

%matplotlib inline

from sklearn.model_selection import train_test_split

from sklearn.linear_model import LinearRegression

from sklearn.feature_selection import RFE

from sklearn.linear_model import RidgeCV, LassoCV, Ridge, Lasso

X = df1.drop("Survived",1)

y = df1["Survived"]

plt.figure(figsize=(12,10))

cor = df1.corr()

sns.heatmap(cor, annot=True, cmap=plt.cm.RdPu)

plt.show()

cor_target = abs(cor["Survived"])

relevant_features = cor_target[cor_target>0.5]

relevant_features

X_1 = sm.add_constant(X)

model = sm.OLS(y,X_1).fit()

model.pvalues

cols = list(X.columns)

pmax = 1

while (len(cols)>0):

p= []

X_1 = X[cols]

X_1 = sm.add_constant(X_1)

model = sm.OLS(y,X_1).fit()

p = pd.Series(model.pvalues.values[1:],index = cols)  

pmax = max(p)

feature_with_p_max = p.idxmax()

if(pmax>0.05):

    cols.remove(feature_with_p_max)
    
else:

    break
    selected_features_BE = cols

print(selected_features_BE)

model = LinearRegression()

rfe = RFE(model,step= 4)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

print(rfe.support_)

print(rfe.ranking_)

nof_list=np.arange(1,6)

high_score=0

nof=0

score_list =[]

for n in range(len(nof_list)):

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size = 0.3, random_state = 0)

model = LinearRegression()

rfe = RFE(model,step=nof_list[n])

X_train_rfe = rfe.fit_transform(X_train,y_train)

X_test_rfe = rfe.transform(X_test)

model.fit(X_train_rfe,y_train)

score = model.score(X_test_rfe,y_test)

score_list.append(score)

if(score>high_score):

    high_score = score
    
    nof = nof_list[n]
print("Optimum number of features: %d" %nof)

print("Score with %d features: %f" % (nof, high_score))

cols = list(X.columns)

model = LinearRegression()

rfe = RFE(model, step=2)

X_rfe = rfe.fit_transform(X,y)

model.fit(X_rfe,y)

temp = pd.Series(rfe.support_,index = cols)

selected_features_rfe = temp[temp==True].index

print(selected_features_rfe)

reg = LassoCV()

reg.fit(X, y)

print("Best alpha using built-in LassoCV: %f" % reg.alpha_)

print("Best score using built-in LassoCV: %f" %reg.score(X,y))

coef = pd.Series(reg.coef_, index = X.columns)

print("Lasso picked " + str(sum(coef != 0)) + " variables and eliminated the other " + str(sum(coef == 0)) + " variables")

imp_coef = coef.sort_values()

import matplotlib

matplotlib.rcParams['figure.figsize'] = (8.0, 10.0)

imp_coef.plot(kind = "barh")

plt.title("Feature importance using Lasso Model")

plt.show()
```

# OUPUT
![Screenshot 2023-05-08 101733](https://user-images.githubusercontent.com/121300272/236747991-cf6ec445-957a-497b-896a-5cc927f1d17c.png)

![Screenshot 2023-05-08 101746](https://user-images.githubusercontent.com/121300272/236747995-615abde5-4a40-4d46-bc5d-414edfe9aa35.png)

![Screenshot 2023-05-08 101816](https://user-images.githubusercontent.com/121300272/236748007-34f5f879-2f99-4a64-9eaf-877505c8c6d5.png)

![Screenshot 2023-05-08 101903](https://user-images.githubusercontent.com/121300272/236748012-7021513c-06a2-4fd4-b9cf-a20a4e18f5cc.png)

![Screenshot 2023-05-08 101931](https://user-images.githubusercontent.com/121300272/236748032-a02e3728-72a3-463e-ad0f-7a6ac8ea77b8.png)

![Screenshot 2023-05-08 102015](https://user-images.githubusercontent.com/121300272/236748040-a3842a93-4a7c-4f1b-9596-adb4a9207877.png)


![Screenshot 2023-05-08 102037](https://user-images.githubusercontent.com/121300272/236748042-e7b3774e-ed16-4d31-ab1d-045d4f603b17.png)

![Screenshot 2023-05-08 102109](https://user-images.githubusercontent.com/121300272/236748049-de3f4ddd-f0f7-49b4-8bfa-2f5da8b210ba.png)

![Screenshot 2023-05-08 102141](https://user-images.githubusercontent.com/121300272/236748063-41455b34-a36f-4327-9f8c-2f49303ca56f.png)

![Screenshot 2023-05-08 102225](https://user-images.githubusercontent.com/121300272/236748071-4306f5e0-be3c-4ac6-882e-05c0df1d1d4a.png)

![Screenshot 2023-05-08 102250](https://user-images.githubusercontent.com/121300272/236748077-b163ad6c-c965-46f0-ba03-1a89b611f65a.png)

![Screenshot 2023-05-08 102324](https://user-images.githubusercontent.com/121300272/236748086-572c7ca2-dc28-4024-add1-21fbb49fb384.png)

![Screenshot 2023-05-08 102354](https://user-images.githubusercontent.com/121300272/236748093-740000d9-741e-4cb8-9b2a-c9693e340393.png)

![Screenshot 2023-05-08 102429](https://user-images.githubusercontent.com/121300272/236748105-d6c278ac-7044-4fe9-b830-cbb09c46444d.png)


# RESULT

The various feature selection techniques are performed on a dataset and saved the data to a file.
