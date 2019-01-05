# -*- coding: utf-8 -*-
"""
Created on Sat Jan  5 07:59:27 2019

@author: Krish
"""

import pandas as pd
import numpy as np
from pandas import Series
import matplotlib.pyplot as plt
plt.style.use(style='ggplot')
plt.rcParams['figure.figsize'] = (10, 6)

train=pd.read_csv('D:/emaildsml@gmail.com/House Price Prediction/train.csv')
test=pd.read_csv('D:/emaildsml@gmail.com/House Price Prediction/test.csv')
#Check data size
print ("Train data shape:", train.shape)
print ("Test data shape:", test.shape)
###Data Exploration
train.SalePrice.describe()
#Check skewness
'''When performing regression, sometimes it makes sense to log-transform the target variable when it is skewed. 
One reason for this is to improve the linearity of the data'''
print ("Skew is:", train.SalePrice.skew())
plt.hist(train.SalePrice, color='blue')
plt.show()
'''np.log() will transform the variable, and np.exp() will reverse the transformation.'''
target = np.log(train.SalePrice)
print ("Skew is:", target.skew())
plt.hist(target, color='blue')
plt.show()
##Working with munerical features
numeric_features = train.select_dtypes(include=[np.number])
numeric_features.dtypes
'''The DataFrame.corr() method displays the correlation (or relationship) between the columns. 
We'll examine the correlations between the features and the target.'''
corr = numeric_features.corr()
print (corr['SalePrice'].sort_values(ascending=False)[:5], '\n')
print (corr['SalePrice'].sort_values(ascending=False)[-5:])
'''The first five features are the most positively correlated with SalePrice, while the next five are the most negatively correlated.'''
#Let's dig deeper on OverallQual
train.OverallQual.unique()
#create a pivot table with target
quality_pivot = train.pivot_table(index='OverallQual',
                                  values='SalePrice', aggfunc=np.median)
quality_pivot.plot(kind='bar', color='blue')
plt.xlabel('Overall Quality')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()

#relationship between the Ground Living Area GrLivArea and SalePrice
plt.scatter(x=train['GrLivArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Above grade (ground) living area square feet')
plt.show()
'''we see that increases in living area correspond to increases in price. 
We will do the same for GarageArea'''
##Garage Area
plt.scatter(x=train['GarageArea'], y=target)
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
'''Notice that there are many homes with 0 for Garage Area, indicating that they don't have a garage.
There are a few outliers as well.'''
train = train[train['GarageArea'] < 1200]
plt.scatter(x=train['GarageArea'], y=np.log(train.SalePrice))
plt.xlim(-200,1600) # This forces the same scale as before
plt.ylabel('Sale Price')
plt.xlabel('Garage Area')
plt.show()
##NUlls
nulls = pd.DataFrame(train.isnull().sum().sort_values(ascending=False)[:25])
nulls.columns = ['Null Count']
nulls.index.name = 'Feature'
nulls

##Categorical
categoricals = train.select_dtypes(exclude=[np.number])
categoricals.columns()
#1-hot
train['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
test['enc_street'] = pd.get_dummies(train.Street, drop_first=True)
print ('Encoded: \n') 
print (train.enc_street.value_counts())
print (train.Street.value_counts())

def encode(x): return 1 if x == 'Partial' else 0
train['enc_condition'] = train.SaleCondition.apply(encode)
test['enc_condition'] = test.SaleCondition.apply(encode)

condition_pivot = train.pivot_table(index='enc_condition', values='SalePrice', aggfunc=np.median)
condition_pivot.plot(kind='bar', color='blue')
plt.xlabel('Encoded Sale Condition')
plt.ylabel('Median Sale Price')
plt.xticks(rotation=0)
plt.show()
##Null insertion
data = train.select_dtypes(include=[np.number]).interpolate().dropna()
#check
sum(data.isnull().sum() != 0)

y = np.log(train.SalePrice)
X = data.drop(['SalePrice', 'Id'], axis=1)
##Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(
                                    X, y, random_state=42, test_size=.33)

from sklearn import linear_model
lr = linear_model.LinearRegression()

model = lr.fit(X_train, y_train)
##R-Square
print ("R^2 is: \n", model.score(X_test, y_test))
'''This means that our features explain approximately 89% of the variance in our target variable'''

predictions = model.predict(X_test)
##RMSE
from sklearn.metrics import mean_squared_error
print ('RMSE is: \n', mean_squared_error(y_test, predictions))
'''The RMSE measures the distance between our predicted values and actual values.'''

#checking the magnitude of coefficients
predictors = X_train.columns
coef = Series(lr.coef_,predictors).sort_values()
coef.plot(kind='bar', title='Modal Coefficients')
'''We can see that coefficients of enc_street and OverallQual(last 2) is much higher as compared to rest of the coefficients. 
Therefore the total price of house would be more driven by these two features.'''
##Ridge - L2
from sklearn.linear_model import Ridge
for i in range (-2, 3):
    alpha = 10**i
    rm = Ridge(alpha=alpha)
    ridge_model = rm.fit(X_train, y_train)
    preds_ridge = ridge_model.predict(X_test)

    plt.scatter(preds_ridge, y_test, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Ridge Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    ridge_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_ridge))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
    
    
'''In our case, adjusting the alpha did not substantially improve our model. 
As you add more features, regularization can be helpful.'''
##Lasso (Least Absolute Shrinkage Selector Operator) - L1
from sklearn.linear_model import Lasso
for i in range (-2, 3):
    alpha = 10**i
    la = Lasso(alpha=alpha)
    lasso_model = la.fit(X_train, y_train)
    preds_lasso = lasso_model.predict(X_test)

    plt.scatter(preds_lasso, y_test, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Lasso Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    lasso_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_lasso))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
'''In our case, adjusting the alpha did not substantially improve our model. 
As you add more features, regularization can be helpful.'''

##Elasticnet
from sklearn.linear_model import ElasticNet
for i in range (-2, 3):
    alpha = 10**i
    en = ElasticNet(alpha=alpha, l1_ratio=0.5, normalize=False)
    en_model = en.fit(X_train, y_train)
    preds_en = en_model.predict(X_test)

    plt.scatter(preds_en, y_test, alpha=.75, color='b')
    plt.xlabel('Predicted Price')
    plt.ylabel('Actual Price')
    plt.title('Lasso Regularization with alpha = {}'.format(alpha))
    overlay = 'R^2 is: {}\nRMSE is: {}'.format(
                    en_model.score(X_test, y_test),
                    mean_squared_error(y_test, preds_en))
    plt.annotate(s=overlay,xy=(12.1,10.6),size='x-large')
    plt.show()
    
    '''https://www.statisticshowto.datasciencecentral.com/pearsons-coefficient-of-skewness/

Covariance
https://www.investopedia.com/terms/c/correlationcoefficient.asp'''
