# -*- coding: utf-8 -*-
"""
Created on Fri Jul 24 14:39:15 2015

@author: btrani
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import math
from sklearn.cross_validation import train_test_split
from sklearn.neighbors import KNeighborsClassifier

#Read in sales data from file
df = pd.read_excel('rollingsales_manhattan.xls', header=4)

#Clean data to remove missing values and outliers
df = df[df['SALE PRICE'] > 0]
df = df[df['GROSS SQUARE FEET'] > 0]
df = df[df['GROSS SQUARE FEET'] < 500000]
df = df[df['TOTAL UNITS'] > 0]
df = df[df['TOTAL UNITS'] < 400]

#Calculate average sale price by zip code as proxy for zip code
avg_by_zip = df.groupby(['ZIP CODE'])['SALE PRICE'].median().reset_index()
avg_by_zip.columns = ['ZIP CODE', 'avg_sale_by_zip']
df = pd.merge(df, avg_by_zip, on='ZIP CODE', how='outer')

#Transform sale price using log normal function to normalize data
def log(x):
    return math.log(x)

df['log_sale'] = df['SALE PRICE'].apply(log)

#Investigate potential relationships via scatter matrix
a = pd.scatter_matrix(df, figsize = (10,10), diagonal='hist')

#Prep independent and dependent variables for regression
y = np.matrix(df['log_sale']).transpose()
x1 = np.matrix(df['GROSS SQUARE FEET']).transpose()
x2 = np.matrix(df['TOTAL UNITS']).transpose()
x3 = np.matrix(df['avg_sale_by_zip']).transpose()
x = np.column_stack([x1, x2, x3])

#Fit the OLS model
X = sm.add_constant(x)
model = sm.OLS(y, X)
fitted = model.fit()
print fitted.summary()

#Visualize the data
plt.figure()
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(fitted, 1, ax=ax)
plt.show()

plt.figure()
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(fitted, 2, ax=ax)
plt.show()

plt.figure()
fig, ax = plt.subplots()
fig = sm.graphics.plot_fit(fitted, 3, ax=ax)
plt.show()

plt.figure()
plt.plot(x1, ((x1*fitted.params[1]) + (x2*fitted.params[2]) + \
(x3* fitted.params[3]) + fitted.params[0]), '.')
plt.show()

#Use k-NN to predict neighborhood of next sale
#Create new dataframe with just the variables of interest
df2 = df[['TOTAL UNITS', 'avg_sale_by_zip', 'NEIGHBORHOOD']]

#Split dataframe into training (80%) and test(20%) sets
dfTrain, dfTest = train_test_split(df2, test_size=0.2)

#Test values of k for 1-50 to minimize error but avoid overfitting
results = []
for k in range(1,50):
    model_knn = KNeighborsClassifier(n_neighbors=k)
    model_knn.fit(dfTrain[:,:2], dfTrain[:,2])
    expected = dfTest[:,2]
    predicted = model_knn.predict(dfTest[:,:2])
    error_rate = (predicted != expected).mean()
    print('%d:, %.2f' % (k, error_rate))
    results.append([k, error_rate])

#Visualize error rates to help choose k
results = pd.DataFrame(results, columns=['k', 'Error Rate'])

plt.plot(results['k'], results['Error Rate'])
plt.title("Error Rate with Varying K Values")
plt.xlabel("k-values")
plt.ylabel("Error Rate")