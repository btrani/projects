# -*- coding: utf-8 -*-
"""
Created on Wed Jul 15 18:31:17 2015

@author: btrani
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

#Read data in from public source

loan_data = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/\
loansData.csv')

#Prep data for analysis

loan_data['Interest.Rate.Strip'] = map(lambda x: float(x.rstrip('%')),\
loan_data['Interest.Rate'])

loan_data['Loan.Length.Strip'] = map(lambda x: float(x.rstrip('months')),\
loan_data['Loan.Length'])

loan_data['FICO.Score'] = map(lambda x: int(str(x[0:3])),\
loan_data['FICO.Range'])

#Verify cleaning

loan_data['FICO.Range'].head()
loan_data['Interest.Rate.Strip'].head()
loan_data['Loan.Length.Strip'].head()

#Investigate potential relationships

a = pd.scatter_matrix(loan_data, alpha=.05, figsize=(10,10), diagonal='hist')

#Create linear model

intrate = loan_data['Interest.Rate.Strip']
loanamt = loan_data['Amount.Requested']
fico = loan_data['FICO.Score']

y = np.matrix(intrate).transpose()
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()
x = np.column_stack([x1, x2])

X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
print model.summary()

#Plot Interest Rate and FICO Score with 2 linear regression lines

plt.figure()
plt.plot(x1, y, '.')
plt.plot(x1, (x1*model.params[1])+model.params[0]+(10000*model.params[2]), '-'\
,label='$10000 Requested')
plt.plot(x1, (x1*model.params[1])+model.params[0]+(30000*model.params[2]), '-'\
,label='$30000 Requested')
plt.xlim(x1.min()-10, x1.max()+10)
plt.ylim(0, y.max()+2)
plt.title("Linear Regression Example")
plt.ylabel("Interest Rate, %")
plt.xlabel("FICO Score")
plt.legend(loc='best')
plt.show()

loan_data.to_csv('/Users/btrani/Documents/Thinkful/projects/loansData_clean.\
csv', header=True, index=False)