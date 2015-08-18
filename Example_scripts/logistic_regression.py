# -*- coding: utf-8 -*-
"""
Created on Fri Jul 17 12:52:45 2015

@author: btrani
"""

import pandas as pd
import statsmodels.api as sm
import numpy as np
import matplotlib.pyplot as plt

#Open cleaned dataset

loan_data = pd.read_csv('loansData_clean.csv')

#Convert interest rate into binary (Less than 12 is 0, 12 or greater is 1)

loan_data['IR_TF'] = loan_data['Interest.Rate.Strip'].map(lambda x:\
 1 if x < 12 else 0)

#Compute constant and prep data
 
loan_data['Constant'] = 1

ind_vars = ['FICO.Score', 'Amount.Requested', 'Constant']

#Visualize the data as a scatterplot

low = loan_data['IR_TF'] == 1
high = loan_data['IR_TF'] == 0

plt.figure(figsize=(9,7))
plt.scatter(np.extract(low, loan_data['FICO.Score']),
np.extract(low, loan_data['Amount.Requested']), c='b', marker='+',\
label = 'Low Interest Rate')
plt.scatter(np.extract(high, loan_data['FICO.Score']),
np.extract(high, loan_data['Amount.Requested']), c='y', marker='o',\
label = 'High Interest Rate')
plt.xlim(620, 850)
plt.ylim(0, 36000)
plt.xlabel('FICO Score')
plt.ylabel('Amount Requested')
plt.legend()

#Define and fit the model

logit = sm.Logit(loan_data['IR_TF'], loan_data[ind_vars])
result = logit.fit()
coeff = result.params
print result.summary()

#Define a function to take score and loan amount and determine probability of
#obtaining a low-interest loan

def logistic_function(score, amount):
    p = 1/(1 + np.exp(-result.params[2] - score*result.params[0] -\
    amount*result.params[1]))
    if p >= .7:
        print "You have been approved for a low-interest loan."
    else:
        print "You have not been approved for a low-interest loan."

logistic_function(720, 10000)