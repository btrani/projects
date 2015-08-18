# -*- coding: utf-8 -*-
"""
Created on Mon Aug 17 15:41:25 2015

@author: btrani
"""

import pandas as pd
import numpy as np

train = pd.read_csv('/Users/btrani/Git/projects/Kaggle/Springleaf/train.csv').set_index('ID')
test = pd.read_csv('/Users/btrani/Git/projects/Kaggle/Springleaf/test.csv').set_index('ID')


nunique = pd.Series([train[col].nunique() for col in train.columns], index = train.columns)
constants = nunique[nunique<2].index.tolist()

train = train.drop(constants, axis=1)
test = test.drop(constants, axis=1)

labels = train['target']
train.drop('target', axis=1, inplace=True)

from sklearn import preprocessing
obs = train.dtypes == 'object'
obs = obs[obs].index.tolist()
vals = {}
for col in obs:
    vals[col] = preprocessing.LabelEncoder()
    train[col] = vals[col].fit_transform(train[col])
    try:
        test[col] = vals[col].transform(test[col])
    except:
        del test[col]
        del train[col]
        
train = train.fillna(-1)
test = test.fillna(-1)

test_ind = test.index

#Create XGB model   
import xgboost as xgb

offset = 50000
num_round = 100
xgtest = xgb.DMatrix(test)
gb_params = {'max_depth':6, 'eta':1, 'silent':1, \
'objective':'binary:logistic', 'eval_metric': 'auc' }

#Create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset], label=labels[:offset])

#Train model and predict test values
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(gb_params, xgtrain, num_round, watchlist, \
early_stopping_rounds=4)
preds1 = model.predict(xgtest)

#Send predicted scores to csv file
submission = pd.DataFrame({"Id": test_ind, "target": preds1})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/projects/Kaggle/Springleaf/sub_gb_2.csv')
