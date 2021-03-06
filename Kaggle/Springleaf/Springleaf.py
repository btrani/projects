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
train.drop('VAR_0212', axis=1, inplace=True)
train.drop('VAR_0227', axis=1, inplace=True)
train.drop('VAR_0254', axis=1, inplace=True)
train.drop('VAR_0551', axis=1, inplace=True)
train.drop('VAR_0811', axis=1, inplace=True)

test.drop('VAR_0811', axis=1, inplace=True)
test.drop('VAR_0551', axis=1, inplace=True)
test.drop('VAR_0254', axis=1, inplace=True)
test.drop('VAR_0227', axis=1, inplace=True)
test.drop('VAR_0212', axis=1, inplace=True)

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

#Model 1 RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.cross_validation import train_test_split

#Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(train, labels, \
test_size=0.2, random_state=0)

#Train and fit RF model
rf_model = RandomForestClassifier(n_estimators = 1000, max_depth = 12, n_jobs = -1)
rf_model.fit(X_train, y_train)
score = rf_model.score(X_test, y_test)

#Use trained model to predict test values
preds_rf = rf_model.predict_proba(test)

preds_rf = preds_rf[:,1]

#Create XGB model   
import xgboost as xgb

xgtest = xgb.DMatrix(test)

#Create a train and validation dmatrices
offset = 30000 
xgtrain = xgb.DMatrix(train[offset:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset], label=labels[:offset])

#Train model and predict test values

num_round = 200
gb_params = {'max_depth':16, 'eta':.1, 'silent':1, \
'objective':'binary:logistic', 'eval_metric': 'auc'}
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(gb_params, xgtrain, num_round, watchlist, \
early_stopping_rounds=4)
preds1 = model.predict(xgtest, ntree_limit=model.best_iteration)

fscore = [ (v,k) for k,v in model.get_fscore().iteritems() ]
fscore.sort(reverse=True)

model.dump_model('dump.raw.txt')

preds_comb = preds_rf *.5 + preds1 * .5 

#Send predicted scores to csv file
submission = pd.DataFrame({"Id": test_ind, "target": preds_comb})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/projects/Kaggle/Springleaf/sub_comb_1.csv')