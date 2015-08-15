# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:18:49 2015

@author: btrani
First script for the Kaggle Liberty Mutual competition
"""

import pandas as pd
import numpy as np

#Import train and test data provided by LM
df_train = pd.read_csv('/Users/btrani/Git/Data/LM/train.csv', index_col = 0)
df_test = pd.read_csv('/Users/btrani/Git/Data/LM/test.csv', index_col = 0)

#Subset out the lables to train the model
labels = df_train['Hazard']
df_train.drop('Hazard', axis=1, inplace=True)

#Drop least important variables
#df_train.drop('T2_V12', axis=1, inplace=True)
#df_train.drop('T2_V8', axis=1, inplace=True)
#df_train.drop('T1_V17', axis=1, inplace=True)

#df_test.drop('T2_V12', axis=1, inplace=True)
#df_test.drop('T2_V8', axis=1, inplace=True)
#df_test.drop('T1_V17', axis=1, inplace=True)

columns = df_train.columns
test_ind = df_test.index

train = np.array(df_train)
test = np.array(df_test)

#Convert categorical variables
from sklearn import preprocessing
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])
    
train = train.astype(float)
test = test.astype(float)

#Model #1 RandomForestRegressor tuned using GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.grid_search import GridSearchCV

#Split data into train and test
#X_train, X_test, y_train, y_test = train_test_split(train, labels, \
#test_size=0.2, random_state=0)

#Determine parameters for grid search
param_grid = {'n_estimators': [1000], 'max_depth': [10,20], \
'min_samples_leaf': [5], 'n_jobs' : [-1]}

#Use KFold cross validation
cv = cross_validation.KFold(len(train), n_folds = 10)

#Train and fit RF model
model = GridSearchCV(RandomForestRegressor(), param_grid, cv=cv)
model = model.fit(train, labels)

#Use trained model to predict test values
predict = model.predict(test)

#Send predicted scores to csv file
submission = pd.DataFrame({"Id": test_ind, "Hazard": predict})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/projects/Kaggle/Liberty_Mutual/sub_8.csv')

#Model #2 Extreme Gradient Boost using xgboost
import xgboost as xgb

#Subset the data and set up model parameters
offset = 10000
num_round = 1000
xgtest = xgb.DMatrix(test)
gb_params = {"objective":"reg:linear", "eta": 0.005, "min_child_weight": 6, \
"subsample": 0.7, "colsample_bytree": 0.7, "scale_pos_weight": 1, "silent": 1, 
"max_depth": 12}

#Create a train and validation dmatrices 
xgtrain = xgb.DMatrix(train[offset:,:], label=labels[offset:])
xgval = xgb.DMatrix(train[:offset,:], label=labels[:offset])

#Train model and predict test values
watchlist = [(xgtrain, 'train'),(xgval, 'val')]
model = xgb.train(gb_params, xgtrain, num_round, watchlist, \
early_stopping_rounds=4)
preds1 = model.predict(xgtest)

#Send predicted scores to csv file
submission = pd.DataFrame({"Id": test_ind, "Hazard": preds1})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/projects/Kaggle/Liberty_Mutual/sub_gb_3.csv')

