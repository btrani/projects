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

#Drop least variables
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

from sklearn import preprocessing
for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])
    
train = train.astype(float)
test = test.astype(float)

from sklearn.ensemble import RandomForestRegressor
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.grid_search import GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(train, labels, \
test_size=0.2, random_state=0)
param_grid = {'n_estimators': [1000], 'max_depth': [10,20], \
'min_samples_leaf': [5], 'n_jobs' : [-1]}
cv = cross_validation.KFold(len(X_train), n_folds = 10)
model = GridSearchCV(RandomForestRegressor(), param_grid, cv=cv)
model = model.fit(X_train, y_train)

print model.best_params_

model.score(X_test, y_test)

predict = model.predict(test)

#print sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), columns), 
#             reverse=True)
#print model.score(train, labels)

submission = pd.DataFrame({"Id": test_ind, "Hazard": predict})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/projects/Kaggle/Liberty_Mutual/sub_8.csv')

#from sklearn.ensemble import GradientBoostingRegressor
#gb_model = GradientBoostingRegressor(n_estimators = 500, max_depth = 20, \
#min_samples_leaf = 3)
#gb_model = gb_model.fit(X_train, y_train)

#gb_model.score(X_test, y_test)
#gb_predict = model.predict(test)

#submission = pd.DataFrame({"Id": test_ind, "Hazard": gb_predict})
#submission = submission.set_index("Id")
#submission.to_csv('/Users/btrani/Git/projects/Kaggle/Liberty_Mutual/sub_gb_2.csv')
