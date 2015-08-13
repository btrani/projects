# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:18:49 2015

@author: btrani
First script for the Kaggle Liberty Mutual competition
"""

import pandas as pd
import numpy as np

df_train = pd.read_csv('/Users/btrani/Git/Data/LM/train.csv', index_col = 0)
df_test = pd.read_csv('/Users/btrani/Git/Data/LM/test.csv', index_col = 0)

labels = df_train['Hazard']
df_train.drop('Hazard', axis=1, inplace=True)
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
model = RandomForestRegressor(n_estimators = 500, max_depth = 20, \
min_samples_leaf = 4)
model = model.fit(train, labels)
predict = model.predict(test)

print sorted(zip(map(lambda x: round(x, 4), model.feature_importances_), columns), 
             reverse=True)
print model.score(train, labels)

submission = pd.DataFrame({"Id": test_ind, "Hazard": predict})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/projects/Kaggle/Liberty_Mutual/sub_6.csv')