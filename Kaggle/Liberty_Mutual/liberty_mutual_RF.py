# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 13:18:49 2015

@author: btrani
First script for the Kaggle Liberty Mutual competition
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.ensemble import RandomForestRegressor


df_train = pd.read_csv('/Users/btrani/Git/Kaggle/Liberty_Mutual/train.csv', index_col = 0)
df_test = pd.read_csv('/Users/btrani/Git/Kaggle/Liberty_Mutual/test.csv', index_col = 0)

labels = df_train['Hazard']
df_train.drop('Hazard', axis=1, inplace=True)

columns = df_train.columns
test_ind = df_test.index

train = np.array(df_train)
test = np.array(df_test)

for i in range(train.shape[1]):
    lbl = preprocessing.LabelEncoder()
    lbl.fit(list(train[:,i]) + list(test[:,i]))
    train[:,i] = lbl.transform(train[:,i])
    test[:,i] = lbl.transform(test[:,i])
    
train = train.astype(float)
test = test.astype(float)

model = RandomForestRegressor(n_estimators = 500, max_depth=10)

model.fit(train, labels)

prediction = model.predict(test)
submission = pd.DataFrame({"Id": test_ind, "Hazard": prediction})
submission = submission.set_index("Id")
submission.to_csv('/Users/btrani/Git/Kaggle/Liberty_Mutual/sub_1.csv')