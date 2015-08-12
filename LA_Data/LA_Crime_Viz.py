# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:51:32 2015

@author: btrani
Short script to visualize 2014 LA city crime data
Can be be further broken out by crime type, time of day, etc.
Inspired by Daniel Forsyth's blog post about NYC taxi trips found here:
http://www.danielforsyth.me/mapping-nyc-taxi-data/
"""

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rcParams

df = pd.read_csv('/Users/btrani/Git/LA_Data/LAPD_Crime_and_Collision_Raw_Data_-_2014.csv')

df = df.dropna(subset = ['Location 1'])

df[['lat','long']] = df['Location 1'].apply(eval).apply(pd.Series)

pd.options.display.mpl_style = 'default' #Better Styling  
new_style = {'grid': False} #Remove grid  
matplotlib.rc('axes', **new_style)
rcParams['figure.dpi'] = 500
rcParams['figure.figsize'] = (20, 20)

P=df.plot(kind='scatter', x='long', y='lat',color='white',
xlim=(-118.7, -118.1), ylim=(33.61, 34.4), s=.02,alpha=.6)
P.set_axis_bgcolor('black') #Background Color
P.axes.get_xaxis().set_visible(False)
P.axes.get_yaxis().set_visible(False)
plt.savefig('/Users/btrani/Git/LA_Data/LA_Crime.jpeg', bbox_inches='tight')