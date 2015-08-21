# -*- coding: utf-8 -*-
"""
Created on Thu Aug 20 14:09:50 2015

@author: btrani
"""

import numpy as np
import pandas as pd
import datetime
import urllib
 
# Read in our data. We've aggregated it by date already, so we don't need to worry about paging
query = ('https://data.smgov.net/resource/xx64-wi4x.json?$$app_token=g5iIFV3PzVEgEGqkxekFlTlxW')
raw_data = pd.read_json(query)