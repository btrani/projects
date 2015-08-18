# -*- coding: utf-8 -*-
"""
Created on Sun Jul 19 11:53:51 2015

@author: btrani
"""

import pandas as pd
import sqlite3 as lite
import datetime
import collections
import matplotlib.pyplot as plt

#Connect to the database that houses the citi bike data
con = lite.connect('citi_bike.db')
cur = con.cursor()

#Pull data into a data frame        
df = pd.read_sql_query("SELECT * FROM available_bikes ORDER BY\
 execution_time",con,index_col='execution_time')

#Calculate activity over time for each station
hour_change = collections.defaultdict(int)
for col in df.columns:
    station_vals = df[col].tolist()
    station_id = col[1:] #trim the "_"
    station_change = 0
    for k,v in enumerate(station_vals):
        if k < len(station_vals) - 1:
            station_change += abs(station_vals[k] - station_vals[k+1])
    hour_change[int(station_id)] = station_change #convert the station id back to integer

#Find the station with the most activity
def keywithmaxval(d):
    v = list(d.values())
    k = list(d.keys())

    return k[v.index(max(v))]

max_station = keywithmaxval(hour_change)

#Join this data with the reference data to provide complete info
cur.execute("SELECT id, stationname, latitude, longitude FROM \
citibike_reference WHERE id = ?", (max_station,))
data = cur.fetchone()
print "The most active station is station id %s at %s latitude: %s \
longitude: %s " % data
print "With " + str(hour_change[max_station]) + " bicycles coming and going in the \
hour between " + datetime.datetime.fromtimestamp(int(df.index[0])).strftime\
('%Y-%m-%dT%H:%M:%S') + " and " + datetime.datetime.fromtimestamp(int(df.\
index[-1])).strftime('%Y-%m-%dT%H:%M:%S')

#Plot results as a bar chart
plt.bar(hour_change.keys(), hour_change.values())
plt.show()