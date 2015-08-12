# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 17:23:35 2015

@author: btrani
"""

import datetime
import requests
import pandas as pd
import sqlite3 as lite
import collections
import matplotlib.pyplot as plt

#Insert lat/long for selected cities
cities = {"Denver_CO": "39.761850,-104.881105",
          "Los_Angeles_CA": "34.019394,-118.410825",
          "New_York_NY": "40.663619,-73.938589",
          "Seattle_WA": "47.620499,-122.350876"}

#Connect to database to store temp data
con = lite.connect('weather.db')
cur = con.cursor()

with con:
    cur.execute('CREATE TABLE temps (day_of_reading INT, Los_Angeles_CA REAL, \
    Denver_CO REAL, Seattle_WA REAL, New_York_NY REAL);')
    con.commit()

#Assign values to be used in the API call and date iterations
end_date = datetime.datetime.now()
current_date = end_date - datetime.timedelta(days=30)
url = "https://api.forecast.io/forecast/"  
api_key = 'c88ed1352711e0044721c8999f619ea8/'
exclude = "exclude=currently,minutely,hourly,flags"

#Create list of dates for the prior 30 days
with con:
    cur.execute('DELETE FROM temps')
    con.commit()
    while end_date > current_date:
        cur.execute('INSERT INTO temps(day_of_reading) VALUES (?)', (int(\
        current_date.strftime('%s')),))
        current_date += datetime.timedelta(days=1)
    con.commit()

#Query the DarkSky API to gather temp data for each city for the past 30 days
for k, v in cities.iteritems():
    current_date = end_date - datetime.timedelta(days=30)    
    while end_date > current_date:
        r = requests.get(url + api_key + v + "," + \
        str(current_date.strftime("%Y-%m-%dT%H:%M:%S")) + "?" + exclude)
        
        with con:
            cur.execute('UPDATE temps SET ' + k + '=' + str(r.json()['daily']\
            ['data'][0]['temperatureMax']) + ' WHERE day_of_reading = \
            ' + current_date.strftime('%s'))
    
        current_date += datetime.timedelta(days=1)
        
con.commit()

#Pull temp data from database into data frame
df = pd.read_sql_query('SELECT * FROM temps ORDER BY day_of_reading', con,\
index_col='day_of_reading')

con.close()

#Calculate the daily change for each city and aggregate
daily_change = collections.defaultdict()
for col in df.columns:
    city_temp = df[col].tolist()
    city_name = col
    temp_change = 0
    for k,v in enumerate(city_temp):
        if k < len(city_temp) - 1:
            temp_change += abs(city_temp[k] - city_temp[k+1])
    daily_change[str(city_name)] = temp_change

#Pull out the key and value with the highest variation for the past 30 days
def keywithmaxval(d):
    v = list(d.values())
    k = list(d.keys())

    return str(k[v.index(max(v))])

def maxval(d):
    v = list(d.values())

    return str(v[v.index(max(v))])
    
print "In the past 30 days, " + (keywithmaxval(daily_change)) + " had the \
most variation in its temperature (" + maxval(daily_change) + " degrees)."

#Use a bar chart to visually confirm findings
plt.bar(range(len(daily_change)), daily_change.values(), align='center')
plt.xticks(range(len(daily_change)), daily_change.keys())

plt.show()


