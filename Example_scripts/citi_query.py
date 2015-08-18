# -*- coding: utf-8 -*-
"""
Created on Mon Jul 20 13:57:26 2015

@author: btrani
"""

import requests
from pandas.io.json import json_normalize
import pandas as pd
import sqlite3 as lite
import time
from dateutil.parser import parse
import collections

#Connect to the Citibike json datasource
r = requests.get('http://www.citibikenyc.com/stations/json')

df = json_normalize(r.json()['stationBeanList'])

#Create sqlite database to house data
con = lite.connect('citi_bike.db')
cur = con.cursor()

with con:
    cur.execute('CREATE TABLE citibike_reference (id INT PRIMARY KEY,\
    totalDocks INT, city TEXT, altitude INT, stAddress2 TEXT, longitude NUMERIC,\
    postalCode TEXT, testStation TEXT, stAddress1 TEXT, stationName TEXT, \
    landMark TEXT, latitude NUMERIC, location TEXT)')

#Create generic SQL statement to iterate over stations
sql = "INSERT INTO citibike_reference (id, totalDocks, city, altitude,\
stAddress2, longitude, postalCode, testStation, stAddress1, stationName,\
landMark, latitude, location) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)"

with con:
    for station in r.json()['stationBeanList']:
        cur.execute(sql,(station['id'],station['totalDocks'],station['city'],\
        station['altitude'],station['stAddress2'],station['longitude'],\
        station['postalCode'],station['testStation'],station['stAddress1'],\
        station['stationName'],station['landMark'],station['latitude'],\
        station['location']))
        
station_ids = df['id'].tolist()

station_ids = ['_' + str(x) + ' INT' for x in station_ids]

with con:
    cur.execute("CREATE TABLE available_bikes (execution_time INT, " + ", "\
    .join(station_ids) + ");")

#Update data every minute for an hour to get a sense of bike usage
for i in range(60):
    r = requests.get('http://www.citibikenyc.com/stations/json')
    exec_time = parse(r.json()['executionTime'])    
    
    
    cur.execute('INSERT INTO available_bikes (execution_time) VALUES (?)',\
    (exec_time.strftime('%s'),))
    con.commit()
    
    id_bikes = collections.defaultdict(int)        
    for station in r.json()['stationBeanList']:
        id_bikes[station['id']] = station['availableBikes']

    for k, v in id_bikes.iteritems():
        cur.execute("UPDATE available_bikes SET _" + str(k) + " = " + \
        str(v) + " WHERE execution_time = " + exec_time.strftime('%s') + \
        ";")
    con.commit()
    
    time.sleep(60)
    
con.close()