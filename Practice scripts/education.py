# -*- coding: utf-8 -*-
"""
Created on Wed Jul 22 16:27:52 2015

@author: btrani
"""

from bs4 import BeautifulSoup
import requests
import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
import sqlite3 as lite
import csv
import math

#Open UN data webpage
url = "http://web.archive.org/web/20110514112442/http://unstats.un.org/unsd/\
demographic/products/socind/education.htm"

r = requests.get(url)

#Send page to BS to make scraping the data easier
soup = BeautifulSoup(r.content)

#Scrape data from educational years table. NOTE: due to html tag structure,
#only a subset of countries could be grabbed. Save to text file.
f = open('output.txt', 'w') 
for row in soup.findAll('tr', {'class': 'tcont'}, limit=93):        
    tds = row('td')
    country = tds[0].string
    year = tds[1].string
    total = tds[4].string
    men = tds[7].string
    women = tds[10].string
    record = (country, year, total, men, women, '\n')  
    f.write(",".join(record))

f.close()

#SOpen csv file with data frame to make it easier to work with
df = pd.read_csv('output.txt', header=None)
df = df.drop(5, 1)
df.columns = ['Country', 'Year', 'Total', 'Men', 'Women']

#Create SQL database to store and join data
con = lite.connect('life_expect.db')
cur = con.cursor()

#Create database tables and write educational years data
with con:
    cur.execute('CREATE TABLE un_data (country TEXT, year INT, total INT, \
    men INT, women INT);')
    cur.execute('CREATE TABLE gdp (country_name TEXT, _1999 REAL, _2000 REAL, \
    _2001 REAL, _2002 REAL, _2003 REAL, _2004 REAL, _2005 REAL, _2006 REAL, \
    _2007 REAL, _2008 REAL, _2009 REAL, _2010 REAL);')
    con.commit()

sql = "INSERT INTO un_data (country, year, total, men, women) VALUES \
(?,?,?,?,?)"

with con:
    for index, row in df.iterrows():
        cur.execute(sql, (row['Country'], row['Year'], row['Total'], row['Men']\
        , row['Women']))
    con.commit()

#Open GDP data for all countries by year and write to GDP table
with open('gdp_data.csv','rU') as inputFile:
    next(inputFile) # skip the first two lines
    next(inputFile)
    next(inputFile)
    next(inputFile)
    header = next(inputFile)
    inputReader = csv.reader(inputFile)
    for line in inputReader:
        with con:
            cur.execute('INSERT INTO gdp (country_name, _1999, _2000, _2001, \
            _2002, _2003, _2004, _2005, _2006, _2007, _2008, _2009, _2010) \
            VALUES ("' + line[0] + '","' + '","'.join(line[43:-5]) + '");')
    con.commit()

#Stack data to create a row for each country/year combination then join data
#from un_data and gdp tables. Send data to data frame
with con:
    cur.execute('SELECT * FROM (SELECT DISTINCT * \
    FROM (SELECT country_name, _1999 AS Total, "1999" AS Year FROM gdp\
    UNION ALL SELECT country_name, _2000 AS Total, 2000 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2001 AS Total, 2001 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2002 AS Total, 2002 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2003 AS Total, 2003 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2004 AS Total, 2004 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2005 AS Total, 2005 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2006 AS Total, 2006 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2007 AS Total, 2007 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2008 AS Total, 2008 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2009 AS Total, 2009 AS Year FROM gdp\
    UNION ALL SELECT country_name, _2010 AS Total, 2010 AS Year FROM gdp)\
    AS x\
    ORDER BY x.country_name)\
    INNER JOIN un_data\
    ON un_data.country = country_name')
    data = cur.fetchall()
    df2 = pd.DataFrame(data)

#Clean data and prep for OLS regression
df2.columns = ['Country', 'GDP', 'Year', 'UN_Country', 'UN_Year', 'Total', \
'Men', 'Women']
df2 = df2[df2.Year == df2.UN_Year]
df2 = df2.drop(['UN_Country', 'UN_Year'], 1)
df2['GDP'] = df2['GDP'].convert_objects(convert_numeric=True)
df2 = df2.dropna()

def log(x):
    return math.log(x)

df2['Log_GDP'] = df2['GDP'].apply(log)

#Create independent and dependent variables
x1 = np.matrix(df2['Total']).transpose()
y = np.matrix(df2['Log_GDP']).transpose()

#Fit OLS model
X = sm.add_constant(x1)
model = sm.OLS(y, X).fit()
print model.summary()

#Create scatterplot to visualize data
plt.plot(x1, y, ".")
plt.title("Total Educational Years and GDP")
plt.ylabel("Log-transformed GDP")
plt.xlabel("Total Educational Years")