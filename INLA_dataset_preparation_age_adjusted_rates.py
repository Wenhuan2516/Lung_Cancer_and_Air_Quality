#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import dask.dataframe as ddf
from pandas import Series, DataFrame
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sn
import plotly.express as px


# ### Load lung cancer death data

# In[2]:


years = [year for year in range(2009, 2019)]
years


# In[3]:


data = []
for year in years:
    death_year =ddf.read_csv("/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/age_adjusted_county_level/county_level_age_adjusted_rates/age_adjusted_rates_" + str(year) + ".csv", dtype={'fips': str, 'year': int, 'county_death': str, 'county_residence': str, 'state_residence': 'object', 'state_death': 'object'}).compute().drop(columns={'Unnamed: 0'})
    data.append(death_year)


# In[4]:


cancer = pd.concat(data)
cancer


# In[5]:


lung_cancer = cancer[['year', 'statefips', 'fips', 'state', 'total_deaths', 'total_pop', 'total_age_adjusted_rate']]
lung_cancer


# In[6]:


lung_cancer['total_age_adjusted_rate'].describe()


# In[7]:


def defineRateRange(r):
    if r <= 10:
        return '<=10'
    elif r >= 10.1 and r <= 20:
        return '10.1-20'
    elif r >= 20.1 and r <= 30:
        return '20.1-30'
    elif r >= 30.1 and r <= 40:
        return '30.1-40'
    elif r >= 40.1 and r <= 50:
        return '40.1-50'
    elif r >= 50.1 and r <= 60:
        return '50.1-60'
    elif r >= 60.1 and r <= 70:
        return '60.1-70'
    elif r >= 70.1 and r <= 80:
        return '70.1-80'
    elif r >= 80.1 and r < 90:
        return '80.1-90'
    elif r >= 90.1 and r <= 100:
        return '90.1-100'
    else:
        return '>100'


# In[8]:


lung_cancer['RateRange'] = lung_cancer['total_age_adjusted_rate'].apply(defineRateRange)
lung_cancer


# In[9]:


lung_cancer['total_deaths'].describe()


# In[10]:


def defineDeathRange(d):
    if d < 20:
        return '<20'
    elif d >= 20 and d < 40:
        return '20-40'
    elif d >= 40 and d < 60:
        return '40-60'
    elif d >= 60 and d < 80:
        return '60-80'
    elif d >= 80 and d < 100:
        return '80-100'
    else:
        return '>100'


# In[11]:


lung_cancer['DeathRange'] = lung_cancer['total_deaths'].apply(defineDeathRange)


# In[12]:


lung_cancer


# In[26]:


color_map = {'<=10': '#5f50a2', '10.1-20': '#3388bd', '20.1-30': '#66c2a6', '30.1-40': '#abdda4', '40.1-50': '#e6f598', '50.1-60': '#ffffbf',
            '60.1-70': '#fde08b', '70.1-80': '#fcad61', '80.1-90': '#f36d44', '90.1-100': '#b63e4f', '>100': '#9e0143'}


# In[14]:


year = 2009


# In[15]:


lung_cancer['year'].dtype


# In[ ]:





# In[16]:


county = pd.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/data_imputation_tutorial/Mortality_Data_Imputation/county_adjacency.csv', dtype = {'fips': str}).drop(columns={'Unnamed: 0'})
county


# In[17]:


lung_cancer


# In[18]:


years = [year for year in range(2009, 2019)]


# In[19]:


data = []
for year in years:
    cancer_year = lung_cancer[lung_cancer['year'] == year]
    cancer_county = county.merge(cancer_year, on = ['fips'], how = 'left')
    cancer_county['year'] = year
    data.append(cancer_county)


# In[20]:


df_cancer = pd.concat(data)
df_cancer


# In[21]:


df_cancer['total_deaths'] = df_cancer['total_deaths'].fillna(0)
df_cancer.head()


# ### Create a table to show how many counties have deaths 0, less than 10, less than 20 in each year

# In[22]:


table_data = {}
for year in years:
    amount_list = []
    df_year = df_cancer[df_cancer['year'] == year]
    df_zero = df_year[df_year['total_deaths'] == 0]
    zero_amount = len(df_zero['fips'].unique())
    df_ten = df_year[df_year['total_deaths'] < 10]
    ten_amount = len(df_ten['fips'].unique())
    df_tw = df_year[df_year['total_deaths'] < 20]
    tw_amount = len(df_tw['fips'].unique())
    amount_list = [zero_amount, ten_amount, tw_amount]
    table_data[year] = amount_list


# In[23]:


table_data


# In[24]:


# Convert the dictionary to a list of tuples
data_items = table_data.items()
data_list = [(year, values[0], values[1], values[2]) for year, values in data_items]

# Create a DataFrame using the data
df_table = pd.DataFrame(data_list, columns=['Year', 'Equal to 0', 'Less than 10', 'Less than 20'])

df_table


# In[25]:


df_table['percent_equal_to_0'] = (df_table['Equal to 0']/3233)*100
df_table['percent_less_than_10'] = (df_table['Less than 10']/3233)*100
df_table['percent_less_than_20'] = (df_table['Less than 20']/3233)*100
df_table


# In[26]:


df_cancer


# In[27]:


color_death = {'<20': 'white' , '20-40': '#f7c34a', '40-60': '#f1a62f', '60-80': '#c86a1a', '80-100': '#963a0c', '>100': '#6a0601'}


# In[28]:


year = 2018


# In[37]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import pandas as pd
import plotly.express as px

print('Lung Cancer Deaths in ' + str(year))
df_year = df_cancer[df_cancer['year'] == year]
df_year = df_year.sort_values("total_deaths", axis = 0)
fig = px.choropleth_mapbox(df_year, geojson=counties, locations='fips', color='DeathRange',
                            color_discrete_map = color_death,
                            mapbox_style = 'carto-positron',
                            zoom = 3, center = {"lat": 37.0902, "lon": -95.7129}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Find the average age adjusted rates in each 3 years: 2010-2012, 2013-2015, 2016-2018

# In[29]:


df_group = df_cancer[['year', 'fips', 'total_deaths', 'total_pop', 'total_age_adjusted_rate']]
df_group.head()


# In[30]:


def yearGroup(df, start_year, end_year):
    df_range = df[(df['year'] >= start_year) & (df_group['year'] >= end_year)]
    df_range = df_range[df_range['total_deaths'] >= 20]
    df_range['year_range'] = str(start_year) + '-' + str(end_year)
    df_range = df_range.drop('year', axis = 1)
    df_range = df_range.groupby(['year_range', 'fips']).mean()
    df_range = df_range.reset_index()
    df_range['RateRange'] = df_range['total_age_adjusted_rate'].apply(defineRateRange)
    return df_range


# In[31]:


df_group1 = yearGroup(df_group, 2010, 2012)
df_group1.head()


# In[32]:


df_group2 = yearGroup(df_group, 2013, 2015)
df_group2.head()


# In[33]:


df_group3 = yearGroup(df_group, 2016, 2018)
df_group3.head()


# In[34]:


df_group1 = df_group1.rename(columns = {'RateRange': 'Age Adjusted Rate'})
df_group2 = df_group2.rename(columns = {'RateRange': 'Age Adjusted Rate'})
df_group3 = df_group3.rename(columns = {'RateRange': 'Age Adjusted Rate'})


# In[35]:


color_map = {'<=10': '#5f50a2', '10.1-20': '#3388bd', '20.1-30': '#66c2a6', '30.1-40': '#abdda4', '40.1-50': '#e6f598', '50.1-60': '#ffffbf',
            '60.1-70': '#fde08b', '70.1-80': '#fcad61', '80.1-90': '#f36d44', '90.1-100': '#b63e4f', '>100': '#9e0143'}


# In[61]:


year_range = '2016-2018'


# In[62]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import pandas as pd
import plotly.express as px

print('Lung Cancer Age Adjusted Rate in ' + year_range)

df_year = df_group3.sort_values("total_age_adjusted_rate", axis = 0, ascending = False)
fig = px.choropleth_mapbox(df_year, geojson=counties, locations='fips', color='Age Adjusted Rate',
                            color_discrete_map = color_map,
                            mapbox_style = 'carto-positron',
                            zoom = 3, center = {"lat": 37.0902, "lon": -95.7129}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Prepare the dataset for INLA model

# In[36]:


df_cancer = df_cancer.sort_values(['fips', 'year'])
df_cancer


# In[54]:


df_final = df_cancer[['county', 'fips', 'year', 'total_deaths', 'total_pop', 'total_age_adjusted_rate']]
df_final


# In[38]:


covariates = ddf.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/SDoH/covariates/covariate_*.csv', dtype = {'fips': str, 'year': int, 'median_household_income': 'float64'}).compute().drop(columns={'Unnamed: 0'})
covariates


# In[39]:


years


# In[40]:


pm10_data = []
for year in years:
    pm10_year = ddf.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/Air_Quality_Data/PM10_imputed/PM10_imputed_' + str(year) + '.csv', dtype = {'fips': str, 'year': int, 'median_household_income': 'float64'}).compute().drop(columns={'Unnamed: 0'})
    pm10_data.append(pm10_year)
    
pm10 = pd.concat(pm10_data)
pm10 = pm10.drop(['statefips', 'countyfips', 'date', 'month'], axis = 1)
pm10 = pm10.rename(columns = {'PM10': 'pm10'})
pm10 = pm10.groupby(['year', 'fips']).mean()
pm10 = pm10.reset_index()
pm10.head()


# In[41]:


average_pm10 = pm10['pm10'].mean()
average_pm10


# In[42]:


pm10['pm10'] = pm10['pm10'].fillna(average_pm10)
pm10.head()


# In[43]:


so2_data = []
for year in years:
    so2_year = ddf.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/Air_Quality_Data/SO2_imputed/SO2_imputed_' + str(year) + '.csv', dtype = {'fips': str, 'year': int, 'median_household_income': 'float64'}).compute().drop(columns={'Unnamed: 0'})
    so2_data.append(so2_year)
    
so2 = pd.concat(so2_data)
so2 = so2.drop(['statefips', 'countyfips', 'date', 'month'], axis = 1)
so2 = so2.rename(columns = {'SO2': 'so2'})
so2 = so2.groupby(['year', 'fips']).mean()
so2 = so2.reset_index()
so2.head()


# In[44]:


so2['so2'].isna().sum()


# In[45]:


average_so2 = so2['so2'].mean()
so2['so2'] = so2['so2'].fillna(average_so2)
so2.head()


# In[46]:


co_data = []
for year in years:
    co_year = ddf.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/Air_Quality_Data/CO_imputed/CO_imputed_' + str(year) + '.csv', dtype = {'fips': str, 'year': int, 'median_household_income': 'float64'}).compute().drop(columns={'Unnamed: 0'})
    co_data.append(co_year)
    
co = pd.concat(co_data)
co = co.drop(['statefips', 'countyfips', 'date', 'month'], axis = 1)
co = co.groupby(['year', 'fips']).mean()
co = co.reset_index()
co.head()


# In[47]:


co['CO'].isna().sum()


# In[48]:


average_co = co['CO'].mean()
co['CO'] = co['CO'].fillna(average_co)
co.head()


# In[49]:


covariates = covariates.merge(pm10, on = ['year', 'fips'], how = 'left')
covariates = covariates.merge(so2, on = ['year', 'fips'], how = 'left')
covariates = covariates.merge(co, on = ['year', 'fips'], how = 'left')
covariates.head()


# In[62]:


covariates.to_csv('covariates_total_0918.csv')


# In[50]:


covariate_list = covariates.drop(['year', 'fips'], axis = 1)
covariate_fips = covariates[['year', 'fips']]


# In[51]:


# Standardize the covariates
standardized_covariate_list = (covariate_list - covariate_list.mean()) / covariate_list.std()
standardized_covariate_list


# In[52]:


covariates_updated = pd.concat([covariate_fips, standardized_covariate_list], axis = 1)
covariates_updated


# In[55]:


df_final = df_final[df_final['total_deaths'] >= 20]
df_final


# In[56]:


data = []
for year in years:
    cancer_year = df_final[df_final['year'] == year]
    cancer_county = county.merge(cancer_year, on = ['fips', 'county'], how = 'left')
    cancer_county['year'] = year
    data.append(cancer_county)


# In[57]:


df_final_new = pd.concat(data)
df_final_new.head()


# In[58]:


df_final_updated = df_final_new.merge(covariates_updated, on = ['year', 'fips'], how = 'left')
df_final_updated


# In[60]:


df_final_updated = df_final_updated.drop(['Neighbors', 'Neighbor Code'], axis = 1)
df_final_updated = df_final_updated.sort_values(['fips', 'year'])
df_final_updated


# In[61]:


df_final_updated.to_csv('lung_cancer_age_adjusted_0918_before_INLA.csv')


# ### Show the maps after INLA

# In[2]:


cancer_imputed = pd.read_csv('lung_cancer_age_adjusted_0918_after_INLA.csv', dtype = {'fips': str})
cancer_imputed


# In[3]:


def convertFips(code):
    return str(code).rjust(5, '0')


# In[4]:


cancer_imputed['fips'] = cancer_imputed['fips'].apply(convertFips)
cancer_imputed.head()


# In[5]:


rates = cancer_imputed[['county', 'fips', 'year', 'total_deaths', 'total_pop', 'total_age_adjusted_rate', 'lung_cancer_estimate']]
rates.head()


# In[8]:


rates['Age Adjusted Rate Estimate'] = rates['lung_cancer_estimate'].apply(defineRateRange)
rates.head()


# In[9]:


year = 2014


# In[75]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import pandas as pd
import plotly.express as px

print('Lung Cancer Mortality Rate after Imputed in ' + str(year))
cancer_year = rates[rates['year'] == year]
cancer_year = cancer_year.sort_values("lung_cancer_estimate", axis = 0, ascending = False)
fig = px.choropleth_mapbox(cancer_year, geojson=counties, locations='fips', color='Age Adjusted Rate Estimate',
                            color_discrete_map = color_map,
                            mapbox_style = 'carto-positron',
                            zoom = 3, center = {"lat": 37.0902, "lon": -95.7129}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Check 3 year groups

# In[14]:


def yearGroupNew(df, start_year, end_year):
    df_range = df[(df['year'] >= start_year) & (df['year'] >= end_year)]
    df_range['year_range'] = str(start_year) + '-' + str(end_year)
    df_range = df_range.drop(['year', 'county', 'Age Adjusted Rate Estimate'], axis = 1)
    df_range = df_range.groupby(['year_range', 'fips']).mean()
    df_range = df_range.reset_index()
    df_range['Age Adjusted Rate Estimate'] = df_range['lung_cancer_estimate'].apply(defineRateRange)
    return df_range


# In[15]:


rates.head()


# In[16]:


rates['lung_cancer_estimate'].isna().sum()


# In[17]:


df_group1 = yearGroupNew(rates, 2010, 2012)
df_group2 = yearGroupNew(rates, 2013, 2015)
df_group3 = yearGroupNew(rates, 2016, 2018)


# In[18]:


df_group1.head()


# In[19]:


df_group2.head()


# In[20]:


df_group3.head()


# In[21]:


df_group1


# In[22]:


df_group1['Age Adjusted Rate Estimate'].isna().sum()


# In[23]:


df_group1['Age Adjusted Rate Estimate'].unique()


# In[30]:


year_range = '2016-2018'


# In[31]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import pandas as pd
import plotly.express as px

print('Lung Cancer Mortality Rate after Imputed in ' + year_range)
df_group3 = df_group3.sort_values("lung_cancer_estimate", axis = 0, ascending = False)
fig = px.choropleth_mapbox(df_group1, geojson=counties, locations='fips', color='Age Adjusted Rate Estimate',
                            color_discrete_map = color_map,
                            mapbox_style = 'carto-positron',
                            zoom = 3, center = {"lat": 37.0902, "lon": -95.7129}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:




