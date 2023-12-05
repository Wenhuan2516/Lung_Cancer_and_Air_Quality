#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import geopandas as gpd
from pysal.model import mgwr
from mgwr.gwr import GWR, GWRResults
from pysal.lib import weights
from pysal.explore import esda


# In[2]:


import dask.dataframe as ddf


# In[3]:


years = [year for year in range(2012, 2017)]


# In[4]:


data = []
for year in years:
    df =  ddf.read_csv(r'/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/age_adjusted_county_level/county_level_age_adjusted_rates/age_adjusted_rates_' + str(year) + '.csv', 
                       dtype={'year': int, 'fips': str}).compute().drop(columns={'Unnamed: 0'})
    data.append(df)


# In[5]:


cancer = pd.concat(data)
cancer


# In[6]:


social = ddf.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/INLA/covariates_total_0918.csv', dtype = {'year': int, 'fips': str}).compute()
social = social.loc[:, ~social.columns.str.contains('^Unnamed')]
social


# In[7]:


social_updated = social.merge(cancer[['year', 'fips', 'total_age_adjusted_rate']], on = ['year', 'fips'], how = 'inner')
social_updated.head()


# In[9]:


social_average = social_updated.drop('year', axis = 1)
social_average = social_average.groupby(['fips']).mean()
social_average = social_average.reset_index()
social_average


# In[10]:


rurality = pd.read_csv('/global/cfs/cdirs/m1532/Projects_MVP/geospatial/SHAP_Aggregated_0918/county_rurality.csv', dtype = {'FIPS code': str})
rurality


# In[11]:


def formatFips(fips):
    return fips.rjust(5, '0')


# In[12]:


rurality['FIPS code'] = rurality['FIPS code'].apply(formatFips)
rurality.head()


# In[13]:


rurality = rurality.rename(columns = {'FIPS code': 'fips'})
rural_2006 = rurality[['fips', '2006 code']]
rural_2013 = rurality[['fips', '2013 code']]


# In[14]:


social_rurality = social_average.merge(rural_2013, on = ['fips'], how = 'inner')
social_rurality = social_rurality.rename(columns = {'2013 code': 'rurality'})
social_rurality


# In[15]:


import geopandas as gpd


# In[16]:


ur_files = ddf.read_csv(r'/global/cfs/cdirs/m1532/Projects_MVP/geospatial/data_imputation_tutorial/Climate_Data_Imputation/Step1_generate_geographic_level_data/county_boundary.csv', dtype = {'STATEFP': str, 'COUNTYFP': str, 'TRACTCE': str, 'NAME': str,'NAMELSAD': str, 'GEOID': str})
county = ur_files.compute()
county = county.loc[:, ~county.columns.str.contains('^Unnamed')]
county


# In[17]:


county = county.rename(columns = {'GEOID': 'fips'})
county.head()


# In[18]:


county_new = social_rurality.merge(county[['fips', 'geometry']], on = ['fips'], how = 'left')
county_new.head()


# In[19]:


county_new.columns


# In[20]:


variable_list = ['median_age', 'percent_age_65_and_older',
       'percent_age_18_34', 'percent_age_35_64', 'percent_asian',
       'percent_american_indian', 'percent_not_hispanic_white',
       'percent_black', 'percent_hispanic', 'median_household_income',
       'median_per_capita_income', 'percent_below_poverty',
       'percent_no_high_school_diploma',
       'percent_workers_management_occupation',
       'percent_workers_service_occupation',
       'percent_workers_sales_occupation',
       'percent_workers_argriculture_occupation',
       'percent_workers_construction_occupation',
       'percent_workers_production_occupation', 'percent_divorced',
       'percent_male', 'percent_over_crowding', 'smoke_amount_under_100',
       'smoke_amount_greater_100', 'smoke_everyday', 'smoke_somedays',
       'still_smoking', 'stopped_smoking', 'general_health_poor',
       'health_care_coverage_no', 'chronic_disease_yes', 'cost_too_high_yes',
       'time_unchecked_greater_5Y', 'time_unchecked_never', 'pm25', 'ozone',
       'no2', 'AQI', 'pm10', 'so2', 'CO', 'rurality']


# In[21]:


df_normal = county_new.drop(['fips', 'geometry'], axis = 1)
df_normal.head()


# In[22]:


import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Create a MinMaxScaler object for Min-Max Normalization
scaler_minmax = MinMaxScaler()

# Perform Min-Max Normalization and update the DataFrame
df_normalized_minmax = pd.DataFrame(scaler_minmax.fit_transform(df_normal), columns= df_normal.columns)
df_normalized_minmax


# In[23]:


df_new = pd.concat([county_new[['fips', 'geometry']], df_normalized_minmax], axis =1)
df_new.head()


# In[24]:


from shapely import wkt


# In[25]:


def findCentroid(g):
    p1 = wkt.loads(g)
    return p1.centroid.wkt


# In[26]:


df_new['Centroid'] = df_new['geometry'].apply(findCentroid)
df_new.head()


# In[27]:


def findLon(point):
    strList = point.split(' ')
    lon = strList[1][1:]
    lon = float(lon)
    return lon


# In[28]:


def findLat(point):
    strList = point.split(' ')
    lat = strList[2][:-1]
    lat = float(lat)
    return lat


# In[29]:


df_new['lat'] = df_new['Centroid'].apply(findLat)
df_new['lon'] = df_new['Centroid'].apply(findLon)
df_new.head()


# In[30]:


# Convert the 'polygon_strings' column to a 'geometry' column containing Polygon objects
df_new['geometry'] = df_new['geometry'].apply(wkt.loads)

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(df_new, geometry='geometry')

# Now 'gdf' is a GeoDataFrame with a 'geometry' column containing Polygon objects
gdf.head()


# In[31]:


gdf = gdf.dropna()


# In[32]:


# The dependent variable
y = gdf[['total_age_adjusted_rate']].values

# The independent variables - insert your variables here
X = gdf[variable_list].values

# Convert the GeoDataFrame to a PySAL weights object
w = weights.DistanceBand.from_dataframe(gdf, threshold = 0.5, binary=True, silence_warnings=True)


# In[33]:


from shapely.geometry import Point


# In[34]:


gdf['coors'] = gdf.apply(lambda row: Point(row['lon'], row['lat']), axis=1)

# Convert the DataFrame to a GeoDataFrame
gdf = gpd.GeoDataFrame(gdf, geometry='geometry')
gdf.head()


# In[35]:


gdf['coors'].dtype


# In[36]:


coordinates = list(gdf.coors.apply(lambda point: (point.x, point.y)))


# In[37]:


gwr_model = GWR(coordinates, y, X, w)


# In[38]:


import warnings
warnings.filterwarnings("ignore")


# In[39]:


#from mgwr.gwr import GWR, Sel_BW
from libpysal.weights import DistanceBand, Queen
from mgwr.gwr import GWR
from mgwr.sel_bw import Sel_BW

# ... after preparing your data and the model ...

# Select the bandwidth using a criterion such as AIC
bw = Sel_BW(coordinates, y, X).search(criterion='AIC')

# Fit the model with the selected bandwidth
model = GWR(coordinates, y, X, bw=bw)
results = model.fit()


# In[40]:


# Inspect the results
print(results.summary())


# In[41]:


results


# In[42]:


print(dir(results))


# In[43]:


gdf['gwr_fitted_values'] = results.predy


# In[44]:


gdf


# In[45]:


gdf['gwr_residuals'] = results.resid_response


# In[46]:


gdf.head()


# In[47]:


gdf['gwr_fitted_values'].describe()


# In[48]:


gdf['total_age_adjusted_rate'].describe()


# In[49]:


max_value = gdf['gwr_fitted_values'].max()
min_value = gdf['gwr_fitted_values'].min()


# In[50]:


gdf['Vulnerability Index'] = (gdf['gwr_fitted_values'] - min_value) / (max_value - min_value)
gdf.head()


# In[52]:


gdf.to_csv('GWT_results.csv')


# In[51]:


gdf['Vulnerability Index'].describe()


# In[57]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(gdf, geojson=counties, locations='fips', color='Vulnerability Index',
                           color_continuous_scale="rainbow",
                           range_color=(0, 0.5),
                           scope="usa",
                           hover_name="fips",
                           labels={'Vulnerability Index':'Vulnerability Index'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Compare with the real lung mortality from 2017 to 2020

# In[58]:


years = [year for year in range(2017, 2021)]


# In[59]:


data = []
for year in years:
    df =  ddf.read_csv(r'/global/cfs/cdirs/m1532/Projects_MVP/geospatial/Lung_cancer/age_adjusted_county_level/county_level_age_adjusted_rates/age_adjusted_rates_' + str(year) + '.csv', 
                       dtype={'year': int, 'fips': str}).compute().drop(columns={'Unnamed: 0'})
    data.append(df)


# In[60]:


cancer = pd.concat(data)
cancer.head()


# In[61]:


cancer_real = cancer[['fips', 'total_age_adjusted_rate']]
cancer_real = cancer_real.groupby(['fips']).mean()
cancer_real = cancer_real.reset_index()
cancer_real


# In[62]:


max_value = cancer_real['total_age_adjusted_rate'].max()
min_value = cancer_real['total_age_adjusted_rate'].min()


# In[63]:


cancer_real['Age Adjusted Rate Pencentile'] = (cancer_real['total_age_adjusted_rate'] - min_value) / (max_value - min_value)


# In[64]:


cancer_real.head()


# In[67]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)

import plotly.express as px

fig = px.choropleth(cancer_real, geojson=counties, locations='fips', color='total_age_adjusted_rate',
                           color_continuous_scale="rainbow",
                           range_color=(0, 60),
                           scope="usa",
                           hover_name="fips",
                           labels={'total_age_adjusted_rate':'total_age_adjusted_rate'}
                          )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# ### Create the map based on the percentile

# In[68]:


min_value_cancer = gdf['Vulnerability Index'].min()
max_value_cancer = gdf['Vulnerability Index'].max()

# Use cut to create 8 bins from min to max
gdf['index_interval'] = pd.qcut(gdf['Vulnerability Index'], q=8)
gdf.head()


# In[69]:


for item in list(gdf['index_interval'].unique()):
    print(item)


# In[70]:


def defineIndexRange(v):
    if v >= 0 and v<= 0.264:
        return '0-0.264'
    elif v > 0.264 and v <= 0.304:
        return '0.264-0.304'
    elif v > 0.304 and v <= 0.336:
        return '0.304-0.336'
    elif v < 0.336 and v <= 0.366:
        return '0.336-0.366'
    elif v > 0.366 and v <= 0.395:
        return '0.366-0.395'
    elif v > 0.395 and v <=0.427:
        return '0.395-0.427'
    elif v > 0.427 and v <= 0.469:
        return '0.427-0.469'
    else:
        return '0.469-1'


# In[71]:


gdf['Vulnerability_Index'] = gdf['Vulnerability Index'].apply(defineIndexRange)
gdf.head()


# In[72]:


color_map1 = {'0-0.264': '#ffff80', '0.264-0.304': '#fce062',  '0.304-0.336': '#f7c34a', '0.336-0.366': '#f1a62f', 
             '0.366-0.395': '#c86a1a', '0.395-0.427': '#963a0c', '0.427-0.469': '#6a0601', '0.469-1':'#4a0400'}


# In[75]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import pandas as pd
import plotly.express as px

gdf_ordered = gdf.sort_values("Vulnerability Index", axis = 0, ascending = False)
fig = px.choropleth_mapbox(gdf_ordered, geojson=counties, locations='fips', color='Vulnerability_Index',
                            color_discrete_map = color_map1,
                            mapbox_style = 'carto-positron',
                            zoom = 3, center = {"lat": 37.0902, "lon": -95.7129}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[74]:


cancer_real


# In[76]:


cancer_real = cancer_real.rename(columns = {'total_age_adjusted_rate': 'lung cancer age adjusted mortality rate'})


# In[77]:


min_value_cancer = cancer_real['lung cancer age adjusted mortality rate'].min()
max_value_cancer = cancer_real['lung cancer age adjusted mortality rate'].max()

# Use cut to create 8 bins from min to max
cancer_real['rate_interval'] = pd.qcut(cancer_real['lung cancer age adjusted mortality rate'], q=8)
cancer_real.head()


# In[78]:


for item in list(cancer_real['rate_interval'].unique()):
    print(item)


# In[79]:


def defineRateRange(v):
    if v >= 0 and v<= 21.608:
        return '0-21.608'
    elif v > 21.608 and v <= 27.796:
        return '21.608-27.796'
    elif v > 27.796 and v <= 32.444:
        return '27.796-32.444'
    elif v < 32.444 and v <= 37.066:
        return '32.444-37.066'
    elif v > 37.066 and v <= 41.853:
        return '37.066-41.853'
    elif v > 41.853 and v <= 47.29:
        return '41.853-47.29'
    elif v > 47.29 and v <= 55.501:
        return '47.29-55.501'
    else:
        return '55.501-249.488'


# In[80]:


cancer_real['age adjusted rate'] = cancer_real['lung cancer age adjusted mortality rate'].apply(defineRateRange)
cancer_real.head()


# In[81]:


color_map2 = {'0-21.608': '#ffff80', '21.608-27.796': '#fce062',  '27.796-32.444': '#f7c34a', '32.444-37.066': '#f1a62f', 
             '37.066-41.853': '#c86a1a', '41.853-47.29': '#963a0c', '47.29-55.501': '#6a0601', '55.501-249.488':'#4a0400'}


# In[82]:


from urllib.request import urlopen
import json
with urlopen('https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json') as response:
    counties = json.load(response)
import pandas as pd
import plotly.express as px

df_ordered = cancer_real.sort_values("lung cancer age adjusted mortality rate", axis = 0, ascending = False)
fig = px.choropleth_mapbox(df_ordered, geojson=counties, locations='fips', color='age adjusted rate',
                            color_discrete_map = color_map2,
                            mapbox_style = 'carto-positron',
                            zoom = 3, center = {"lat": 37.0902, "lon": -95.7129}
                            )
fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})
fig.show()


# In[ ]:




