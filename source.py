#!/usr/bin/env python
# coding: utf-8

# # Project Title

# ***FYI*** I put this in a  word doc that was submited seperatly, it makes this look alot better. I also have some hand-drawn simple graphs. 
# 
# Brayden Cummins – Checkpoint 1: Project Idea and datasets
# Topic: How can the US and other countries fight Climate Change?
# Project Questions: 
# •	For this analysis, I want to look at all the countries around the world and see which ones have the least greenhouse gas emissions vs the most greenhouse gas emissions. 
# •	I want to compare these countries and then look at how these countries do economically. For example, is there a relationship between these countries’ GDP and their carbon footprint?
# •	Then after I look at just the country by their GDP, I want to see if population impacts greenhouse gas emissions.
# •	Then after we get the analysis of those first few questions, I want to then take it a step further and see what bigger countries like the US can do to help the smaller countries focus on being more environmentally friendly.
# What would the answer look like?
# •	My answer for this project would be to visualize in python what countries are doing the best and worst at fighting climate change. I want to be able to filter these graphs down by year and by country so you can do a more in-depth analysis of certain years/countries.  
# •	Also, with the population data, I want to see if the governments should crack down more on the big businesses or on the everyday person. I want to see how much the average person affects greenhouse emissions. 
# •	I have always wondered which countries are doing the best and the worst when fighting climate change, and this project will fulfill my questions, by matching up greenhouse emissions with other information that could be factored into why that country is doing so good or bad in their emissions. 
# •	On these visualizations, I possibly want to connect the datasets and show them in combined visualizations. So, put the data side by side to better analyze it.
# •	Some basic hand-drawn examples of what I am looking for are below.
#    
# My Data Sources: 
# •	The datasets that I have identified for this project are 2 CSV/Excel files, and an International Database. I found 2 of the datasets from Kaggle, and one from the US Census. 
# •	I changed my datasets from the last assignment. As I read the feedback, I came to realize that I needed better datasets. 
# •	Here are the links to the datasets:
# o	https://www.kaggle.com/datasets/yoannboyere/co2-ghg-emissionsdata
# o	https://www.kaggle.com/datasets/alejopaullier/-gdp-by-country-1999-2022
# o	https://www.census.gov/data-tools/demo/idb/#/table?COUNTRY_YEAR=2022&COUNTRY_YR_ANIM=2022
# How to relate the datasets:
# •	Relating my datasets will be easy, it will be by country and country code. 
# •	All my datasets clearly differentiate the countries and the years that all the data is coming from. If I want to join the datasets, I will have to do it by country (the full name) I think. They should all have the countries spelled the same, so I don’t think there will be an issue. 
# Importing Data into Python:
# •	I imported my datasets in python, and I uploaded it all to the GitHub repository I submitted.

# In[152]:


# Importing first dataset from kaggle using opendatasets. The CO2_GHG_emissions-data.

import pandas as pd
import opendatasets as od
dataset_url = 'https://www.kaggle.com/datasets/yoannboyere/co2-ghg-emissionsdata'
od.download(dataset_url, data_dir='./data')


# In[153]:


ghg = pd.read_csv('data/co2-ghg-emissionsdata/co2_emission.csv')
ghg.head(5)


# In[154]:


dataset_url = 'https://www.kaggle.com/datasets/alejopaullier/-gdp-by-country-1999-2022'
od.download(dataset_url, data_dir='./data')


# In[155]:


gdp = pd.read_csv('data/-gdp-by-country-1999-2022/GDP by Country 1999-2022.csv')
gdp.head()


# In[156]:


gdp.describe()


# In[157]:


ghg.describe()


# In[158]:


import requests

def global_pop(year):
    call = "https://api.census.gov/data/timeseries/idb/5year?get=NAME,POP,CBR,CDR,E0,AREA_KM2&GENC=*&YR=" + year
    response = requests.get(call)
    census_data = pd.DataFrame.from_records(response.json()[1:], columns=response.json()[0])
    return census_data

global_pop = global_pop("2014")
global_pop.describe()


# # Cleaning my datasets - Checkpoint 2 Data Cleaning

# In[159]:


##I am working on finding out which year has the best data in my datasets. So I'm trying to find out what year has the least amount of 0's and Nulls. First, I need to clean up the datasets and columns
gdp_2021 = pd.DataFrame(gdp, columns=['Country', '2021'])
gdp_2021['2021'] = gdp_2021['2021'].str.replace(",", "")

gdp_2021['2021'] = gdp_2021['2021'].astype(float)
gdp_2021.describe()


# In[160]:


#So I can see here that out of the 180 countries in this dataset, there are 15 countries with 0 GDP.
#I'm going to look for a different year to see if its the same. 
gdp_2021[gdp_2021['2021'] == 0]


# In[161]:


#As you can see, 2014 has only 1 country with 0's for the GDP, so I'm going to use 2014 for my calculations. 
gdp_2014 = pd.DataFrame(gdp, columns=['Country', '2014'])
gdp_2014['2014'] = gdp_2014['2014'].str.replace(",", "")
gdp_2014['2014'] = gdp_2014['2014'].astype(float)

gdp_2014[gdp_2014['2014'] == 0]


# In[162]:


#Cleaning up this dataframe
gdp_2014.columns = ('Country', 'GDP (In US Billions)')
gdp_2014.head(15)


# In[163]:


ghg.info()


# In[164]:


#I figured out that this dataset had continents in it as well, so I dropped the rows that didn't have a country code. Now I will check for more nulls. 
ghg_2014_1 = ghg[ghg['Year'] == 2014]
ghg_2014 = ghg_2014_1.dropna()

len(ghg_2014)


# In[165]:


ghg_2014.head(5)


# In[166]:


#We have no nulls, that's great!
ghg_2014.isnull().sum()


# In[167]:


#Same with this dataset, no nulls. We already checked for 0's in this one, so we sould be good. 
gdp_2014.isnull().sum()


# In[168]:


#Checking for zeros for the green house gas emissions. 
ghg_2014[ghg_2014['Annual CO₂ emissions (tonnes )'] == 0].count()


# In[169]:


#I added this later on, but I need to divide the emissions values to a lower value to graph. So thats what I do here.
ghg_2014['Annual CO₂ emissions (tonnes )'] = ghg_2014['Annual CO₂ emissions (tonnes )'] / 1000
ghg_2014['Annual CO₂ emissions (tonnes )'] = ghg_2014['Annual CO₂ emissions (tonnes )'].round(1)
ghg_2014.head()


# In[170]:


#Now, I need to work on the last dataset. The first two I have got rid of the nulls, zeros, not needed rows, and changed all the datatypes so they can be used.
#In my data, there is going to be large and small numbers. Smaller and bigger countries, so getting rid ot outliers would not be so good. While i'm using the data, 
# I will fact check some numbers to make sure they are correct. 


# In[171]:


global_pop.head(15)


# In[172]:


#This dataset is looking good, no nulls except for the E0 column, so i'll drop the columns that i'm not going to use. 
global_pop.isnull().sum()


# In[173]:


#Now I have the columns that I want
global_pop.drop(global_pop.columns[[2,3,4,7]], axis=1, inplace=True)
global_pop.head()


# In[174]:


gdp_2014.info()


# In[175]:


ghg_2014.columns = ('Country', 'Code', 'Year', 'Annual Emissions (Tonnes X 1000)')
global_pop.columns = ('Country', 'Population', 'Area (KM)', 'GENC')


# # Now that I have clean code, I'm going to try and join them based on their name.
# ## I can already see I'm going to have to do some manipulating of the names, because not all of them are the same. 

# In[176]:


data_check = ghg_2014.merge(global_pop, how='outer', on=['Country'])
data_check2 = global_pop.merge(ghg_2014, how='outer', on=['Country'])
len(data_check)


# In[177]:


#trying to figure out why I have nulls, its probably because of the country names not matching up
data_check[data_check['Population'].isnull()]


# In[178]:


test = global_pop[global_pop['Country'].str.startswith('K')]
test


# In[179]:


global_pop.at[28, 'Country'] = 'Bahamas'
global_pop.at[214, 'Country'] = 'British Virgin Islands'
global_pop.at[46, 'Country'] = 'Cape Verde'
global_pop.at[36, 'Country'] = 'Republic of the Congo'
global_pop.at[34, 'Country'] = 'Democratic Republic of Republic of the Congo'
global_pop.at[38, 'Country'] = 'Cote d\'Ivoire'
global_pop.at[47, 'Country'] = 'Curacao'
global_pop.at[49, 'Country'] = 'Czech Republic'
global_pop.at[65, 'Country'] = 'Faeroe Islands'
global_pop.at[134, 'Country'] = 'Macao'
global_pop.at[130, 'Country'] = 'Macedonia'
global_pop.at[64, 'Country'] = 'Micronesia (country)'
global_pop.at[132, 'Country'] = 'Myanmar'
global_pop.at[107, 'Country'] = 'North Korea'
global_pop.at[108, 'Country'] = 'South Korea'
global_pop.at[192, 'Country'] = 'Swaziland'
global_pop.at[198, 'Country'] = 'Timor'
global_pop.at[179, 'Country'] = 'Saint Helena'
global_pop.at[82, 'Country'] = 'French Guiana'


# In[180]:


data_check = ghg_2014.merge(global_pop, how='outer', on=['Country'])
data_check[data_check['Population'].isnull()]


# In[181]:


## What I did above was match up the names, All the names didnt match. So I went through the global_pop dataframe and replaced the values that didnt match with the matching names. I used its index to replace 
## the values. 

#I started out with around 20 something values that didnt match and I ended with 8, I couldnt find matches for the ones that are left.


# In[182]:


##This is the inner join, so we should only get values that match. 
final_data_1 = ghg_2014.merge(global_pop, how='inner', on=['Country'])
len(final_data_1)


# In[183]:


final_data = final_data_1.merge(gdp_2014, how='outer', on=['Country'])

final_data[final_data['Population'].isnull()]


# In[184]:


test2 = gdp_2014[gdp_2014['Country'].str.startswith('Timor')]
test2


# In[185]:


gdp_2014.at[177, 'Country'] = 'Yemen'
gdp_2014.at[0, 'Country'] = 'Afghanistan'
gdp_2014.at[10, 'Country'] = 'Bahamas'
gdp_2014.at[23, 'Country'] = 'Brunei'
gdp_2014.at[37, 'Country'] = 'Democratic Republic of Republic of the Congo'
gdp_2014.at[38, 'Country'] = 'Republic of the Congo'
gdp_2014.at[40, 'Country'] = 'Cote d\'Ivoire'
gdp_2014.at[68, 'Country'] = 'French Guiana'
gdp_2014.at[71, 'Country'] = 'Hong Kong'
gdp_2014.at[76, 'Country'] = 'Iran'
gdp_2014.at[88, 'Country'] = 'Kyrgyzstan'
gdp_2014.at[89, 'Country'] = 'Laos'
gdp_2014.at[97, 'Country'] = 'Macedonia'
gdp_2014.at[134, 'Country'] = 'Sao Tome and Principe'
gdp_2014.at[141, 'Country'] = 'Slovakia'
gdp_2014.at[147, 'Country'] = 'Saint Kitts and Nevis'
gdp_2014.at[148, 'Country'] = 'Saint Lucia'
gdp_2014.at[149, 'Country'] = 'Saint Vincent and the Grenadines'
gdp_2014.at[155, 'Country'] = 'Syria'
gdp_2014.at[156, 'Country'] = 'Taiwan'
gdp_2014.at[160, 'Country'] = 'Timor'


# In[186]:


final_data = final_data_1.merge(gdp_2014, how='outer', on=['Country'])
final_data[final_data['Population'].isnull()]


# In[187]:


##I did the same thing with the names when I joined the 3rd dataset. I got it down to a few missing countries. Now time for the inner join. Ater the inner I should have my main dataset ready. 
test3 = final_data_1[final_data_1['Country'].str.startswith('United')]
test3


# In[188]:


#No Nulls in the final data, now that I have the data clean. I want to show it in some visualizations.
final_data = final_data_1.merge(gdp_2014, how='inner', on=['Country'])

final_data.isnull().sum()


# In[189]:


final_data['Population'] = final_data['Population'].astype(int)


# # Data Visualization - Showing the seperate datasets, then the final.

# In[190]:


#First going to show the ghg_2014 dataset, this will show the ghg by country for the year 2014. 
import matplotlib as mpl
import matplotlib.pyplot as plt

plt.scatter(ghg_2014['Country'], ghg_2014['Annual Emissions (Tonnes X 1000)'])
plt.show()


# In[191]:


test4 = final_data[final_data['Country'].str.startswith('China')]
test4


# In[192]:


plt.scatter(gdp_2014['Country'], gdp_2014['GDP (In US Billions)'])


# In[193]:


## As you can see from graphing these first few datasets. We have A LOT of smaller countries with smaller GDP and smaller emissions that are making the graphs look very weird.
## The few larger countries are making the graphs look weird with their bigger values, but I could have guessed that was going to happen. So lets try and get better visualizations.
#Hopefully when I do calculations like green house gas / population these graphs start to look better. I will have to do more data manipulation for that. 
## Also, there is a world value in the GHG dataset, so the final dataset should be better for that. 


# In[194]:


#Trying to get the top 5 couuntires
import seaborn as sns
sns.scatterplot(x = ghg_2014['Country'], y = ghg_2014['Annual Emissions (Tonnes X 1000)'], data = ghg_2014.sort_values('Annual Emissions (Tonnes X 1000)').tail(5))


# In[195]:


print(ghg_2014.sort_values('Annual Emissions (Tonnes X 1000)').tail(5))


# In[196]:


#This is showing the top 5 countries with the largest GHG emissions. 
sns.scatterplot(data=final_data.sort_values('Annual Emissions (Tonnes X 1000)').tail(5), x='Country', y='Annual Emissions (Tonnes X 1000)', hue='Annual Emissions (Tonnes X 1000)')


# In[197]:


#This shows the population
sns.scatterplot(data=final_data.sort_values('Population').tail(5), x='Country', y='Population')


# In[198]:


sns.scatterplot(data=final_data.sort_values('GDP (In US Billions)').tail(5), x='Country', y='GDP (In US Billions)')


# In[199]:


final_data['Area (KM)'] = final_data['Area (KM)'].astype('int')


# In[200]:


final_data = final_data.drop('GENC', axis=1)
final_data = final_data.drop('Code', axis=1)
final_data = final_data.drop('Year', axis=1)


# In[201]:


final_data.info()


# In[202]:


# As you can see from the three graphs above using Seaborn. I want to use the data from population, GDP, and each countires emissions to see if there is a correlation between any of them.
# As you can see, there looks to be some sort of correlation in all 3 of these columns. So, eventually, I want to be able to show the smaller countires as well. 
# And for the machine learning, you could have a algorithm that takes in population, gpd, and and tell you what your emissions might be based on this info. 


# # What's next for me
# ## I want to make some new columns that have like population divided by GHG, or even bring in the area of the country and do GHG divided by area to see if there is a correlation there. 
# ## I want to see if any of these factors increase or descrease emissions, and I want to be able to show them. 

# In[203]:


## Im going to go through the EDA questions to try and answer them since I have a better understaind of my data now. 

## Some intresting information that I have found is that I think there are going to be correlations in all of the data that I havepulling into this project, I think that all of these
# datasets have affects on GHG emissions

## The distribution of the varaibles, I don't quite fully understand that question. But I think that every country will be different, and they will all be different.

## I don't see any issues with my data at this point.I have cleaned them up pretty well. I think I COULD have an issue with showing the differnet countries with bigger countries making the graphs look 
## weird, but when I do the division per population and area I think that could make it look better.

##There are going to be "outliers" because of countires like china and the US that are just huge in every category. I can't take them out those, those are 2 pretty important countires.

## I cleaned the data to now have any missing values, so I should be good there.

## There arnt any duplicate values either. The outer joins that I had would have showed me if there were duplicate countires, and I didnt see any. The inner joins would have cleaned that as well I think.

## I changed data types as well, a lot. The data types gave me issues because I forgot to change a few of them, and when I went to graph them (population for example), they were being really weird. 


# # Machine learing Plan

# In[204]:


## I want to be able to have a algorithm where it can try and predict GHG emissions based on population, GDP, and Area of the country. I think that all of these have a correlation to GHG emissions and I think
## this is very possible, I just need to figure out how to do it. I've never done machine learning stuff before.

## I haven't identified any issues yet, but I probably will run into some. 

## The issues will probably have to do with the big numbers, so I will have to clean the data some more to make python happy. 


# # Machine Learning

# In[205]:


# First I want to see if there is any correlation between the columns, also start with showing a histogram of the data. 
final_data.hist(bins = 50, figsize = (20,15))


# In[206]:


pd.plotting.scatter_matrix(final_data)


# In[207]:


## This shows that there is a pretty good correlation between Emissions and population, and Emmisions and GDP. But no so much for the area of the country
corr = final_data.corr()
sns.heatmap(corr, annot = True, cmap = 'coolwarm')


# In[208]:


## I'm going to drop the area column because it does not correlate very well, and I think that it will affect the outcome of the model.
## I think GDP and Population will do just fine.

final_data = final_data.drop('Area (KM)', axis=1)


# In[209]:


from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
## For preprocessing
from sklearn.preprocessing import (
  OneHotEncoder,
  OrdinalEncoder,
  StandardScaler
)
from sklearn.impute import (
  SimpleImputer
)
## For model selection
from sklearn.model_selection import (
  StratifiedShuffleSplit,
  train_test_split,
  cross_val_score,
  KFold,
  GridSearchCV
)


# In[210]:


train_set, test_set = train_test_split(final_data, test_size = 0.2, random_state = 32)


# In[211]:


train_set.info()


# In[212]:


data_X = train_set.drop('Annual Emissions (Tonnes X 1000)', axis=1)
data_Y = train_set['Annual Emissions (Tonnes X 1000)']
display(data_X.head())
display(data_Y.head())


# In[213]:


data_X_num = data_X.drop('Country', axis=1)
data_X_cat = data_X['Country']


# In[214]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OrdinalEncoder

data_X_num_scaled = StandardScaler().fit_transform(data_X_num)
data_X_num_scaled = pd.DataFrame(data_X_num_scaled, columns = data_X_num.columns, index = data_X_num.index)

data_X_num_scaled.head(10)


# In[216]:


## Since I only have country as my only categorical column and I dont need to do anything to it, i'm not going to use a pipeline since I already scaled my data.

data_X_num_scaled[:5]


# In[217]:


from sklearn.linear_model import LinearRegression

GHG_lin_reg = LinearRegression()
GHG_lin_reg.fit(data_X_num_scaled, data_Y)


# In[218]:


## Well, as you can see I'm not sure if I did everything correctly, but were going to keep trying. 
import numpy as np
from sklearn.metrics import mean_squared_error
predictions = GHG_lin_reg.predict(data_X_num_scaled)
lin_mse = mean_squared_error(data_Y, predictions)
lin_mse = np.sqrt(lin_mse)
lin_mse


# In[225]:


data_X.info()


# In[222]:


plt.scatter(data_X, data_Y, color="black")
plt.plot(data_X, GHG_lin_reg.predict(data_X), color="blue", linewidth=3)

plt.show()


# In[215]:


#This never works for me, I have to use the terminal
get_ipython().system('jupyter nbconvert --to python source.ipynb')

