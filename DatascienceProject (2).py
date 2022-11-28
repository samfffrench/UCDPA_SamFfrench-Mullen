#!/usr/bin/env python
# coding: utf-8

# In[413]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as plticker

path = r'C:\Users\Sam ffrench-Mullen\Desktop\Data Course\cause_of_death.xlsx' #path to excel file downloaded to desktop

cause_of_death = pd.read_excel(path, index_col=0) # function reads excel file and converts to dataframe

cause_of_death.set_index('Year', append = True) # Sets Year Column as index, creates a multiindex

cause_of_death = cause_of_death.replace(np.nan, 0) # replace missing values with 0

# divided the number of deaths for each cause by total deaths to give the proportion of deaths due to each cause for each country and year

cause_of_death['Rate_Meningitis'] = (cause_of_death['Deaths - Meningitis - Sex: Both - Age: All Ages (Number)'] / cause_of_death['total_deaths']) * 100

cause_of_death['Rate_Dementia'] = (cause_of_death["Deaths - Alzheimer's disease and other dementias - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Parkinson's"] = (cause_of_death["Deaths - Parkinson's disease - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Nutrition"] = (cause_of_death["Deaths - Nutritional deficiencies - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Malaria"] = (cause_of_death["Deaths - Malaria - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Drowning"] = (cause_of_death["Deaths - Drowning - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Murder"] = (cause_of_death["Deaths - Interpersonal violence - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Maternal_Disorders"] = (cause_of_death["Deaths - Maternal disorders - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_HIV"] = (cause_of_death["Deaths - HIV/AIDS - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Drug_Use"] = (cause_of_death["Deaths - Drug use disorders - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_TB"] = (cause_of_death["Deaths - Tuberculosis - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_CV_Disease"] = (cause_of_death["Deaths - Cardiovascular diseases - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Lower_Resp_Inf"] = (cause_of_death["Deaths - Lower respiratory infections - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Neonate"] = (cause_of_death["Deaths - Neonatal disorders - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Alcohol"] = (cause_of_death["Deaths - Alcohol use disorders - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_SelfHarm"] = (cause_of_death["Deaths - Self-harm - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Natural_Disaster"] = (cause_of_death["Deaths - Exposure to forces of nature - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Diarrhea"] = (cause_of_death["Deaths - Diarrheal diseases - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Heat/Cold"] = (cause_of_death["Deaths - Environmental heat and cold exposure - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Cancer"] = (cause_of_death["Deaths - Neoplasms - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Terrorism"] = (cause_of_death["Deaths - Conflict and terrorism - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Diabetes"] = (cause_of_death["Deaths - Diabetes mellitus - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_CKD"] = (cause_of_death["Deaths - Chronic kidney disease - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Poison"] = (cause_of_death["Deaths - Poisonings - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Protein_Def"] = (cause_of_death["Deaths - Protein-energy malnutrition - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Terror"] = (cause_of_death["Terrorism (deaths)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Road_Acc"] = (cause_of_death["Deaths - Road injuries - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Chronic_Resp"] = (cause_of_death["Deaths - Chronic respiratory diseases - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_LiverDisease"] = (cause_of_death["Deaths - Cirrhosis and other chronic liver diseases - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Digestive"] = (cause_of_death["Deaths - Digestive diseases - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Fire"] = (cause_of_death["Deaths - Fire, heat, and hot substances - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_death["Rate_Hepatitis"] = (cause_of_death["Deaths - Acute hepatitis - Sex: Both - Age: All Ages (Number)"] / cause_of_death['total_deaths']) * 100

cause_of_deathrate = cause_of_death.drop(cause_of_death.iloc[:,1:33], axis=1) # Slice Dataframe to remove numbers of deaths from dataframe to leave only the proportion

CountryDeath = cause_of_deathrate.loc[(cause_of_death['Year'] >= 2016) & (cause_of_death['Year'] <= 2019)] # Slices Dataframe to leave values for years between 2000 and 2019


# In[414]:


path_3 = r'C:\Users\Sam ffrench-Mullen\Desktop\Data Course\gross-national-income-per-capita.csv'

GNI = pd.read_csv(path_3, index_col = 0) # function reads csv file and converts to dataframe stored in GNI

GNI.set_index('Year', append = True) # Sets Year Column as index, creates a multiindex

GNI_Recent= GNI.loc[(GNI['Year'] >= 2000) & (GNI['Year'] <= 2019)] # Slices Dataframe to leave values for years between 2000 and 2019

GNI_Recent = GNI_Recent.replace(np.nan, 0) # replace missing values with 0

GNI_Recent.drop('Code', inplace=True, axis=1) # Removes the code column from the Dataframe


# In[415]:


Death_GNI_Pop = pd.merge(CountryDeath, GNI_Recent['GNI per capita, PPP (constant 2017 international $)'], on = ['Entity']) # Merge Dataframes on the Entity column

Death_GNI_Pop.drop_duplicates(subset = 'total_deaths', keep='first', inplace=True) # drop duplicates from the new Dataframe

#Low income: less than $1,036
#Lower-middle income: between $1,036 and $4,045
#Upper-middle income: between $4,046 and $12,535
#High income: greater than $12,535

def Income_Level_Category(Death_GNI_Pop): # Def function of conditionals to assign an income level to countries
    if Death_GNI_Pop['GNI per capita, PPP (constant 2017 international $)'] <= 1036:
        return "Low Income" 
    elif (Death_GNI_Pop['GNI per capita, PPP (constant 2017 international $)'] > 1036) & (Death_GNI_Pop['GNI per capita, PPP (constant 2017 international $)'] <= 4045):
        return "Lower-Middle Income"
    elif (Death_GNI_Pop['GNI per capita, PPP (constant 2017 international $)'] > 4045) & (Death_GNI_Pop['GNI per capita, PPP (constant 2017 international $)'] <= 12535):
         return "Upper-Middle Income"   
    else:
        return "High Income"

Death_GNI_Pop['Income_Level'] = Death_GNI_Pop.apply(Income_Level_Category, axis=1) # creates a new column in Dataframe which uses the above function where the income level of each country is stored

Indexed_Death_GNI_Pop = Death_GNI_Pop.set_index('Year', append = True) # creates new Dataframe  with year column as index


# In[444]:


Indexed_Death_GNI_Pop = Death_GNI_Pop.set_index('Year', append = True)

# Creating the groupby dictionary 
Infection_dict = {'Rate_Meningitis':'Rate_Death_From_Infection',
                'Rate_Malaria':'Rate_Death_From_Infection',
               'Rate_HIV':'Rate_Death_From_Infection',
                 'Rate_TB':'Rate_Death_From_Infection',
                 'Rate_Lower_Resp_Inf' :'Rate_Death_From_Infection',
                 'Rate_Diarrhea' :'Rate_Death_From_Infection',
                 'Rate_Hepatitis' :'Rate_Death_From_Infection'} # Dictionary which assigned each cause of death, realted to infectious disease, 
  
# Groupby the groupby_dict created above 
GNI_Pop_Infection = Indexed_Death_GNI_Pop.groupby(Infection_dict, axis = 1).sum() # Creates New Dataframe where a new column Rate_Death_From_Infection which is the sum of rates of deaths from different infectious diseases

GNI_Level_Pop_Infection = pd.merge(GNI_Pop_Infection, Indexed_Death_GNI_Pop['Income_Level'], left_index=True, right_index=True) # Merge Dataframes on indexes as they were both the same (Multindex:Country and Year) 

Inf_Death_at_Income_Level = GNI_Level_Pop_Infection.groupby(['Income_Level'])['Rate_Death_From_Infection'].mean()# the mean proportion of deaths from infection for countries at each income level
print(Inf_Death_at_Income_Level)

Inf_Death_at_Income_Level.plot(x="Income_Level", y="Rate_Death_From_Infection", kind="bar", color ='maroon',
        width = 0.3) # creates a bar plot with income level on x axis and mean proportion on y axis. Color and width of bars also set
plt.xlabel("Income Level", rotation='horizontal', size=8)#

plt.xticks(rotation='horizontal',size=7)
plt.ylabel("% of Total Deaths due to Infectious Disease", size=8)
plt.title("Deaths from Infectious Disease by the Income Level of Countries", size=10)
plt.show()



# In[424]:


import requests # library to handle requests
from bs4 import BeautifulSoup # library to parse HTML documents

Url ="https://en.wikipedia.org/wiki/Prevalence_of_tobacco_use" # urlof website for scraping
table_class="wikitable sortable jquery-tablesorter" # type of tableWikipedia
response=requests.get(Url)


# parse data from the html into a beautifulsoup object
soup = BeautifulSoup(response.text, 'html.parser')
Tobacco = soup.find('table',{'class':"wikitable"}) #code finds the wikitable code in the html code of the webpage

Prevalence_Of_Tobacco = pd.read_html(str(Tobacco)) # converts the wikitable code fromthe website to a string which is converted to a ist by the function
# convert list to dataframe
Prevalence_Of_Tobacco = pd.DataFrame(Prevalence_Of_Tobacco[0])

Prevalence_Of_Tobacco.set_index(keys=['Country'], drop = True, append = False, inplace = True) # set the country column to an index

CountryDeath2019 = cause_of_deathrate.loc[(cause_of_death['Year'] == 2019)] # create new dataframe sliced to only contain data from 2019

Tobacco_cancer_death = pd.merge(Prevalence_Of_Tobacco, CountryDeath2019['Rate_Cancer'], left_index=True, right_index=True) # merge dataframes on indexes

Income_Level_2019 = Death_GNI_Pop.loc[(Death_GNI_Pop['Year'] == 2019)] # create new dataframe sliced to only contain data from 2019

Income_Tobacco_Cancer = pd.merge(Tobacco_cancer_death, Income_Level_2019['Income_Level'], left_index=True, right_index=True) #  # merge dataframes on indexes
# New Dataframes of the above dataframe slicedby Income Level
HighIncome_Tobacco_Cancer = Income_Tobacco_Cancer.loc[(Income_Tobacco_Cancer['Income_Level'] == 'High Income')]
UM_Income_Tobacco_Cancer = Income_Tobacco_Cancer.loc[(Income_Tobacco_Cancer['Income_Level'] == 'Upper-Middle Income')]
UL_Income_Tobacco_Cancer = Income_Tobacco_Cancer.loc[(Income_Tobacco_Cancer['Income_Level'] == 'Lower-Middle Income')]
LowIncome_Tobacco_Cancer = Income_Tobacco_Cancer.loc[(Income_Tobacco_Cancer['Income_Level'] == 'Low Income')]

High = HighIncome_Tobacco_Cancer.mean()
UM = UM_Income_Tobacco_Cancer.mean()
UL = UL_Income_Tobacco_Cancer.mean()
Low = LowIncome_Tobacco_Cancer.mean()

Highmax = HighIncome_Tobacco_Cancer.max()
UMmax = UM_Income_Tobacco_Cancer.max()
ULmax = UL_Income_Tobacco_Cancer.max()
Lowmax = LowIncome_Tobacco_Cancer.max()

print(High)
print(UM)
print(UL)
print(Low)

print(Highmax)
print(UMmax)
print(ULmax)
print(Lowmax)


import matplotlib.pyplot as plt
fig, axes = plt.subplots(nrows=2, ncols=2) # creation of axes for subplot 2 columns and 2 rows


axes[0][0].scatter(HighIncome_Tobacco_Cancer['2020'], HighIncome_Tobacco_Cancer['Rate_Cancer'], c ="pink", linewidths = 1, edgecolor ="pink", s = 10)# plot of scatter plot with x and y values as well as color and size of markers
axes[0][0].set_title("High Income Countries", size=10)#plot title
axes[0][0].set_xlabel("% of Smokers in Population", size=8) # x label title
axes[0][0].set_ylabel("% Deaths due to Cancer", size=8) # y label title
m, b = np.polyfit(HighIncome_Tobacco_Cancer['2020'], HighIncome_Tobacco_Cancer['Rate_Cancer'], 1) # creates the least squares fit association between columns
axes[0][0].plot(HighIncome_Tobacco_Cancer['2020'], m*(HighIncome_Tobacco_Cancer['2020']) + b, linewidth = 0.35) # function of a line to give trendline

axes[0][1].scatter(UM_Income_Tobacco_Cancer['2020'], UM_Income_Tobacco_Cancer['Rate_Cancer'], c ="red", linewidths = 1, edgecolor ="pink", s = 10)# plot of scatter plot with x and y values as well as color and size of markers
axes[0][1].set_title("Upper-Middle Income Countries", size=10)#plot title
axes[0][1].set_xlabel("% of Smokers in Population", size=8)# x label title
axes[0][1].set_ylabel("% Deaths due to Cancer", size=8)# y label title
m, b = np.polyfit(UM_Income_Tobacco_Cancer['2020'], UM_Income_Tobacco_Cancer['Rate_Cancer'], 1)# creates the least squares fit association between columns
axes[0][1].plot(UM_Income_Tobacco_Cancer['2020'], m*(UM_Income_Tobacco_Cancer['2020']) + b, color='green', linewidth = 0.35)# function of a line to give trendline

axes[1][0].scatter(UL_Income_Tobacco_Cancer['2020'], UL_Income_Tobacco_Cancer['Rate_Cancer'], c ="blue", linewidths = 1, edgecolor ="pink", s = 10)# plot of scatter plot with x and y values as well as color and size of markers
axes[1][0].set_title("Upper-Lower Income Countries", size=10)#plot title
axes[1][0].set_xlabel("% of Smokers in Population", size=8)# x label title
axes[1][0].set_ylabel("% Deaths due to Cancer", size=8)# y label title
m, b = np.polyfit(UL_Income_Tobacco_Cancer['2020'], UL_Income_Tobacco_Cancer['Rate_Cancer'], 1)# creates the least squares fit association between columns
axes[1][0].plot(UL_Income_Tobacco_Cancer['2020'], m*(UL_Income_Tobacco_Cancer['2020']) + b, color='yellow', linewidth = 0.35)# function of a line to give trendline

axes[1][1].scatter(LowIncome_Tobacco_Cancer['2020'], LowIncome_Tobacco_Cancer['Rate_Cancer'], c ="green", linewidths = 1, edgecolor ="pink", s = 10)# plot of scatter plot with x and y values as well as color and size of markers
axes[1][1].set_title("Low Income Countries", size=10)#plot title
axes[1][1].set_xlabel("% of Smokers in Population", size=8)# x label title
axes[1][1].set_ylabel("% Deaths due to Cancer", size=8)# y label title
m, b = np.polyfit(LowIncome_Tobacco_Cancer['2020'], LowIncome_Tobacco_Cancer['Rate_Cancer'], 1)# creates the least squares fit association between columns
axes[1][1].plot(LowIncome_Tobacco_Cancer['2020'], m*(LowIncome_Tobacco_Cancer['2020']) + b, color='red', linewidth = 0.35)# plot of scatter plot with x and y values as well as color and size of markers

plt.tight_layout() # adjusts subplot are so that the subplots fits in to the figure area and don't overlap
plt.show()




# In[446]:


CountryDeath = cause_of_deathrate.loc[cause_of_death['Year'] == 2019] # new dataframe of slice of year 2019

CountryDeath.drop('Year', inplace=True, axis=1) # removes the year columfrom dataframe as dataframe contains data for just one year

CountryDeath.drop('total_deaths', inplace=True, axis=1) # removes total deaths column from dataframe

CountryDeath_max_value = CountryDeath.idxmax(axis=1)# function returns the index of the column in each row with the highest value

x = CountryDeath_max_value.value_counts(sort=True, ascending=False, dropna=True) # tallies the number of fows returned for each column
print(x)
my_labels = ["Cardiovascular Disease", "Cancer", "HIV", "Malaria", "Infant Mortality", "Diarrhea", "Tuberculosis", "Lower Reespiratory Tract Infection"] # list of the titles of the columns returned to create a legend for pie chart

plt.pie(x, shadow=True, startangle=90) # plot pie chart and give it a shadow and start the biggest portion at 90 degrees

plt.legend(my_labels, loc="lower left", bbox_to_anchor =(0.9, 0.25)) # adds legend to chart and positions it so it doesn't obscure pie

plt.title('Leading Cause of Death in Each Country') # assign title of the plot

plt.show() 


# In[447]:





# In[ ]:




