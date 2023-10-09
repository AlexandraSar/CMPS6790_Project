import geonamescache
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import os
#NOTE IMPORTANT
#37114  and 37124 are INVALID LINES IN THE RAW DATA, THEY NEED TO BE MANUALLY DELETED

script_directory = os.getcwd()
def get_states(lon, lat):
    ''' Input two 1D lists of floats/ints '''
    geolocator = Nominatim(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/116.0.0.0 Safari/537.36")

    # a list of states 
    # use a coordinate tool from the geopy library
    # get the state name

    location = geolocator.reverse("%s, %s" % (lat, lon), timeout=10)
    state = location.raw['address']['state']
    return state

def get_unemployment_state(state, years):
    unemp_data = pd.read_excel(script_directory + '\\Data\\RAW\\ststdsadata.xlsx')

    unemp_row = unemp_data.loc[unemp_data['State and area'] ==state ]
    return unemp_row.loc[unemp_data['Year'].isin( years)]  



def set_state():
    data =pd.read_stata(script_directory + '\\Data\\RAW\\mergefinal.dta')
    data['state'] = ''
    gc = geonamescache.GeonamesCache()
    c = gc.get_cities()
    
    US_cities = {c[key]['name'].lower() : c[key] for key in list(c.keys())
                 if c[key]['countrycode'] == 'US'}
    city_set = set(data['city'])
    for line in city_set:
        if line in US_cities.keys():
            get_states(US_cities[line]['longitude'], US_cities[line]['latitude'])
            state = get_states(US_cities[line]['longitude'], US_cities[line]['latitude'])
            data.loc[data['city'] == line.lower() , 'state'] = state
    data.loc[data['city'] ==  'new york' , 'state'] = 'New York' #We need to add this manually, as for some reason geonamescache wont find it
    data.to_csv(script_directory + '\\Data\\unemployment_with_state.csv', index=False)
    return data



def gen_unemployment_csv():
    """ Generates unemployment_percentage_state.csv, creates summary statistics got Employed populations from the resume applications per field for each state.
    """
    resume_data = pd.read_csv(script_directory + '\\Data\\unemployment_with_state.csv')
    states = list(set(resume_data.state))

    df = pd.DataFrame(columns=['State', 'Unemployment Percentage', 'Admin Employed', 'Admin Unemployed', 'Security Employed', 'Security Unemployed', 'Janitor Employed','Janitor Unemployed', 'Sales Employed', 'Sales Unemployed'])
    #add the AP statistics 
    occupations = ('admin', 'security', 'janitor', 'sales')
    #low_skill_unemp = pd.read_csv('..\\Data\\CLEANED_High School Graduates, No College, 25 yrs.csv')
    for state in states:
        unemp = get_unemployment_state(state, [2019]) #Since the study is from 2019
        perc = unemp['Unemployment Rate Percent'].mean()
        for skill in ('High', 'Low'):            
            d = [f'{state}, {skill}', perc]

            for emp in ( 'Employed', 'Unemployed'):
                for occupation in occupations:
                    applications = resume_data.loc[(resume_data['state'] == state) & (resume_data['employment'] == emp) & (resume_data['occupation'] == occupation) & (resume_data['skill'] == skill)]
                    tot_height = applications['callback'].value_counts(normalize=True) * 100
                    if len(tot_height) ==1:
                        d.append(0.0)
                    else:
                        d.append(tot_height.iloc[1]) #append for every occupation their callback rate
            df.loc[len(df)] =d
    df.to_csv(script_directory + '\\Data\\unemployment_percentage_state.csv', index=False)
    return df