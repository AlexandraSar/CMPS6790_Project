import geonamescache
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import os
import seaborn as sns

from sklearn import preprocessing, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
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

#data = set_state() #Calls to parse and clean the mergefinal and adds the state using geopy
#data
def statistics(clf, X_test, y_test, y_pred,text):
    TN, FP, FN, TP = confusion_matrix(y_test, y_pred).ravel()
    print(f'\nStatistics for {text}:')
    print('True Positive(TP)  = ', TP)
    print('False Positive(FP) = ', FP)
    print('True Negative(TN)  = ', TN)
    print('False Negative(FN) = ', FN)

    accuracy =  (TP + TN) / (TP + FP + TN + FN) 
    fpr, tpr, thresholds = metrics.roc_curve(y_test, y_pred)
    roc_auc = metrics.auc(fpr, tpr)
    #display = metrics.RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc,
    #                               estimator_name=text)
    displayy = metrics.RocCurveDisplay.from_estimator( clf, X_test, y_test)

    #display.plot()
    plt.show()
    print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))

def run_ML(df_majority, df_minority):
    label_encoder = preprocessing.LabelEncoder()
    hot_encoder = preprocessing.OneHotEncoder()


    df_majority_downsampled = resample(df_majority, 
                                     replace=True,     
                                     n_samples=len(df_minority),    
                                     random_state=42) 


    df_resampled = pd.concat([df_minority, df_majority_downsampled])
    
    y = np.array(df_resampled['callback']).reshape(-1, 1).astype('int')
    

    X = df_resampled.drop(columns=['callback', 'adtitle'])
    X.employment = X['employment'].replace({'Unemployed': 0, 'Employed': 1})
    X.skill = X['skill'].replace({'High': 1, 'Low': 0})    
    transformed = pd.get_dummies(X.occupation)
    X = pd.concat([X, transformed], axis=1).drop(['occupation'], axis=1)    
    
    transformed = pd.get_dummies(X.state)
    X = pd.concat([X, transformed], axis=1).drop(['state'], axis=1)
    
    transformed = pd.get_dummies(X.city)
    X = pd.concat([X, transformed], axis=1).drop(['city'], axis=1)
    
    transformed = pd.get_dummies(X.template)
    X = pd.concat([X, transformed], axis=1).drop(['template'], axis=1)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)


    regr = LogisticRegression(max_iter=500)
    clf_LR = regr.fit(X_train, y_train.ravel())
    y_pred_regr = regr.predict(X_test)
        
    statistics(clf_LR, X_test, y_test, y_pred_regr, 'Logistic Regression')
    
    rf = RandomForestClassifier()
    clf_RF = rf.fit(X_train, y_train.ravel())
    y_pred_rf = rf.predict(X_test)
    statistics(clf_RF, X_test, y_test, y_pred_rf, 'Random Forest')
    
    MLP = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    clf_MLP = MLP.fit(X_train, y_train.ravel())
    y_pred_MLP = MLP.predict(X_test)
    statistics(clf_MLP, X_test, y_test, y_pred_MLP, 'MLP')
    
def plot_confusion_matrix(data, labels):
    seaborn.set(color_codes=True)
    plt.figure(1, figsize=(9, 6))
 
    plt.title("Confusion Matrix")
    seaborn.set(font_scale=1.4)
    
    ax = seaborn.heatmap(data, annot=True, cmap="YlGnBu", cbar_kws={'label': 'Scale'})
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set(ylabel="True Label", xlabel="Predicted Label")
 
def random_sample_adtitles(df, number_of_samples = 1):
    df.dropna(inplace = True)

    adtitles = set(df['adtitle'])
    random_sample_adtitles_df = pd.DataFrame(columns = df.columns)

    #I am sure there is some function for this in pandas
    for val in adtitles:
        all_ads = df.loc[df.adtitle == val]
        if len(all_ads) > 0:
            for v in range(number_of_samples):
                
                #Note we have three samples for each ad, so dont bother use it for more than that
                if v != 0 and number_of_samples > 1:
                    included = set()
                    if all_ads.iloc[random_val]['callback'] == 1:
                        next_val = 0
                    else:
                        next_val = 1
                    while int(all_ads.iloc[random_val]['callback']) != next_val and len(included) < len(all_ads):
                        random_val = np.random.choice(len(all_ads))
                        included.add(random_val)
                        random_sample_adtitles_df.loc[-1] = all_ads.iloc[random_val]
                else:
                    random_val = np.random.choice(len(all_ads))
                    random_sample_adtitles_df.loc[-1] = all_ads.iloc[random_val]
                random_sample_adtitles_df.index = random_sample_adtitles_df.index + 1
                random_sample_adtitles_df = random_sample_adtitles_df.sort_index()

    return random_sample_adtitles_df


def set_state(data):
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

def get_education(state, year):
    bach = pd.read_csv(script_directory + "\Data\\CLEANED_EducationReport_college.csv")
    high_school = pd.read_csv(script_directory + "\Data\\CLEANED_EducationReport_highschool.csv")
    some_college = pd.read_csv(script_directory + "\Data\\CLEANED_EducationReport_some_college.csv")
    
    
    return [bach.loc[bach['county2'] == state][year].values,
    high_school.loc[high_school['county2'] == state][year].values,
    some_college.loc[some_college['county2'] == state][year].values]

def gen_unemployment_csv():
    resume_data = pd.read_csv(script_directory + '\\Data\\unemployment_with_state.csv')

    states = list(set(resume_data.state))


    df = pd.DataFrame(columns=['State', 'Unemployment Percentage', 'Admin Employed', 'Admin Unemployed', 'Security Employed', 'Security Unemployed', 'Janitor Employed','Janitor Unemployed', 'Sales Employed', 'Sales Unemployed'])

    #add the AP statistics 
    occupations = ('admin', 'security', 'janitor', 'sales')
    #low_skill_unemp = pd.read_csv('..\\Data\\CLEANED_High School Graduates, No College, 25 yrs.csv')
    for state in states:
        unemp = get_unemployment_state(state, [2015]) #Since the study is from 2019
        perc = unemp['Unemployment Rate Percent'].mean()
        for skill in ('High', 'Low'):            
            d = [f'{state}, {skill}', perc]

            for emp in ('Unemployed', 'Employed'):
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


def clean_data(data):
    """ Generates unemployment_percentage_state.csv, creates summary statistics got Employed populations from the resume applications per field for each state.
    """
    data = set_state(data)
    data = data.iloc[:40279] #Since above printout shows that the two last rows are nonsense.
    #since 37114  and 37124 are INVALID LINES IN THE RAW DATA, THEY NEED TO BE MANUALLY DELETED
    data = data.drop(labels=37124, axis=0)
    data = data.drop(labels=37114, axis=0)
    # columns need to be transformed from 4 columns ('license', 'certificate', 'spanish', 'computer'), into two columns to encompass 'extra qualifications' ( 'license , certificate, computer') and 'spanish'. Does this make sense?
    data['extra_qualifs'] = data.apply(lambda row: 1 if (row['liscense'] == 1 or  row['computer'] == 1 or row['certificate'] == 1) else 0, axis=1)
    #data['college'] = data['college'].apply(lambda row: 0 if np.nan(row) else 1)
    data_cleaned = data[['state', 'zipcode','city','agegroup', 'gender', 'employment', 'occupation','template',
                         'spanish','extra_qualifs', 'internshipflag','customerservice','cpr','techskills','wpm','grammar','college',
                         'employeemonth','volunteer', 'skill', 'callback', 'adtitle']]

    #We clean up the ad titles, do things like remove items in paranthesis and special characters 
    data_cleaned.adtitle = data_cleaned.adtitle.replace(r'(Ê|@|>|#|:|;|-|={1,}|[\t]+$|\*|\$|\+|\/|,|\(.*\)|\d.hr+|\d.k+|\.|\d|\!|~{1,}|\^{1,}|_{1,})',' ', regex=True).str.lower()
    data_cleaned.adtitle = data_cleaned.adtitle.replace(r'(\s{2,}|\t)', ' ', regex=True) #After the above we might have double or more spaces, so we remove them. We have to do this as if we dont add the space above, we end up titles with words stuck together
    data_cleaned.adtitle = data_cleaned.adtitle.replace(r'(\s+$|\s+^)', '', regex=True) #remove trailing and starting spaces
    #data_cleaned.employment = data_cleaned['employment'].replace({'Unemployed': 0, 'Employed': 1})
    #data_cleaned.skill = data_cleaned['skill'].replace({'High': 1, 'Low': 0})
    data_cleaned.gender = data_cleaned['gender'].replace({'Female': 1, 'Male': 0})
    #Add some needed 
    data_cleaned['hs_diploma_state'] = ''
    data_cleaned['college_diploma_state'] = ''
    data_cleaned['some_college_state'] = ''
    data_cleaned['unemployment_perc'] = ''

    data_cleaned['ADL Minimum Firm Size'] = ''
    data_cleaned['ADL Larger Damages than ADEA'] = ''
    data_cleaned['DDL Minimum Firm Size'] = ''
    data_cleaned['DDL Larger Damages than ADA'] = ''
    data_cleaned['DDL Broader Definition of Disability'] = ''

    #Add unemployment and educational levels for each row
    for state in set(data_cleaned['state']):
        unemploy = get_unemployment_state(state,[2015])
        stats = get_education(state, '2012')

        data_cleaned['hs_diploma_state'] = data_cleaned['hs_diploma_state'].mask(data_cleaned["state"] == state, stats[1])
        data_cleaned['some_college_state'] = data_cleaned['some_college_state'].mask(data_cleaned["state"] == state, stats[2])
        data_cleaned['college_diploma_state'] = data_cleaned['college_diploma_state'].mask(data_cleaned["state"] == state, stats[0])
        data_cleaned['unemployment_perc'] = data_cleaned['unemployment_perc'].mask(data_cleaned["state"] == state, unemploy['Unemployment Rate Percent'].values[0])
        
        adl = get_ADL_DDL(state)
        data_cleaned['ADL Minimum Firm Size'] = data_cleaned['ADL Minimum Firm Size'].mask(data_cleaned["state"] == state, adl['ADL Minimum Firm Size'].values[0])
        data_cleaned['ADL Larger Damages than ADEA'] = data_cleaned['ADL Larger Damages than ADEA'].mask(data_cleaned["state"] == state, adl['ADL Larger Damages than ADEA'].values[0])
        data_cleaned['DDL Minimum Firm Size'] = data_cleaned['DDL Minimum Firm Size'].mask(data_cleaned["state"] == state, adl['DDL Minimum Firm Size'].values[0])
        data_cleaned['DDL Larger Damages than ADA'] = data_cleaned['DDL Larger Damages than ADA'].mask(data_cleaned["state"] == state, adl['DDL Larger Damages than ADA'].values[0])
        data_cleaned['DDL Broader Definition of Disability'] = data_cleaned['DDL Broader Definition of Disability'].mask(data_cleaned["state"] == state, adl['DDL Broader Definition of Disability'].values[0])

    
    data_cleaned.to_csv(script_directory + '\\Data\\unemployment_with_state.csv', index=False)

    return data_cleaned

def get_ADL_DDL(state):
    data = pd.read_csv(script_directory + '\Data\StateLawsAge.csv')
    state_data = data.loc[data['State'] == state]
    return state_data