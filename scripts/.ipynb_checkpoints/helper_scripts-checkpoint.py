import geonamescache
from geopy.geocoders import Nominatim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from geopy.geocoders import Nominatim
import os
import seaborn as sns

from sklearn.utils._testing import ignore_warnings

from sklearn.metrics import confusion_matrix
from sklearn.utils import resample

from sklearn.model_selection import StratifiedShuffleSplit

from sklearn import preprocessing, svm, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression, Lasso
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.utils import resample
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning

pd.options.mode.chained_assignment = None  # default='warn'

#NOTE IMPORTANT
#37114  and 37124 are INVALID LINES IN THE RAW DATA, THEY NEED TO BE MANUALLY DELETED

script_directory = os.getcwd()

def statistics(clf, X_test, y_test, y_pred,text, analytics_dict, y_test_full):
    old_test = y_test_full.loc[(y_test_full.agegroup == 3.0)]
    young_test = y_test_full.loc[((y_test_full.agegroup == 1.0) | (y_test_full.agegroup == 2.0))]

    #Since we need to test the old and the combined group together, we need to recover the indices from the full dataframe (y_test_full)!
    
    #So we take the intersection of indices from the X_test, since it is a dataframe,  and the full dataframe to identify test data that is for the desired agegroup.
    #then we take those indices and intersect them with the target cariable for the prediction and test y values.
    intersec_young = np.nonzero(np.in1d(X_test.index, young_test.index))
    intersec_old = np.nonzero(np.in1d(X_test.index, old_test.index))
    analytics_dict = {}
    TN_y, FP_y, FN_y, TP_y = confusion_matrix(y_test[intersec_young], y_pred[intersec_young]).ravel()
    TN_o, FP_o, FN_o, TP_o = confusion_matrix(y_test[intersec_old], y_pred[intersec_old]).ravel()

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
    display = metrics.RocCurveDisplay.from_estimator( clf, X_test, y_test)
    accuracy_5 = cross_val_score(clf, X_test, y_test.ravel(), scoring='accuracy', cv = 5)

    print('Cross Validate 5-fold : ')
    print(accuracy_5)
    
    accuracy_10 = cross_val_score(clf, X_test, y_test.ravel(), scoring='accuracy', cv = 10)

    print('Cross Validate 5-fold : ')
    print(accuracy_10)
    #display.plot()
    plt.show()
    print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
    analytics_dict = {'accuracy' : accuracy, 'cross_val_5' : accuracy_5, 'cross_val_10' : accuracy_10, 'dp_parity': ((TP_y + FP_y) / (TP_y + FP_y + FN_y + TN_y)) -((TP_o + FP_o) / (TP_o + FP_y + FN_o + TN_o)),
                       'dp_parity_gap': ((TP_y + TN_y) / (TP_y + FP_y + FN_y + TN_y)) -((TP_o + TN_o) / (TP_o + FP_y + FN_o + TN_o)), 'prevalence': (TP+FP)/(TP+FP+TN+FN),
                      'PPR': (TP_y/ (TP_y+FP_y)) - (TP_o/ (TP_o+FP_o)), 'recall' : TP/ (TP+FN), 'TP' : TP,
                      'FP_dif' : (( FP_y) / (TP_y + FP_y + FN_y + TN_y)) -((FP_o) / (TP_o + FP_y + FN_o + TN_o)), 'TP_dif' : (( TP_y) / (TP_y + FP_y + FN_y + TN_y)) -((FP_o) / (TP_o + FP_y + FN_o + TN_o)),
                            'FP': FP, 'TN' : TN, 'FN' : FN, 'FPR' : fpr, 'TPR' : tpr, 'thresholds' : thresholds, 'TP_rate' :  (TP)/(TP+FP+TN+FN),'FP_rate' :  (FP)/(TP+FP+TN+FN), 'AUC' : roc_auc}
    return analytics_dict

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

def statistics(clf, X_test, y_test, y_pred,text, analytics_dict, y_test_full):
    old_test = y_test_full.loc[(y_test_full.agegroup == 3.0)]
    young_test = y_test_full.loc[((y_test_full.agegroup == 1.0) | (y_test_full.agegroup == 2.0))]

    #Since we need to test the old and the combined group together, we need to recover the indices from the full dataframe (y_test_full)!
    
    #So we take the intersection of indices from the X_test, since it is a dataframe,  and the full dataframe to identify test data that is for the desired agegroup.
    #then we take those indices and intersect them with the target cariable for the prediction and test y values.
    intersec_young = np.nonzero(np.in1d(X_test.index, young_test.index))
    intersec_old = np.nonzero(np.in1d(X_test.index, old_test.index))
    analytics_dict = {}
    TN_y, FP_y, FN_y, TP_y = confusion_matrix(y_test[intersec_young], y_pred[intersec_young]).ravel()
    TN_o, FP_o, FN_o, TP_o = confusion_matrix(y_test[intersec_old], y_pred[intersec_old]).ravel()

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
    display = metrics.RocCurveDisplay.from_estimator( clf, X_test, y_test)
    accuracy_5 = cross_val_score(clf, X_test, y_test.ravel(), scoring='accuracy', cv = 5)

    print('Cross Validate 5-fold : ')
    print(accuracy_5)
    
    accuracy_10 = cross_val_score(clf, X_test, y_test.ravel(), scoring='accuracy', cv = 10)

    print('Cross Validate 5-fold : ')
    print(accuracy_10)
    #display.plot()
    plt.show()
    print('Accuracy of the binary classifier = {:0.3f}'.format(accuracy))
    analytics_dict = {'accuracy' : accuracy, 'cross_val_5' : accuracy_5, 'cross_val_10' : accuracy_10, 'dp_parity': ((TP_y + FP_y) / (TP_y + FP_y + FN_y + TN_y)) -((TP_o + FP_o) / (TP_o + FP_y + FN_o + TN_o)),
                       'dp_parity_gap': ((TP_y + TN_y) / (TP_y + FP_y + FN_y + TN_y)) -((TP_o + TN_o) / (TP_o + FP_y + FN_o + TN_o)), 'prevalence': (TP+FP)/(TP+FP+TN+FN),
                      'PPR': (TP_y/ (TP_y+FP_y)) - (TP_o/ (TP_o+FP_o)), 'recall' : TP/ (TP+FN), 'TP' : TP,
                      'FP_dif' : (( FP_y) / (TP_y + FP_y + FN_y + TN_y)) -((FP_o) / (TP_o + FP_y + FN_o + TN_o)), 'TP_dif' : (( TP_y) / (TP_y + FP_y + FN_y + TN_y)) -((FP_o) / (TP_o + FP_y + FN_o + TN_o)),
                            'FP': FP, 'TN' : TN, 'FN' : FN, 'FPR' : fpr, 'TPR' : tpr, 'thresholds' : thresholds, 'TP_rate' :  (TP)/(TP+FP+TN+FN),'FP_rate' :  (FP)/(TP+FP+TN+FN), 'AUC' : roc_auc}
    return analytics_dict
    

@ignore_warnings(category=ConvergenceWarning)
def run_ML(df, remove_age = False, perc = 0.3, eq_base_rate = False, hi_base_rate = False, balanced_outcomes = False):
    """
    Args:
    
    remove_age: boolean or integer, default boolean False. Which age group do we remove
        1 - young
        2- middle
        3- old
    perc: perentage of rate to reove
    Returns:
    """
    analytics_dict = {}
    label_encoder = preprocessing.LabelEncoder()
    hot_encoder = preprocessing.OneHotEncoder()
    
    #Change to baserate difference, change to percentage of group
    #read under and oversample regime
    """df_majority_downsampled = resample(df_majority, 
                                     replace=True,     
                                     n_samples=len(df_minority),    
                                     random_state=42) """


    #df_resampled = pd.concat([df_minority, df_majority_downsampled])
    df_resampled = df
    X = df_resampled

    
    transformed = pd.get_dummies(X.occupation)
    X = pd.concat([X, transformed], axis=1).drop(['occupation'], axis=1)    
    
    transformed = pd.get_dummies(X.state)
    X = pd.concat([X, transformed], axis=1).drop(['state'], axis=1)
    
    transformed = pd.get_dummies(X.city)
    X = pd.concat([X, transformed], axis=1).drop(['city'], axis=1)
    
    transformed = pd.get_dummies(X.template)
    X = pd.concat([X, transformed], axis=1).drop(['template'], axis=1)
    X.employment = X['employment'].replace({'Unemployed': 0, 'Employed': 1})
    X.skill = X['skill'].replace({'High': 1, 'Low': 0})
    
    
    X.dropna(inplace = True)
    
    #y = np.array(X['callback']).reshape(-1, 1).astype('int')
    y = X['callback']

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=0)
    sss.get_n_splits(X, y)
    
    for train_index, test_index in sss.split(X, y):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        
    y_train = np.array(y_train).reshape(-1, 1).astype('int')
    y_test = np.array(y_test).reshape(-1, 1).astype('int')

    #X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)
    
    #Fewer Young/Middle: Randomly select young or middle people and delete them from the 80% -- remove different percentages, e.g., 10%/30%/70%/100% 
    if remove_age is not False:
        dropIndices1 = np.random.choice(X_train.loc[(X_train.agegroup == remove_age[0]) | (X_train.agegroup == remove_age[1])].index.tolist(), size = int(X_train.shape[0]*perc))
        X_train.drop(dropIndices1)
        
    #Equalize Base Rate: Randomly delete old people that did not get a callback so that the callback rate of young/middle == callback rate of old in the 80% of the data 
    if eq_base_rate is True:
        #quickhack
        callbacks_young = len(X_train.loc[((X_train['agegroup'] == 2.0) | (X_train['agegroup'] == 1.0)) & (X_train['callback'] == 1.0)])
        base_rate_young = callbacks_young / len(X_train.loc[((X_train['agegroup'] == 2.0) | (X_train['agegroup'] == 1.0))])

        callbacks_old = len(X_train.loc[((X_train['agegroup'] == 3.0)) & (X_train['callback'] == 1.0)])
        base_rate_old = callbacks_old / len(X_train.loc[(X_train['agegroup'] == 3.0)])

        dropIndices1 = np.random.choice(X_train.loc[(X_train['agegroup'] == 3.0) & (X_train['callback'] == 0.0)].index.tolist(), size = int(abs(base_rate_young -base_rate_old * 100)))
        X_train.drop(dropIndices1)
    
    #Higher Base Rate Difference: Randomly delete young/middle people that did not get a call back in 80% of the data – again at different percentages 10%/30%/70%/100% 
    if hi_base_rate is True:
        dropIndices1 = np.random.choice(X_train.loc[((X_train['agegroup'] == 1.00) | (X_train['agegroup'] == 2.00)) & (X_train['callback'] == 1.0)].index.tolist(), size = int(X_train.shape[0]*perc))
        X_train.drop(dropIndices1)
    
    if balanced_outcomes is True:
        #Balanced Outcomes: Find the group with the lowest total call backs and then down sample all groups so that the number of callbacks and the number of non-callbacks is the same
        callbacks_old = len(X_train.loc[((X_train['agegroup'] == 3.0)) & (X_train['callback'] == 1.0)])
        callbacks_middle = len(X_train.loc[((X_train['agegroup'] == 2.0)) & (X_train['callback'] == 1.0)])
        callbacks_young = len(X_train.loc[((X_train['agegroup'] == 1.0)) & (X_train['callback'] == 1.0)])
        
        base_rate_young = callbacks_young / len(X_train.loc[(X_train['agegroup'] == 1.0)])
        base_rate_middle = callbacks_middle / len(X_train.loc[(X_train['agegroup'] == 2.0)])

        no_callbacks_min = len(X_train.loc[((X_train['agegroup'] == 3.0)) & (X_train['callback'] == 0.0)])  
        base_rate_old = callbacks_old / len(X_train.loc[(X_train['agegroup'] == 3.0)])

        #drop extra for middle group
        dropIndices1 = np.random.choice(X_train.loc[(X_train['agegroup'] == 2.0) & (X_train['callback'] == 0.0)].index.tolist(), size = int(abs(base_rate_middle -base_rate_old * 100)))
        X_train.drop(dropIndices1)
        dropIndices1 = np.random.choice(X_train.loc[(X_train['agegroup'] == 2.0) & (X_train['callback'] == 1.0)].index.tolist(), size = int(abs(base_rate_middle -base_rate_old * 100)))
        X_train.drop(dropIndices1)
        
        #drop extra for young group
        dropIndices1 = np.random.choice(X_train.loc[(X_train['agegroup'] == 1.0) & (X_train['callback'] == 0.0)].index.tolist(), size = int(abs(base_rate_young -base_rate_old * 100)))
        X_train.drop(dropIndices1)
        dropIndices1 = np.random.choice(X_train.loc[(X_train['agegroup'] == 1.0) & (X_train['callback'] == 1.0)].index.tolist(), size = int(abs(base_rate_young -base_rate_old * 100)))
        X_train.drop(dropIndices1)

    X_train = X_train.drop(columns=['callback', 'adtitle'])    
    X_test = X_test.drop(columns=['callback', 'adtitle'])   
    
    orig_X = X_test.copy()

    X_train.drop(['agegroup'], axis=1)
    X_test.drop(['agegroup'], axis=1)
    
    regr = LogisticRegression(max_iter=300)
    clf_LR = regr.fit(X_train, y_train.ravel())
    y_pred_regr = regr.predict(X_test)
    coeffs_LR = clf_LR.coef_
    analytics_dict['LR '] = statistics(clf_LR, X_test, y_test, y_pred_regr, 'Logistic Regression', analytics_dict,orig_X)
    #print("Coefficients for Logistic Regression:")
    #print(coeffs_LR)

    rf = RandomForestClassifier()
    clf_RF = rf.fit(X_train, y_train.ravel())
    y_pred_rf = rf.predict(X_test)
    analytics_dict['RFR'] = statistics(clf_RF, X_test, y_test, y_pred_rf, 'Random Forest', analytics_dict, orig_X)
    
    MLP = MLPClassifier(hidden_layer_sizes=(150,100,50), max_iter=300,activation = 'relu',solver='adam',random_state=1)
    clf_MLP = MLP.fit(X_train, y_train.ravel())
    y_pred_MLP = MLP.predict(X_test)
    analytics_dict['MLP'] = statistics(clf_MLP, X_test, y_test, y_pred_MLP, 'MLP', analytics_dict, orig_X)
    
    return analytics_dict

    
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
    
    
    return [bach.loc[bach['county2'] == state][year].values,    high_school.loc[high_school['county2'] == state][year].values]
            
            
def analytics(analytics_dict):
    """demographic parity ; receiving positive outcomes at the same rate,
    the demographic parity gap; difference in the two groups parities, 
    the accuracy parity ; the division of accurate outcomes between the two groups, 
    equalized odds ; that equal true positive and false positive rates between the two different groups match,
    predictive rate parity; whether a classifiers precision rates are equivalent for subgroups under consideration.
    """
    keys_t = ''
    keys = ''                                                                                                                   
    dp_str = ''
    for key_t in analytics_dict.keys():
        keys_t += '\t' + key_t
        model_dict = analytics_dict[key_t]
        ti_k = '\t '

        for key in model_dict.keys():
            ti_k += f'{key_t} {key} {key_t} \t'
            dp_str +=  f' {key_t} {key}: {np.round(model_dict[key]["dp_parity"], 3)},\t'
        dp_str+= '\n'
    print('Demographic parities')
    print('Receiving positive outcomes at the same rate (difference in the two groups parities) -- Equal Positive Prediction Rates ')
    print(keys)
    print(dp_str)
        
    
    dp_str_gp = ''
    for key_t in analytics_dict.keys():       
        model_dict = analytics_dict[key_t]

        for key in model_dict.keys():
            ti_k += f'{key_t} {key} {key_t} \t'
            dp_str_gp +=  f'{key_t} {key}: {np.round(model_dict[key]["dp_parity_gap"], 3)}: \t'
        dp_str_gp += '\n'
    print('\nDemographic parities gap for', key_t)
    print('The division of accurate outcomes between the two groups')
    print('\t', keys)
    print(dp_str_gp)

    dp_str_TP = ''
    dp_str_FP = ''
    for key_t in analytics_dict.keys():
        for key in model_dict.keys():
            dp_str_TP += f'TP {key_t} {key}: {np.round(model_dict[key]["TP_dif"], 3)} , '
            dp_str_FP += f'FP {key_t} {key}: {np.round(model_dict[key]["FP_dif"], 3)} , '
        dp_str_TP += '\n'
        dp_str_FP += '\n'
    print('\nEqualised Odds')
    print('Equal true positive and false positive rates between the two different groups match')
    print('\t', keys)
    print(dp_str_TP, '\n', dp_str_FP)
    
    dp_str_precision = ''
    for key_t in analytics_dict.keys():
        model_dict = analytics_dict[key_t]

        for key in model_dict.keys():
            dp_str_precision += f'{key} {key_t}: {np.round(model_dict[key]["PPR"], 3)}'
    print('\nPredictive Rate Parity')
    print('Whether a classifiers precision rates are equivalent for subgroups under consideration. ')
    print('\t', keys)
    print(dp_str_precision)
        
    print('\nROC AUC')
    roc_auc = ''
    for key_t in analytics_dict.keys():
        prev = analytics_dict[key_t]
        for key1 in prev.keys():
            roc_auc += f'{key_t}, {key1}: {np.round(prev[key1]["AUC"], 3)}\t'
        roc_auc += '\n'
    print(roc_auc)
            
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