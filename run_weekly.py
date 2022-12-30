##########IMPORTS##########


from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
from webdriver_manager.chrome import ChromeDriverManager

import time
import pandas as pd
import re
from bs4 import BeautifulSoup
import datetime as dt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler


##########FUNCTIONS##########


def retrieve_this_weeks_fights():

    # Instantating constants

    regex_weight = r"\d{3}\sl"
    regex_reach = r'\d{2}"'
    regex_dob = r',\s\d{4}'
    regex_various_stats = r'\d{1}\.\d{1,2}'
    regex_various_stats_2 = r'\d{1,2}%'
    regex_record = r'\d{1,2}-\d{1,2}-\d{1,2}'
    year = dt.date.today().year
    fight_data = pd.DataFrame()
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument("user-data-dir=/Users/hsinger24/Library/Application Support/Google/Chrome/Default1")
    options.add_argument("--start-maximized")
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument("--disable-setuid-sandbox")
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('http://ufcstats.com/statistics/events/completed')
    upcoming_card_data = pd.DataFrame(columns = ['name', 'weight', 'reach', 'age', 'slpm', 'sapm', 'td_avg', 'sub_avg', 'strk_acc', 'strk_def', 'td_acc',
                                                'td_def', 'wins', 'losses'])

    # Getting data for upcoming card

    links_home_page = driver.find_elements(By.TAG_NAME, 'a') 
    del links_home_page[:6] 
    upcoming_card = links_home_page[0] 
    upcoming_card.click() 
    time.sleep(2)
    links_upcoming_card = driver.find_elements(By.TAG_NAME, 'a') 
    del links_upcoming_card[:4]
    for i in range(7):
        del links_upcoming_card[-1]
    for i, link in enumerate(links_upcoming_card):
        if link.text=='View\nMatchup':
            del links_upcoming_card[i]
    num_fighters = len(links_upcoming_card)
    fighter = links_upcoming_card[0]
    fighter.click()
    for i in range(num_fighters):
        if i!=0:
            links_upcoming_card = driver.find_elements(By.TAG_NAME, 'a')
            del links_upcoming_card[:4]
            for j, link in enumerate(links_upcoming_card):
                if link.text=='View\nMatchup':
                    del links_upcoming_card[j]
            fighter= links_upcoming_card[i]
            fighter.click() 
        time.sleep(2)
        html = driver.page_source
        soup = BeautifulSoup(html, 'html.parser')
        name = soup.find_all('span', class_ = 'b-content__title-highlight')[0].text.replace('\n', '').replace(' ', '')
        weight = float(re.findall(regex_weight, soup.prettify())[0].strip('l').replace(' ', ''))
        reach = float(re.findall(regex_reach, soup.prettify())[1].strip('"'))
        dob = int(re.findall(regex_dob, soup.prettify())[0].strip(',').replace(' ', ''))
        age = year - dob
        slpm = float(re.findall(regex_various_stats, soup.prettify())[1])
        sapm = float(re.findall(regex_various_stats, soup.prettify())[2])
        td_avg = float(re.findall(regex_various_stats, soup.prettify())[3])
        sub_avg = float(re.findall(regex_various_stats, soup.prettify())[4])
        strk_acc = float(re.findall(regex_various_stats_2, soup.prettify())[0].strip('%'))
        strk_def = float(re.findall(regex_various_stats_2, soup.prettify())[1].strip('%'))
        td_acc = float(re.findall(regex_various_stats_2, soup.prettify())[2].strip('%'))
        td_def = float(re.findall(regex_various_stats_2, soup.prettify())[3].strip('%'))
        record = re.findall(regex_record, soup.prettify())[0]
        record = record.split('-')
        wins = float(record[0])
        losses = float(record[1])
        list_of_stats = [name, weight, reach, age, slpm, sapm, td_avg, sub_avg, strk_acc, strk_def, td_acc, td_def, wins, losses]
        series = pd.Series(list_of_stats, index = upcoming_card_data.columns)
        upcoming_card_data = upcoming_card_data.append(series, ignore_index = True)
        driver.back()
    evens = upcoming_card_data.iloc[::2]
    evens.reset_index(drop = True, inplace = True)
    odds = upcoming_card_data.iloc[1::2]
    odds.reset_index(drop = True, inplace = True)
    final= pd.merge(evens, odds, left_index = True, right_index = True)
    final.columns = ['fighter_1', 'weight_1', 'reach_1', 'age_1', 'slpm_1', 'sapm_1', 'td_avg_1', 'sub_avg_1', 
                    'strk_acc_1', 'strk_def_1', 'td_acc_1','td_def_1', 'wins_1', 'losses_1', 'fighter_2', 'weight_2', 
                    'reach_2', 'age_2', 'slpm_2', 'sapm_2', 'td_avg_2', 'sub_avg_2', 'strk_acc_2', 'strk_def_2', 
                    'td_acc_2','td_def_2', 'wins_2', 'losses_2']
    final['result'] = -5
    final['SUB_OVR']= 0
    final['KO_OVR'] = 0
    print(final.tail())
    driver.quit()
    return final

def append_fight_data(this_weeks_fights):
    mma_data = pd.read_csv('mma_data.csv', index_col = 0)
    mma_data = mma_data.append(this_weeks_fights)
    mma_data.reset_index(inplace = True, drop = True)
    mma_data.to_csv('mma_data.csv')
    return

def this_weeks_predictions(this_weeks_fights):
    
    # Importing data to train RF
    data = pd.read_csv('mma_data.csv', index_col=0)

    # Filtering out unwanted rows
    data = data[data.slpm_2 + data.sapm_2 != 0]
    data = data[data.slpm_1 + data.sapm_1 != 0]
    data = data[data.result >= 0]

    # Engineering some columns
    data['reach_diff'] = data.reach_1 - data.reach_2
    data['age_diff'] = data.age_1 - data.age_2
    data['slpm_diff'] = data.slpm_1 - data.slpm_2
    data['sapm_diff'] = data.sapm_1 - data.sapm_2
    data['td_acc_diff'] = data.td_acc_1 - data.td_acc_2
    data['td_def_diff'] = data.td_def_1 - data.td_def_2
    data['td_avg_diff'] = data.td_avg_1 - data.td_avg_2
    data['sub_avg_diff'] = data.sub_avg_1 - data.sub_avg_2
    data['strk_acc_diff'] = data.strk_acc_1 - data.strk_acc_2
    data['strk_def_diff'] = data.strk_def_1 - data.strk_def_2
    data['wins_diff'] = data.wins_1 - data.wins_2
    data['losses_diff'] = data.losses_1 - data.losses_2
    data['win_pct_1'] = data.wins_1/(data.losses_1 + data.wins_1)
    data['win_pct_2'] = data.wins_2/(data.losses_2 + data.wins_2)
    data['win_pct_diff'] = data.win_pct_1 - data.win_pct_2

    # Droping unecessary columnns and scaling data
    x_cols = ['reach_diff', 'age_diff', 'slpm_diff', 'sapm_diff', 'td_acc_diff', 'td_def_diff',
                'td_avg_diff', 'sub_avg_diff', 'strk_acc_diff', 'strk_def_diff', 'wins_diff',
                'losses_diff', 'win_pct_diff', 'weight_1', 'age_1']
    y_col = ['result']
    x, y = data[x_cols], data[y_col]

    # Creating parameter grid for RF model
    n_estimators = [int(x) for x in np.linspace(start = 3, stop = 15, num = 13)]
    max_features = [int(x) for x in np.linspace(start = 3, stop = 10, num = 8)]
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]
    param_grid = {
        'n_estimators' : n_estimators,
        'max_features' : max_features,
        'max_depth' : max_depth
    }

    # Running Grid Search
    grid_search = GridSearchCV(RandomForestClassifier(random_state = 0), param_grid, cv = 4)
    grid_search.fit(x, y)
    
    # Saving best model from grid search
    rf = grid_search.best_estimator_

    # Scaling data for LR

    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # Creating parameter grid for LR model
    c = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid = {
        'C' : c
    }

    # Running Grid Search
    grid_search = GridSearchCV(LogisticRegression(random_state = 0, max_iter = 500), param_grid, cv = 4)
    grid_search.fit(x, y)
    
    # Saving best model from grid search
    lr = grid_search.best_estimator_

    # Preparing prediction data & predicting
    x_data_pred = this_weeks_fights[x_cols]
    this_weeks_fights['Prediction_RF'] = rf.predict(x_data_pred)
    this_weeks_fights['Prediction_LR'] = lr.predict_proba(x_data_pred)[:, 1]

    # Saving date and predicted data
    this_weeks_fights['Date'] = dt.date.today()
    this_weeks_fights.to_csv(f'Predictions/predictions_{dt.date.today()}.csv')

    return


##########SCRIPT##########


# Appending this week's fight data to existing dataset
this_weeks_fights = retrieve_this_weeks_fights()
append_fight_data(this_weeks_fights)

# Re-traininng RF model & using it to predict fights
this_weeks_predictions(this_weeks_fights)
    