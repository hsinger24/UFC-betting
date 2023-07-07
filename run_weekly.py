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
from fuzzywuzzy import fuzz
from bs4 import BeautifulSoup
import datetime as dt
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

import gym
from stable_baselines3 import PPO
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

import email, smtplib, ssl
from email import encoders
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

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
        try:
            weight = float(re.findall(regex_weight, soup.prettify())[0].strip('l').replace(' ', ''))
            reach = float(re.findall(regex_reach, soup.prettify())[1].strip('"'))
        except:
            weight = 0
            reach = 0
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
    final = final[(final.weight_1 != 0) & (final.weight_2 != 0)]
    print(final.tail())
    driver.quit()
    return final

def append_fight_data(this_weeks_fights):
    mma_data = pd.read_csv('mma_data.csv', index_col = 0)
    mma_data = mma_data.append(this_weeks_fights)
    mma_data.reset_index(inplace = True, drop = True)
    mma_data.to_csv('mma_data.csv')
    return

def ml_data_prep(target):

    # Importing data to train RF
    data = pd.read_csv('mma_data.csv', index_col=0)

    # Filtering out unwanted rows
    data = data[data.slpm_2 + data.sapm_2 != 0]
    data = data[data.slpm_1 + data.sapm_1 != 0]
    data = data[data.result >= 0]

    # Engineering some columns
    data['strike_diff_1'] = data.slpm_1 - data.sapm_1
    data['strike_diff_2'] = data.slpm_2 - data.sapm_2
    data['strike_diff'] = data.strike_diff_1 - data.strike_diff_2
    data['td_diff_1'] = data.td_acc_1 - data.td_def_1
    data['td_diff_2'] = data.td_acc_2 - data.td_def_2
    data['td_diff'] = data.td_diff_1 - data.td_diff_2
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
                'losses_diff', 'win_pct_diff', 'weight_1', 'age_1', 'strike_diff', 'td_diff']
    y_col = target

    x, y = data[x_cols], data[y_col]
    y = y.values.ravel()

    return x, y, x_cols

def create_grid_search(model, param_grid, x, y, cv = 10):
    # Running Grid Search
    grid_search = GridSearchCV(model, param_grid, cv = cv)
    grid_search.fit(x, y)
    
    # Outputting results
    best_model = grid_search.best_estimator_
    
    return best_model

def this_weeks_predictions(this_weeks_fights):
    
    # Getting x and y for models
    x, y, x_cols = ml_data_prep(target = 'result')
    x_scaled = StandardScaler().fit_transform(x)
    # x_ko, y_ko, x_cols = ml_data_prep(target = 'KO_OVR')
    # x_ko_scaled = StandardScaler().fit_transform(x_ko)
    # x_sub, y_sub, x_cols = ml_data_prep(target = 'SUB_OVR')
    # x_sub_scaled = StandardScaler().fit_transform(x_sub)

    # Prep grid searches
    # RF
    n_estimators = [int(x) for x in np.linspace(start = 3, stop = 15, num = 13)]
    max_features = [int(x) for x in np.linspace(start = 3, stop = 10, num = 8)]
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]
    param_grid_rf = {
        'n_estimators' : n_estimators,
        'max_features' : max_features,
        'max_depth' : max_depth
    }
    # GB
    n_estimators = [int(x) for x in np.linspace(start = 3, stop = 15, num = 13)]
    max_features = [int(x) for x in np.linspace(start = 3, stop = 10, num = 8)]
    max_depth = [int(x) for x in np.linspace(start = 1, stop = 10, num = 10)]
    param_grid_gb = {
        'n_estimators' : n_estimators,
        'max_features' : max_features,
        'max_depth' : max_depth
    }
    # LR
    c = [0.001, 0.01, 0.1, 1, 10, 100]
    param_grid_lr = {
        'C' : c
    }
    # LGBM
    max_iter = [int(x) for x in np.linspace(start = 5, stop = 15, num = 11)]
    max_leaf_nodes = [int(x) for x in np.linspace(start = 4, stop = 10, num = 7)]
    max_depth = [int(x) for x in np.linspace(start = 4, stop = 10, num = 7)]
    learning_rate = [0.001, 0.01, 0.1, 1]
    param_grid_lgbm = {
        'max_iter' : max_iter,
        'max_leaf_nodes' : max_leaf_nodes,
        'max_depth' : max_depth,
        'learning_rate' : learning_rate
    }
    
    # Saving best winner models from grid searches
    rf_winner = create_grid_search(RandomForestClassifier(random_state = 0, class_weight = 'balanced'), param_grid_rf, cv = 10, x = x, y = y)
    gb_winner = create_grid_search(GradientBoostingClassifier(random_state = 0), param_grid_gb, cv = 10, x = x, y = y)
    lgbm_winner = create_grid_search(HistGradientBoostingClassifier(random_state = 0), param_grid_lgbm, cv = 10, x = x, y = y)
    # lr_winner = create_grid_search(LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter = 500), param_grid_lr, cv = 10, x = x_scaled, y = y)
    # rf_ko = create_grid_search(RandomForestClassifier(random_state = 0, class_weight = 'balanced'), param_grid_rf, cv = 10, x = x_ko, y = y_ko)
    # gb_ko = create_grid_search(GradientBoostingClassifier(random_state = 0), param_grid_gb, cv = 10, x = x_ko, y = y_ko)
    # lr_ko = create_grid_search(LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter = 500), param_grid_lr, cv = 10, x = x_ko_scaled, y = y_ko)
    # rf_sub = create_grid_search(RandomForestClassifier(random_state = 0, class_weight = 'balanced'), param_grid_rf, cv = 10, x = x_sub, y = y_sub)
    # gb_sub = create_grid_search(GradientBoostingClassifier(random_state = 0), param_grid_gb, cv = 10, x = x_sub, y = y_sub)
    # lr_sub = create_grid_search(LogisticRegression(random_state = 0, class_weight = 'balanced', max_iter = 500), param_grid_lr, cv = 10, x = x_sub_scaled, y = y_sub)

    # Filtering out fights with UFC newcomers
    this_weeks_fights = this_weeks_fights[this_weeks_fights.slpm_2 + this_weeks_fights.sapm_2 != 0]
    this_weeks_fights = this_weeks_fights[this_weeks_fights.slpm_1 + this_weeks_fights.sapm_1 != 0]

    # Preparing prediction data & predicting
    this_weeks_fights['strike_diff_1'] = this_weeks_fights.slpm_1 - this_weeks_fights.sapm_1
    this_weeks_fights['strike_diff_2'] = this_weeks_fights.slpm_2 - this_weeks_fights.sapm_2
    this_weeks_fights['strike_diff'] = this_weeks_fights.strike_diff_1 - this_weeks_fights.strike_diff_2
    this_weeks_fights['td_diff_1'] = this_weeks_fights.td_acc_1 - this_weeks_fights.td_def_1
    this_weeks_fights['td_diff_2'] = this_weeks_fights.td_acc_2 - this_weeks_fights.td_def_2
    this_weeks_fights['td_diff'] = this_weeks_fights.td_diff_1 - this_weeks_fights.td_diff_2
    this_weeks_fights['reach_diff'] = this_weeks_fights.reach_1 - this_weeks_fights.reach_2
    this_weeks_fights['age_diff'] = this_weeks_fights.age_1 - this_weeks_fights.age_2
    this_weeks_fights['slpm_diff'] = this_weeks_fights.slpm_1 - this_weeks_fights.slpm_2
    this_weeks_fights['sapm_diff'] = this_weeks_fights.sapm_1 - this_weeks_fights.sapm_2
    this_weeks_fights['td_acc_diff'] = this_weeks_fights.td_acc_1 - this_weeks_fights.td_acc_2
    this_weeks_fights['td_def_diff'] = this_weeks_fights.td_def_1 - this_weeks_fights.td_def_2
    this_weeks_fights['td_avg_diff'] = this_weeks_fights.td_avg_1 - this_weeks_fights.td_avg_2
    this_weeks_fights['sub_avg_diff'] = this_weeks_fights.sub_avg_1 - this_weeks_fights.sub_avg_2
    this_weeks_fights['strk_acc_diff'] = this_weeks_fights.strk_acc_1 - this_weeks_fights.strk_acc_2
    this_weeks_fights['strk_def_diff'] = this_weeks_fights.strk_def_1 - this_weeks_fights.strk_def_2
    this_weeks_fights['wins_diff'] = this_weeks_fights.wins_1 - this_weeks_fights.wins_2
    this_weeks_fights['losses_diff'] = this_weeks_fights.losses_1 - this_weeks_fights.losses_2
    this_weeks_fights['win_pct_1'] = this_weeks_fights.wins_1/(this_weeks_fights.losses_1 + this_weeks_fights.wins_1)
    this_weeks_fights['win_pct_2'] = this_weeks_fights.wins_2/(this_weeks_fights.losses_2 + this_weeks_fights.wins_2)
    this_weeks_fights['win_pct_diff'] = this_weeks_fights.win_pct_1 - this_weeks_fights.win_pct_2
    
    x_data_pred = this_weeks_fights[x_cols]

    this_weeks_fights['Prediction_RF_Winner'] = rf_winner.predict_proba(x_data_pred)[:, 1]
    this_weeks_fights['Prediction_GB_Winner'] = gb_winner.predict_proba(x_data_pred)[:, 1]
    this_weeks_fights['Prediction_LGBM_Winner'] = lgbm_winner.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_LR_Winner'] = lr_winner.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_RF_SUB'] = rf_sub.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_GB_SUB'] = gb_sub.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_LR_SUB'] = lr_sub.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_RF_KO'] = rf_ko.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_GB_KO'] = gb_ko.predict_proba(x_data_pred)[:, 1]
    # this_weeks_fights['Prediction_LR_KO'] = lr_ko.predict_proba(x_data_pred)[:, 1]

    # Saving date and predicted data
    this_weeks_fights['Date'] = dt.date.today()

    return this_weeks_fights

def append_predictions(this_weeks_predictions):
    predictions = pd.read_csv('mma_data_predictions.csv', index_col = 0)
    predictions = predictions.append(this_weeks_predictions)
    predictions.reset_index(inplace = True, drop = True)
    predictions.to_csv('mma_data_predictions.csv')
    return

def calculate_odds(odds):
    if odds<0:
        return (abs(odds)/(abs(odds)+100))
    if odds>0:
        return (100/(odds+100))

def calculate_bets_gb(row, diff):
    bet = 0
    fighter = ''
    if (row.Prediction_GB_Winner != 0):
        if row.Prediction_GB_Winner - calculate_odds(row.Fighter_1_Odds) >= diff:
            bet = 100
            fighter = row.Fighter_1
        if (1.0 - row.Prediction_GB_Winner) - calculate_odds(row.Fighter_2_Odds) >= diff:
            bet = 100
            fighter = row.Fighter_2
    if bet > 0:
        rec = f'Bet 100 on {fighter}'
    else:
        rec = 'No bet'
    return rec

def calculate_bets_lgbm(row):
    bet = 0
    if (row.Prediction_LGBM_Winner != 0):
        if row.Prediction_LGBM_Winner > 0.5:
            bet = 100
            fighter = row.Fighter_1
        else:
            bet = 100
            fighter = row.Fighter_2
    if bet > 0:
        rec = f'Bet 100 on {fighter}'
    else:
        rec = 'No bet'
    return rec

def bet_recommender(prediction_df, best_diff, best_fight_number, best_fight_number_lgbm):
    # Instantiating webdriver
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://www.actionnetwork.com/ufc/odds')

    # Getting odds table and formatting
    html = driver.page_source
    tables = pd.read_html(html)
    odds = tables[0]
    odds = odds.iloc[::2]
    odds.reset_index(drop = True, inplace = True)

    # Iterating through to get each fighter's odds
    odds_df = pd.DataFrame(columns = ['Fighter_1', 'Fighter_2', 'Fighter_1_Odds', 'Fighter_2_Odds'])
    fighter_2_regex = r'^[A-Za-z]+\s[A-Za-z]+'
    fighter_1_regex = r'[A-Za-z]+\s[A-Za-z]+(?=[A-Za-z]*\.)'
    flag_regex = r'[^\x00-\x7F]'
    for index, row in odds.iterrows():
        # Getting fighter names
        names_string = re.sub(flag_regex, '', row.Scheduled)
        names_split = names_string.split()
        if len(names_split) == 5:
            fighter_2 = names_split[0] + ' ' + names_split[1][:-2]
            # Splitting middle part to get fighter 1 first name
            need_to_split = names_split[2]
            split = re.findall('[A-Z][^A-Z]*', need_to_split)
            fighter_1 = split[1] + ' ' + names_split[-1]
        else:
            # Case where first name is two names
            try:
                need_to_split = names_split[1]
                split = re.findall('[A-Z][^A-Z]*', need_to_split)
                fighter_2 = names_split[0] + ' ' + split[0]
                if re.findall('[A-Z][^A-Z]*', names_split[1])[1][1] == '.':
                    # Case where second name is three names
                    if len(re.findall('[A-Z][^A-Z]*', names_split[2])) > 1:
                        need_to_split = names_split[2]
                        split = re.findall('[A-Z][^A-Z]*', need_to_split)
                        fighter_1 = split[1] + ' ' + names_split[3] + ' ' + names_split[-1]
            except:
                # Case where first name is three names
                if len(re.findall('[A-Z][^A-Z]*', names_split[2])) > 1:
                    need_to_split = names_split[2]
                    split = re.findall('[A-Z][^A-Z]*', need_to_split)
                    fighter_2 = names_split[0] + ' ' + names_split[1] + ' ' + split[0]
                    # Case where second name is three names
                    if len(re.findall('[A-Z][^A-Z]*', names_split[6])) > 1:
                        need_to_split = names_split[4]
                        split = re.findall('[A-Z][^A-Z]*', need_to_split)
                        fighter_1 = split[1] + ' ' + names_split[5] + ' ' + names_split[-1]
                    # Case where second name is four names
                    else:
                        need_to_split = names_split[4]
                        split = re.findall('[A-Z][^A-Z]*', need_to_split)
                        fighter_1 = split[1] + ' ' + names_split[5] + ' ' + names_split[6] + ' ' + names_split[-1]
                # Case where first name is four names
                else:
                    need_to_split = names_split[3]
                    split = re.findall('[A-Z][^A-Z]*', need_to_split)
                    fighter_2 = names_split[0] + ' ' + names_split[1] + ' ' + names_split[2] + ' ' + split[0]
                    # Case where second name is two names
                    try:
                        if re.findall('[A-Z][^A-Z]*', names_split[-2])[1][1] == '.':
                            need_to_split = names_split[-3]
                            split = re.findall('[A-Z][^A-Z]*', need_to_split)
                            fighter_1 = split[1] +  ' ' + names_split[-1]
                    except:
                        # Case where second name is three names
                        if len(re.findall('[A-Z][^A-Z]*', names_split[7])) > 1:
                            need_to_split = names_split[4]
                            split = re.findall('[A-Z][^A-Z]*', need_to_split)
                            fighter_1 = split[1] + ' ' + names_split[6] + ' ' + names_split[-1]
                        # Case where second name is four names
                        else:
                            need_to_split = names_split[5]
                            split = re.findall('[A-Z][^A-Z]*', need_to_split)
                            fighter_1 = split[1] + ' ' + names_split[6] + ' ' + names_split[7] + ' ' + names_split[-1]
        # Getting fighter odds
        ml_string = row['Best Odds']
        if len(ml_string) == 8:
            ml_fighter_2 = ml_string[:4]
            ml_fighter_1 = ml_string[-4:]
        elif len(ml_string) == 9:
            if (ml_string[4] == '+') | (ml_string[4]=='-'):
                ml_fighter_2 = ml_string[:4]
                ml_fighter_1 = ml_string[-5:]
            else:
                ml_fighter_2 = ml_string[:5]
                ml_fighter_1 = ml_string[-4:]
        elif len(ml_string) == 10:
                ml_fighter_2 = ml_string[:5]
                ml_fighter_1 = ml_string[-5:]
        else:
            continue
        try:
            ml_fighter_2 = float(ml_fighter_2)
        except:
            continue
        try:
            ml_fighter_1 = float(ml_fighter_1)
        except:
            continue
        # Adding data to odds df
        new_data = [fighter_1, fighter_2, ml_fighter_1, ml_fighter_2]
        new_df = pd.DataFrame([new_data])
        new_df.columns = odds_df.columns
        odds_df = pd.concat([odds_df, new_df], ignore_index = True)

    # Calculating GB bets
    odds_df['Prediction_GB_Winner'] = 0
    for index, row in odds_df.iterrows():
        prediction_df['FUZZ_1'] = prediction_df.fighter_1.apply(lambda x: fuzz.ratio(x, row.Fighter_1))
        prediction_df['FUZZ_2'] = prediction_df.fighter_1.apply(lambda x: fuzz.ratio(x, row.Fighter_2))
        try:
            row = prediction_df.loc[(prediction_df.FUZZ_1 > 50) | (prediction_df.FUZZ_2 > 50)]
            gb = row['Prediction_GB_Winner'].values[0]
            if row['FUZZ_1'].values[0] > 50:
                pass
            else:
                gb = 1.0 - gb
            fights_1 = row['wins_1'].values[0] + row['losses_1'].values[0]
            fights_2 = row['wins_2'].values[0] + row['losses_2'].values[0]
            if (fights_1 > best_fight_number) | (fights_2 > best_fight_number):
                odds_df.loc[index, 'Prediction_GB_Winner'] = gb
            else:
                continue
        except:
            continue
    odds_df['Bet_GB'] = odds_df.apply(calculate_bets_gb, diff = best_diff, axis = 1)
    # Calculating LGBM bets
    odds_df['Prediction_LGBM_Winner'] = 0
    for index, row in odds_df.iterrows():
        prediction_df['FUZZ_1'] = prediction_df.fighter_1.apply(lambda x: fuzz.ratio(x, row.Fighter_1))
        prediction_df['FUZZ_2'] = prediction_df.fighter_1.apply(lambda x: fuzz.ratio(x, row.Fighter_2))
        try:
            row = prediction_df.loc[(prediction_df.FUZZ_1 > 50) | (prediction_df.FUZZ_2 > 50)]
            gb = row['Prediction_LGBM_Winner'].values[0]
            if row['FUZZ_1'].values[0] > 50:
                pass
            else:
                gb = 1.0 - gb
            fights_1 = row['wins_1'].values[0] + row['losses_1'].values[0]
            fights_2 = row['wins_2'].values[0] + row['losses_2'].values[0]
            if (fights_1 > best_fight_number_lgbm) | (fights_2 > best_fight_number_lgbm):
                odds_df.loc[index, 'Prediction_LGBM_Winner'] = gb
            else:
                continue
        except:
            continue
    odds_df['Bet_LGBM'] = odds_df.apply(calculate_bets_lgbm, axis = 1)

    return odds_df

def append_bets(this_weeks_bets):
    mma_data = pd.read_csv('mma_bets.csv', index_col = 0)
    mma_data = mma_data.append(this_weeks_bets)
    mma_data.reset_index(inplace = True, drop = True)
    mma_data.to_csv('mma_bets.csv')
    return

def fill_odds():

    # Adding new fights to odds data
    data_filled = pd.read_csv('mma_data_odds.csv', index_col = 0)
    data_all = pd.read_csv('mma_data.csv', index_col = 0)
    data_all = data_all[data_all.result >= 0]
    data_all['Fighter_1_Odds'] = 0
    data_all['Fighter_2_Odds'] = 0
    # Filling odds w/ data with recent fights
    last_row_filled = data_filled.tail(1)
    fighter_1_last = last_row_filled.fighter_1.values[0]
    fighter_2_last = last_row_filled.fighter_2.values[0]
    data_all_copied = data_all.copy()
    data_all_copied.reset_index(inplace = True, drop = True)
    cutoff_unfilled = data_all_copied[(data_all_copied.fighter_1 == fighter_1_last) & 
                                    (data_all_copied.fighter_2 == fighter_2_last)].index[0]
    data_all_new = data_all_copied.iloc[cutoff_unfilled+1:]
    data = pd.concat([data_filled, data_all_new])

    # Filling in odds
    options = Options()
    options.add_argument('--no-sandbox')
    options.add_argument("user-data-dir=/Users/hsinger24/Library/Application Support/Google/Chrome/Default1")
    options.add_argument("--start-maximized")
    options.add_argument('--disable-web-security')
    options.add_argument('--allow-running-insecure-content')
    options.add_argument("--disable-setuid-sandbox")
    driver = webdriver.Chrome(ChromeDriverManager().install())
    driver.get('https://www.bestfightodds.com/archive')
    time.sleep(1)
    for index, row in data.iterrows():
        try:
            if row.Fighter_1_Odds == 0:
                # Formatting name of higher ranked fighter
                fighter_1 = str(row.fighter_1)
                fighter_1 = re.findall('[A-Z][^A-Z]*', fighter_1)    
                fighter_name = ''
                for name in fighter_1:
                    fighter_name = fighter_name + ' ' + name
                # Formatting name of lower ranked fighter
                fighter_2 = str(row.fighter_2)
                fighter_2 = re.findall('[A-Z][^A-Z]*', fighter_2)    
                fighter_name_2 = ''
                for name in fighter_2:
                    fighter_name_2 = fighter_name_2 + ' ' + name
                # Searching for fights w/ higher ranked fighter
                search_bar = driver.find_elements(By.XPATH, '//*[@id="page-content"]/form/p/input[1]')[0]
                search_bar.send_keys(fighter_name)
                driver.find_elements(By.XPATH, '//*[@id="page-content"]/form/p/input[2]')[0].click()
                # Clicking on fighter 1 
                try:
                    driver.find_elements(By.XPATH, '//*[@id="page-content"]/table[1]/tbody/tr[1]/td[2]/a')[0].click()
                    time.sleep(1)
                except:
                    pass
                # Getting odds
                html = driver.page_source
                table = pd.read_html(html)[0]
                table = table[['Matchup', 'Closing range']]
                table['Fuzzy_1'] = table.Matchup.apply(lambda x: fuzz.ratio(x, fighter_name))
                table['Fuzzy_2'] = table.Matchup.apply(lambda x: fuzz.ratio(x, fighter_name_2))
                table = table[(table.Fuzzy_2 > 50) | (table.Fuzzy_1 > 50)].reset_index(drop = True)
                index_opp = table[table.Fuzzy_2 > 50].index[0]
                table_matchup = table.loc[index_opp-1:index_opp, :].reset_index(drop = True)
                # Filling odds
                data.loc[index, 'Fighter_1_Odds'] = table_matchup.loc[0, 'Closing range']
                data.loc[index, 'Fighter_2_Odds'] = table_matchup.loc[1, 'Closing range']
                # Navigating back and clearing text box
                driver.back()
                driver.implicitly_wait(10)
                driver.back()
                driver.implicitly_wait(10)
                driver.find_elements(By.XPATH, '//*[@id="page-content"]/form/p/input[1]')[0].clear()
            else:
                pass
        except:
            driver.quit()
            time.sleep(1)
            options = Options()
            options.add_argument('--no-sandbox')
            options.add_argument("user-data-dir=/Users/hsinger24/Library/Application Support/Google/Chrome/Default1")
            options.add_argument("--start-maximized")
            options.add_argument('--disable-web-security')
            options.add_argument('--allow-running-insecure-content')
            options.add_argument("--disable-setuid-sandbox")
            driver = webdriver.Chrome(ChromeDriverManager().install())
            driver.get('https://www.bestfightodds.com/archive')
    data = data[(data.Fighter_1_Odds != 0) & (data.Fighter_2_Odds != 0)]
    data.dropna(subset = ['Fighter_1_Odds', 'Fighter_2_Odds'], inplace = True)
    data.reset_index(inplace = True, drop = True)

    # Saving updated file
    data.to_csv('mma_data_odds.csv')
    
    return

def calculate_best_bet_construct_gb():
    
    # Internal functions

    def calculate_odds_internal(odds):
        if odds<0:
            return (abs(odds)/(abs(odds)+100))
        if odds>0:
            return (100/(odds+100))
    def calculate_bets_internal(row, diff):
        bet = 0
        if row.Prediction_GB_Winner - calculate_odds_internal(row.Fighter_1_Odds) >= diff:
            bet = 100
        if (1.0 - row.Prediction_GB_Winner) - calculate_odds_internal(row.Fighter_2_Odds) >= diff:
            bet = 100
        return bet
    def calculate_payoff_and_result_internal(row):
        if row.Bet > 0:
            # Calculating Payoff
            if row.Predicted_Result_GB == 1:
                if row.Fighter_1_Odds>0:
                    payoff = (row.Fighter_1_Odds/100)*row.Bet
                else:
                    payoff = row.Bet/((abs(row.Fighter_1_Odds)/100))
            else:
                if row.Fighter_2_Odds>0:
                    payoff = (row.Fighter_2_Odds/100)*row.Bet
                else:
                    payoff = row.Bet/((abs(row.Fighter_2_Odds)/100))
            # Calculating Bet Result
            if row.Predicted_Result_GB == row.result_y:
                bet_result = payoff
            else:
                bet_result = -(row.Bet)
        else:
            bet_result = 0
        return bet_result

    # Setting up data 

    # Joining predictions to table w/ results and getting result
    predictions = pd.read_csv('mma_data_predictions.csv', index_col = 0)
    data = pd.read_csv('mma_data.csv', index_col = 0)
    data = data[data.result >= 0]
    results_data = data[['fighter_1', 'fighter_2', 'result', 'KO_OVR', 'SUB_OVR']]
    odds_data = pd.read_csv('mma_data_odds.csv', index_col = 0)
    merged = predictions.merge(results_data, on = ['fighter_1', 'fighter_2'])
    # Winner results
    merged['Predicted_Result_RF'] = merged.Prediction_RF_Winner.apply(lambda x: 1 if x > 0.5 else 0)
    merged['Predicted_Result_GB'] = merged.Prediction_GB_Winner.apply(lambda x: 1 if x > 0.5 else 0)
    # merged['Accurate_RF'] = merged.apply(lambda x: 1 if x.result_y == x.Predicted_Result_RF else 0, axis = 1)
    # merged['Accurate_GB'] = merged.apply(lambda x: 1 if x.result_y == x.Predicted_Result_GB else 0, axis = 1)
    # # Sub results
    # merged['Predicted_Sub_RF'] = merged.Prediction_RF_SUB.apply(lambda x: 1 if x > 0.5 else 0)
    # merged['Predicted_Sub_GB'] = merged.Prediction_GB_SUB.apply(lambda x: 1 if x > 0.5 else 0)
    # merged['Accurate_RF_SUB'] = merged.apply(lambda x: 1 if x.SUB_OVR_y == x.Predicted_Sub_RF else 0, axis = 1)
    # merged['Accurate_GB_SUB'] = merged.apply(lambda x: 1 if x.SUB_OVR_y == x.Predicted_Sub_GB else 0, axis = 1)
    # # KO Results
    # merged['Predicted_KO_RF'] = merged.Prediction_RF_KO.apply(lambda x: 1 if x > 0.5 else 0)
    # merged['Predicted_KO_GB'] = merged.Prediction_GB_KO.apply(lambda x: 1 if x > 0.5 else 0)
    # merged['Accurate_RF_KO'] = merged.apply(lambda x: 1 if x.KO_OVR_y == x.Predicted_KO_RF else 0, axis = 1)
    # merged['Accurate_GB_KO'] = merged.apply(lambda x: 1 if x.KO_OVR_y == x.Predicted_KO_GB else 0, axis = 1)
    # Getting all the relevant data in one place for bet constructs
    odds_data = odds_data[['fighter_1', 'fighter_2', 'Fighter_1_Odds', 'Fighter_2_Odds']]
    profit_df = merged.merge(odds_data, on = ['fighter_1', 'fighter_2'])
    profit_df = profit_df[(profit_df.Fighter_1_Odds!=0) & (profit_df.Fighter_2_Odds!=0)]

    # Determining best bet construct

    best_diff = 0
    best_profit = 0
    best_fight_number = 0
    for i in [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]:
        profit_df['Bet'] = profit_df.apply(calculate_bets_internal, diff = i, axis = 1)
        profit_df['Bet_Result'] = profit_df.apply(calculate_payoff_and_result_internal, axis = 1)
        print(f'GB - With a cutoff of {i}, betting results are {profit_df.Bet_Result.sum()}')
        if float(profit_df.Bet_Result.sum()) > best_profit:
            best_diff = i
        if profit_df.Bet_Result.sum() > best_profit:
            best_profit = profit_df.Bet_Result.sum()
    # Veteran fights only
    best_profit = 0
    profit_df['Bet'] = profit_df.apply(calculate_bets_internal, diff = best_diff, axis = 1)
    profit_df['Bet_Result'] = profit_df.apply(calculate_payoff_and_result_internal, axis = 1)
    for num_fights in [0, 5, 10, 15, 20, 25]:
        profit_df['Fights_1'] = profit_df.wins_1 + profit_df.losses_1
        profit_df['Fights_2'] = profit_df.wins_2 + profit_df.losses_2
        test = profit_df[(profit_df.Fights_1 > num_fights) | (profit_df.Fights_2 > num_fights)]
        results = test.Bet_Result.sum()
        print(f'GB - For a {num_fights} fight minimum, the model returns {results}')
        if results > best_profit:
            best_fight_number = num_fights
        if results > best_profit:
            best_profit = results
        
    return best_diff, best_fight_number

def calculate_best_bet_construct_lgbm():

    # Calculating straight bet results
    def calculate_straight_internal(row):
        # Calculating Payoff
        if row.Predicted_Result_LGBM == 1:
            if row.Fighter_1_Odds>0:
                payoff = (row.Fighter_1_Odds/100)*row.Bet
            else:
                payoff = row.Bet/((abs(row.Fighter_1_Odds)/100))
        else:
            if row.Fighter_2_Odds>0:
                payoff = (row.Fighter_2_Odds/100)*row.Bet
            else:
                payoff = row.Bet/((abs(row.Fighter_2_Odds)/100))
        # Calculating Bet Result
        if row.Predicted_Result_LGBM == row.result_y:
            bet_result = payoff
        else:
            bet_result = -(row.Bet)
        
        return bet_result

    # Joining predictions to table w/ results and getting result
    predictions = pd.read_csv('mma_data_predictions.csv', index_col = 0)
    data = pd.read_csv('mma_data.csv', index_col = 0)
    data = data[data.result >= 0]
    results_data = data[['fighter_1', 'fighter_2', 'result', 'KO_OVR', 'SUB_OVR']]
    odds_data = pd.read_csv('mma_data_odds.csv', index_col = 0)
    merged = predictions.merge(results_data, on = ['fighter_1', 'fighter_2'])
    # Winner results
    merged['Predicted_Result_RF'] = merged.Prediction_RF_Winner.apply(lambda x: 1 if x > 0.5 else 0)
    merged['Predicted_Result_GB'] = merged.Prediction_GB_Winner.apply(lambda x: 1 if x > 0.5 else 0)
    merged['Predicted_Result_LGBM'] = merged.Prediction_LGBM_Winner.apply(lambda x: 1 if x > 0.5 else 0)
    # Joining to odds df
    odds_data = odds_data[['fighter_1', 'fighter_2', 'Fighter_1_Odds', 'Fighter_2_Odds']]
    profit_df = merged.merge(odds_data, on = ['fighter_1', 'fighter_2'])
    profit_df = profit_df[(profit_df.Fighter_1_Odds!=0) & (profit_df.Fighter_2_Odds!=0)]
    # Calculating results
    profit_df['Bet'] = 100
    profit_df['Bet_Result'] = profit_df.apply(calculate_straight_internal, axis = 1)

    # Determining best bet construct

    best_profit = 0
    best_fight_number = 0
    for num_fights in [10, 15, 20, 25]:
        profit_df['Fights_1'] = profit_df.wins_1 + profit_df.losses_1
        profit_df['Fights_2'] = profit_df.wins_2 + profit_df.losses_2
        test = profit_df[(profit_df.Fights_1 > num_fights) | (profit_df.Fights_2 > num_fights)]
        results = test.Bet_Result.sum()
        print(f'LGBM - For a {num_fights} minimum, the model returns {results}')
        if float(profit_df.Bet_Result.sum()) > best_profit:
            best_fight_number = num_fights
        if profit_df.Bet_Result.sum() > best_profit:
            best_profit = profit_df.Bet_Result.sum()
    
    return best_fight_number



##########SCRIPT##########

# Filling in odds of recent fights
fill_odds()

# Determining best bet construct
best_diff, best_fight_number = calculate_best_bet_construct_gb()
best_fight_number_lgbm = calculate_best_bet_construct_lgbm()

# Appending this week's fight data to existing dataset
this_weeks_fights = retrieve_this_weeks_fights()
append_fight_data(this_weeks_fights)

# Training models & using it to predict fights
this_weeks_predictions = this_weeks_predictions(this_weeks_fights)
append_predictions(this_weeks_predictions)

# Calculating bets 
this_weeks_bets = bet_recommender(this_weeks_predictions, best_diff = best_diff, best_fight_number = best_fight_number, best_fight_number_lgbm = best_fight_number_lgbm)
append_bets(this_weeks_bets)