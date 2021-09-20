from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support.ui import Select
import time
import pandas as pd
import re
from bs4 import BeautifulSoup
import datetime as dt
from webdriver_manager.chrome import ChromeDriverManager

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
    final= pd.merge(evens, odds, left_on = evens.index, right_on = odds.index)
    final.drop('key_0', axis = 1, inplace = True)
    final.columns = ['fighter_1', 'weight_1', 'reach_1', 'age_1', 'slpm_1', 'sapm_1', 'td_avg_1', 'sub_avg_1', 
                    'strk_acc_1', 'strk_def_1', 'td_acc_1','td_def_1', 'wins_1', 'losses_1', 'fighter_2', 'weight_2', 
                    'reach_2', 'age_2', 'slpm_2', 'sapm_2', 'td_avg_2', 'sub_avg_2', 'strk_acc_2', 'strk_def_2', 
                    'td_acc_2','td_def_2', 'wins_2', 'losses_2']
    final['result'] = 0
    print(final.tail())
    driver.quit()
    return final

def append_fight_data(this_weeks_fights):
    mma_data = pd.read_csv('mma_data.csv', index_col = 0)
    mma_data = mma_data.append(this_weeks_fights)
    mma_data.reset_index(inplace = True, drop = True)
    mma_data.to_csv('mma_data.csv')
    return


# Appending this week's fight data to existing dataset
this_weeks_fights = retrieve_this_weeks_fights()
append_fight_data(this_weeks_fights)
    