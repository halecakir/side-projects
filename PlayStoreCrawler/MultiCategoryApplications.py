#!/usr/bin/env python
# coding: utf-8

# In[1]:


import requests
import time
import csv
import random
from time import sleep


# In[2]:


categories = {"categories":[{"cat_key":"OVERALL","name":"Overall"},{"cat_key":"APPLICATION","name":"All apps"},{"cat_key":"GAME","name":"All games"},{"cat_key":"ART_AND_DESIGN","name":"Art & Design"},{"cat_key":"AUTO_AND_VEHICLES","name":"Auto & Vehicles"},{"cat_key":"BEAUTY","name":"Beauty"},{"cat_key":"BOOKS_AND_REFERENCE","name":"Books & Reference"},{"cat_key":"BUSINESS","name":"Business"},{"cat_key":"COMICS","name":"Comics"},{"cat_key":"COMMUNICATION","name":"Communication"},{"cat_key":"DATING","name":"Dating"},{"cat_key":"EDUCATION","name":"Education"},{"cat_key":"ENTERTAINMENT","name":"Entertainment"},{"cat_key":"EVENTS","name":"Events"},{"cat_key":"FINANCE","name":"Finance"},{"cat_key":"FOOD_AND_DRINK","name":"Food & Drink"},{"cat_key":"HEALTH_AND_FITNESS","name":"Health & Fitness"},{"cat_key":"HOUSE_AND_HOME","name":"House & Home"},{"cat_key":"LIFESTYLE","name":"Lifestyle"},{"cat_key":"MAPS_AND_NAVIGATION","name":"Maps & Navigation"},{"cat_key":"MEDICAL","name":"Medical"},{"cat_key":"MUSIC_AND_AUDIO","name":"Music & Audio"},{"cat_key":"NEWS_AND_MAGAZINES","name":"News & Magazines"},{"cat_key":"PARENTING","name":"Parenting"},{"cat_key":"PERSONALIZATION","name":"Personalization"},{"cat_key":"PHOTOGRAPHY","name":"Photography"},{"cat_key":"PRODUCTIVITY","name":"Productivity"},{"cat_key":"SHOPPING","name":"Shopping"},{"cat_key":"SOCIAL","name":"Social"},{"cat_key":"SPORTS","name":"Sports"},{"cat_key":"TOOLS","name":"Tools"},{"cat_key":"TRAVEL_AND_LOCAL","name":"Travel & Local"},{"cat_key":"VIDEO_PLAYERS","name":"Video Players & Editors"},{"cat_key":"WEATHER","name":"Weather"},{"cat_key":"LIBRARIES_AND_DEMO","name":"Libraries & Demo"},{"cat_key":"GAME_ARCADE","name":"Arcade"},{"cat_key":"GAME_PUZZLE","name":"Puzzle"},{"cat_key":"GAME_CARD","name":"Cards"},{"cat_key":"GAME_CASUAL","name":"Casual"},{"cat_key":"GAME_RACING","name":"Racing"},{"cat_key":"GAME_SPORTS","name":"Sport Games"},{"cat_key":"GAME_ACTION","name":"Action"},{"cat_key":"GAME_ADVENTURE","name":"Adventure"},{"cat_key":"GAME_BOARD","name":"Board"},{"cat_key":"GAME_CASINO","name":"Casino"},{"cat_key":"GAME_EDUCATIONAL","name":"Educational"},{"cat_key":"GAME_MUSIC","name":"Music Games"},{"cat_key":"GAME_ROLE_PLAYING","name":"Role Playing"},{"cat_key":"GAME_SIMULATION","name":"Simulation"},{"cat_key":"GAME_STRATEGY","name":"Strategy"},{"cat_key":"GAME_TRIVIA","name":"Trivia"},{"cat_key":"GAME_WORD","name":"Word Games"},{"cat_key":"ANDROID_WEAR","name":"Android Wear"},{"cat_key":"FAMILY","name":"Family All Ages"},{"cat_key":"FAMILY_UNDER_5","name":"Family Ages 5 & Under"},{"cat_key":"FAMILY_6_TO_8","name":"Family Ages 6-8"},{"cat_key":"FAMILY_9_AND_UP","name":"Family Ages 9 & Up"},{"cat_key":"FAMILY_ACTION","name":"Family Action"},{"cat_key":"FAMILY_BRAINGAMES","name":"Family Brain Games"},{"cat_key":"FAMILY_CREATE","name":"Family Create"},{"cat_key":"FAMILY_EDUCATION","name":"Family Education"},{"cat_key":"FAMILY_MUSICVIDEO","name":"Family Music & Video"},{"cat_key":"FAMILY_PRETEND","name":"Family Pretend Play"}]}


# In[3]:


#category_keys = [cat["cat_key"] for cat in categories["categories
category_keys = ["GAME_TRIVIA", "HEALTH_AND_FITNESS", "PARENTING", "BOOKS_AND_REFERENCE", "GAME_MUSIC", "SPORTS", "GAME_EDUCATIONAL", "MEDICAL", "LIBRARIES_AND_DEMO", "GAME_CARD", "FOOD_AND_DRINK", "GAME_SIMULATION", "GAME_SPORTS", "GAME_ADVENTURE", "EVENTS", "ART_AND_DESIGN", "AUTO_AND_VEHICLES", "GAME_RACING", "GAME_PUZZLE", "NOT_FOUND", "BEAUTY", "COMICS", "GAME_ARCADE"]


# In[4]:


def get_top_free_applications(categories):
    def per_category_top_free(category):
        waiting_urls = set()
        url = 'http://localhost:3000/api/apps/?collection=topselling_free&category={}&lang=en'.format(category)
        #top free applications
        response = requests.get(url)
        if response.status_code == 200:
            for app in response.json()["results"]:
                waiting_urls.add(app["url"])
        else:
            print("Request Error")
        return waiting_urls
    all_top_free_apps = set()
    for category in categories:
        all_top_free_apps.update(per_category_top_free(category))
    return all_top_free_apps


# In[5]:


def get_app_category(app_id):
    base_url = 'http://localhost:3000/api/apps/'
    response = requests.get(base_url+app_id)
    if response.status_code == 200:
        gid = response.json()["genreId"].upper()
        return gid
    else:
        print(app_id, " category cannot be found")
        return "NOT_FOUND"


# In[6]:


import pandas as pd
def application_list(filename, sheetname):
    df = pd.read_excel(filename, sheet_name=sheetname)
    data = set()
   
    for index, row in df.iterrows():
        if row["Sentences"].startswith("##"):
            app_id = row["Sentences"].split("##")[1]
            data.add(app_id)
    return data


# In[9]:


dowloaded_list = application_list("SEVIL.xlsx", "Sheet1")


# In[13]:


top_free_apps = get_top_free_applications(category_keys)


# In[14]:


top_free_apps


# In[15]:


from bs4 import BeautifulSoup

def get_similar_ids(app_id):
    print(app_id)
    sleep(0.1)
    page = requests.get("https://play.google.com/store/apps/details?id={}".format(app_id))
    soup = BeautifulSoup(page.content, 'html.parser')
    href = soup.find(class_='LkLjZd ScJHi U8Ww7d xjAeve nMZKrb id-track-click ')
    if href != None:
        href = href["href"]
        sleep(0.2)
        page = requests.get(href)
        soup = BeautifulSoup(page.content, 'html.parser')
        similars = []
        for link in soup.find_all(class_="JC71ub"):
            similars.append(link['href'].split("=")[1])
        return similars
    else:
        return []


# In[16]:


def get_app_ids(top_free_apps):
    waiting_urls = list(top_free_apps)
    random.shuffle(waiting_urls)
    waiting_urls_unique_check = set([url for url in waiting_urls])
    app_ids = set()

    
    counter = 0
    start_time = time.time()
    while (len(app_ids) + len(waiting_urls)) < 100000:
        if counter % 10 == 0:
            elapsed_time = time.time() - start_time
            print("Total number of apps", len(app_ids) + len(waiting_urls))
            print("Collected app ids", len(app_ids))
            print("Elapsed time up to now is {}".format(elapsed_time))
        try:
            url = waiting_urls.pop()
            waiting_urls_unique_check.remove(url)
        except Exception:
            print("All linked applications are traversed")
            break

        #add apk id if it is free, popular, and has longer description than 500 characters
        response = requests.get(url)
        if response.status_code == 200:
            json = response.json()
            if json["minInstalls"] > 10000 and json["priceText"] == "Free" and len(json["description"]) > 500:
                if url.split('/')[-1] not in app_ids:
                    response = requests.get(url+"/permissions")
                    if response.status_code == 200:
                        json = response.json()
                        exist = False
                        for line in json["results"]:
                            if "record audio" in line["permission"]:
                                app_ids.add(url.split('/')[-1] )
                                application_id = url.split('/')[-1]
                                cat = get_app_category(application_id)
                                with open("RECORD_AUDIO/record_audio_{}.txt".format(cat), "a") as target:
                                    if application_id not in dowloaded_list:
                                        target.write(application_id + "\n")
                                

        #add similar app urls
        """
        response = requests.get(url + '/similar')
        similar_count = 0
        if response.status_code == 200:
            for app in response.json()["results"]:
                similar_count += 1
                if app["appId"] not in app_ids:
                    
                    if app["url"] not in waiting_urls_unique_check:
                        
                        waiting_urls_unique_check.add(app["url"])
                        waiting_urls.append(app["url"])
        else:
            print("ERROR")
        print("Similar count for {} is {}".format(url, similar_count))
        """
        for sim_id in get_similar_ids(url.split("/")[-1]):
            sim_id = "http://localhost:3000/api/apps/" + sim_id
            if sim_id not in waiting_urls_unique_check:
                waiting_urls_unique_check.add(sim_id)
                waiting_urls.append(sim_id)
        random.shuffle(waiting_urls)
        counter += 1

    for url in waiting_urls:
        response = requests.get(url)
        if response.status_code == 200:
            json = response.json()
            if json["minInstalls"] > 10000 and json["priceText"] == "Free" and len(json["description"]) > 500:
                if url.split('/')[-1] not in app_ids:
                    app_ids.add(url.split('/')[-1] )
                    cat = get_app_category(application_id)
                    with open("RECORD_AUDIO/record_audio_{}.txt".format(cat), "a") as target:
                        if application_id not in dowloaded_list:
                            target.write(application_id + "\n")


# In[17]:


get_app_ids(top_free_apps)

