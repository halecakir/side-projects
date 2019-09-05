#!/usr/bin/env python
# coding: utf-8

# In[48]:


from bs4 import BeautifulSoup
import pandas as pd
import requests
import time
import csv
import random
from time import sleep
import sys


# In[53]:


class PlayStoreCrawler:
    def __init__(self, permission_type, permission_content, check_downloaded_apps, downloded_file):
        self.permission_type = permission_type
        self.check_downloaded_apps = check_downloaded_apps
        self.permission_content = permission_content
        if self.check_downloaded_apps:
            self.dowloaded_list = self.application_list(downloded_file)
    
    def get_top_free_applications(self, categories):
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

    def get_app_category(self, app_id):
        base_url = 'http://localhost:3000/api/apps/'
        response = requests.get(base_url+app_id)
        if response.status_code == 200:
            gid = response.json()["genreId"].upper()
            return gid
        else:
            print(app_id, " category cannot be found")
            return "NOT_FOUND"
        
    def application_list(self, filename):
        df = pd.read_excel(filename)
        data = set()
        for _, row in df.iterrows():
            if row["Sentences"].startswith("##"):
                app_id = row["Sentences"].split("##")[1]
                data.add(app_id)
        return data
    
    def get_similar_ids(self, app_id):
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
    
    def get_app_ids(self, init_apps):

        waiting_urls = list(init_apps)
        random.shuffle(waiting_urls)
        waiting_urls_unique_check = set([url for url in waiting_urls])
        app_ids = set()
        
        """
        counter = 0
        while (len(app_ids) + len(waiting_urls)) < 100000:
            try:
                if counter % 10 == 0:
                    print("Total number of apps", len(app_ids) + len(waiting_urls))
                    print("Collected app ids", len(app_ids))
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
                                for line in json["results"]:
                                    if self.permission_content in line["permission"]:
                                        application_id = url.split('/')[-1]
                                        app_ids.add(application_id)
                                        cat = self.get_app_category(application_id)
                                        with open("{}/{}.txt".format(self.permission_type, cat), "a") as target:
                                            if application_id not in self.dowloaded_list:
                                                target.write(application_id + "\n")

                #add similar app urls
                for sim_id in self.get_similar_ids(url.split("/")[-1]):
                    sim_id = "http://localhost:3000/api/apps/" + sim_id
                    if sim_id not in waiting_urls_unique_check:
                        waiting_urls_unique_check.add(sim_id)
                        waiting_urls.append(sim_id)
                random.shuffle(waiting_urls)
                counter += 1
            except Exception:
                print("Number of waiting_urls ", len(waiting_urls))
                break
        """
        for url in waiting_urls:
            response = requests.get(url)
            if response.status_code == 200:
                json = response.json()
                if json["minInstalls"] > 10000 and json["priceText"] == "Free" and len(json["description"]) > 500:
                    application_id = url.split('/')[-1]
                    if application_id not in app_ids:
                        app_ids.add(application_id )
                        cat = self.get_app_category(application_id)
                        with open("{}/{}.txt".format(self.permission_type, cat), "a") as target:
                            if application_id not in self.dowloaded_list:
                                target.write(application_id + "\n")


# In[54]:

permission = sys.argv[1]
print(permission)

categories = {"categories":[{"cat_key":"OVERALL","name":"Overall"},{"cat_key":"APPLICATION","name":"All apps"},{"cat_key":"GAME","name":"All games"},{"cat_key":"ART_AND_DESIGN","name":"Art & Design"},{"cat_key":"AUTO_AND_VEHICLES","name":"Auto & Vehicles"},{"cat_key":"BEAUTY","name":"Beauty"},{"cat_key":"BOOKS_AND_REFERENCE","name":"Books & Reference"},{"cat_key":"BUSINESS","name":"Business"},{"cat_key":"COMICS","name":"Comics"},{"cat_key":"COMMUNICATION","name":"Communication"},{"cat_key":"DATING","name":"Dating"},{"cat_key":"EDUCATION","name":"Education"},{"cat_key":"ENTERTAINMENT","name":"Entertainment"},{"cat_key":"EVENTS","name":"Events"},{"cat_key":"FINANCE","name":"Finance"},{"cat_key":"FOOD_AND_DRINK","name":"Food & Drink"},{"cat_key":"HEALTH_AND_FITNESS","name":"Health & Fitness"},{"cat_key":"HOUSE_AND_HOME","name":"House & Home"},{"cat_key":"LIFESTYLE","name":"Lifestyle"},{"cat_key":"MAPS_AND_NAVIGATION","name":"Maps & Navigation"},{"cat_key":"MEDICAL","name":"Medical"},{"cat_key":"MUSIC_AND_AUDIO","name":"Music & Audio"},{"cat_key":"NEWS_AND_MAGAZINES","name":"News & Magazines"},{"cat_key":"PARENTING","name":"Parenting"},{"cat_key":"PERSONALIZATION","name":"Personalization"},{"cat_key":"PHOTOGRAPHY","name":"Photography"},{"cat_key":"PRODUCTIVITY","name":"Productivity"},{"cat_key":"SHOPPING","name":"Shopping"},{"cat_key":"SOCIAL","name":"Social"},{"cat_key":"SPORTS","name":"Sports"},{"cat_key":"TOOLS","name":"Tools"},{"cat_key":"TRAVEL_AND_LOCAL","name":"Travel & Local"},{"cat_key":"VIDEO_PLAYERS","name":"Video Players & Editors"},{"cat_key":"WEATHER","name":"Weather"},{"cat_key":"LIBRARIES_AND_DEMO","name":"Libraries & Demo"},{"cat_key":"GAME_ARCADE","name":"Arcade"},{"cat_key":"GAME_PUZZLE","name":"Puzzle"},{"cat_key":"GAME_CARD","name":"Cards"},{"cat_key":"GAME_CASUAL","name":"Casual"},{"cat_key":"GAME_RACING","name":"Racing"},{"cat_key":"GAME_SPORTS","name":"Sport Games"},{"cat_key":"GAME_ACTION","name":"Action"},{"cat_key":"GAME_ADVENTURE","name":"Adventure"},{"cat_key":"GAME_BOARD","name":"Board"},{"cat_key":"GAME_CASINO","name":"Casino"},{"cat_key":"GAME_EDUCATIONAL","name":"Educational"},{"cat_key":"GAME_MUSIC","name":"Music Games"},{"cat_key":"GAME_ROLE_PLAYING","name":"Role Playing"},{"cat_key":"GAME_SIMULATION","name":"Simulation"},{"cat_key":"GAME_STRATEGY","name":"Strategy"},{"cat_key":"GAME_TRIVIA","name":"Trivia"},{"cat_key":"GAME_WORD","name":"Word Games"},{"cat_key":"ANDROID_WEAR","name":"Android Wear"},{"cat_key":"FAMILY","name":"Family All Ages"},{"cat_key":"FAMILY_UNDER_5","name":"Family Ages 5 & Under"},{"cat_key":"FAMILY_6_TO_8","name":"Family Ages 6-8"},{"cat_key":"FAMILY_9_AND_UP","name":"Family Ages 9 & Up"},{"cat_key":"FAMILY_ACTION","name":"Family Action"},{"cat_key":"FAMILY_BRAINGAMES","name":"Family Brain Games"},{"cat_key":"FAMILY_CREATE","name":"Family Create"},{"cat_key":"FAMILY_EDUCATION","name":"Family Education"},{"cat_key":"FAMILY_MUSICVIDEO","name":"Family Music & Video"},{"cat_key":"FAMILY_PRETEND","name":"Family Pretend Play"}]}
if permission == "RECORD_AUDIO":
    category_keys = ["GAME_TRIVIA", "HEALTH_AND_FITNESS", "PARENTING", "BOOKS_AND_REFERENCE", "GAME_MUSIC", "SPORTS", "GAME_EDUCATIONAL", "MEDICAL", "LIBRARIES_AND_DEMO", "GAME_CARD", "FOOD_AND_DRINK", "GAME_SIMULATION", "GAME_SPORTS", "GAME_ADVENTURE", "EVENTS", "ART_AND_DESIGN", "AUTO_AND_VEHICLES", "GAME_RACING", "GAME_PUZZLE", "NOT_FOUND", "BEAUTY", "COMICS", "GAME_ARCADE"]
else:
    category_keys = [cat["cat_key"] for cat in categories["categories"]]


permission_phrase = "record audio" if permission == "RECORD_AUDIO" else "read your contacts"
downloaded_apps_check = True if permission == "RECORD_AUDIO" else False
crawler = PlayStoreCrawler(permission, permission_phrase, downloaded_apps_check, sys.argv[2])
top_free_apps = crawler.get_top_free_applications(category_keys)
# In[55]:
crawler.get_app_ids(top_free_apps)

