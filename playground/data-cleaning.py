#!/usr/bin/env python
# coding: utf-8

# In[2]:


import os,sys,inspect
current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir) 

from utils.io_utils import IOUtils
from utils.nlp_utils import NLPUtils


# In[29]:


DIR_NAME = os.path.abspath('')
IN_PATH = os.path.join(DIR_NAME, "../../../data/play_store_data/app_ids_record_audio_ek2.csv")
SELECTED_PATH = os.path.join(DIR_NAME, "../../../data/play_store_data/permissionsList_357.txt")
OUT_PATH = os.path.join(DIR_NAME, "../../../data/play_store_data/record_audio_selected_ek2.xls")
VERSION1 = os.path.join(DIR_NAME, "../../../data/play_store_data/record_audio_selected_2000_first.csv")


# In[30]:


import csv, time
def clean_play_store_data(file_path):
    """TODO"""
    number_of_apps = 0
    data = {}
    with open(file_path) as stream:
        reader = csv.reader(stream)
        next(reader)
        start_time = time.time()
        for row in reader:
            number_of_apps += 1
            app_id = row[0]
            text = row[1]
            data[app_id] = []
            for sentence in NLPUtils.sentence_tokenization(text):
                data[app_id].append(sentence)
    return data


# In[31]:


import re
def remove_given_pattern(regex, data):
    updated_data = {}
    for  key in list(data.keys()):
        new_lines = []
        for line in data[key]:
            updated = re.sub(regex, '', line)
            new_lines.append(updated)
        updated_data[key] = new_lines
    return updated_data


# In[37]:


def select_given_apps(data, file_path):
    selected_apps = {}
    last_key = None
    with open(file_path) as stream:
        for line in stream:
            line = line.rstrip()
            if line.startswith("%%"):
                last_key = line.split("%%")[1]
                selected_apps[last_key] = {"uses-permission" : [], "permission" : []}
            else:
                if line:
                    if line.startswith("uses-permission"):
                        selected_apps[last_key]["uses-permission"].append(line)
                    elif line.startswith("permission"):
                        selected_apps[last_key]["permission"].append(line)
    new_data = {}
    for app_id in selected_apps:
        if app_id in data:
            new_data[app_id] = {}
            new_data[app_id]["data"] = data[app_id]
            new_data[app_id]["permissions"] = selected_apps[app_id]
    return new_data


# In[38]:


def remove_emoji(data):
    updated_data = {}
    import demoji
    demoji.download_codes()
    for key in list(data.keys()):
        new_lines = []
        for line in data[key]:
            updated = demoji.replace(line, repl="").strip()
            if updated:
                new_lines.append(updated)
        updated_data[key] = new_lines
    return updated_data


# In[50]:


def write_excel(data, outfile, eliminated_apps={}, count=2000):
    import xlwt
    header = ["Count", "Sentences", "Manually Marked", "uses-permission", "permission"]
    
    style = xlwt.XFStyle()
    # font
    font = xlwt.Font()
    font.bold = True
    style.font = font
    
    workbook = xlwt.Workbook()
    sheet = workbook.add_sheet('Sheet')
    sheet.write(0, 0, "Count", style=style) 
    sheet.write(0, 1, "Sentences", style=style) 
    sheet.write(0, 2, "Manually Marked", style=style)
    sheet.write(0, 3, "uses-permission", style=style)
    sheet.write(0, 4, "permission", style=style)

    row_number = 1
    app_num = 1
    for idx, app_id in zip(range(count), data):
        if app_id not in eliminated_apps:
            sheet.write(row_number, 0, "#{}".format(app_num), style=style)
            sheet.write(row_number, 1, "##{}".format(app_id), style=style)
            uses_permissions = ":".join(data[app_id]["permissions"]["uses-permission"])
            permissions = ":".join(data[app_id]["permissions"]["permission"])
            sheet.write(row_number, 3, ":{}".format(uses_permissions))
            sheet.write(row_number, 4, ":{}".format(permissions))
            row_number += 1
            for sentence in data[app_id]["data"]:
                sheet.write(row_number, 1, sentence)
                row_number += 1
            app_num += 1
    workbook.save(outfile)


# In[47]:


def get_prev_versions(versions):
    id_list = set()
    with open(versions) as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if row[1].startswith("##"):
                id_list.add(row[1].split("##")[1])
    return id_list


# In[41]:


regex = r"^[^\w\!\?\\\(\)\[\]\“\‘\"]+"
data = clean_play_store_data(IN_PATH)
data = remove_given_pattern(regex, data)
data = remove_emoji(data)
data = select_given_apps(data, SELECTED_PATH)


# In[48]:


len(data)


# In[51]:


prev_id_list = get_prev_versions(VERSION1)
write_excel(data, OUT_PATH, prev_id_list, 2000)


# In[36]:


def word_counter(data, count=1000):
    counter = 0
    for idx, app_id in zip(range(count), data):
        for sentence in data[app_id]["data"]:
            counter += len(sentence.split(" "))
            
    return counter


# In[37]:


count = word_counter(data)


# In[38]:


print(count)

