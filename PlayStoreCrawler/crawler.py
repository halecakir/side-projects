import requests
import time
import csv

def get_app_ids():
    start_url = 'http://localhost:3000/api/apps/?collection=topselling_free&category=COMMUNICATION&lang=en'
    waiting_urls = set()
    app_ids = set()

    #top free communication
    response = requests.get(start_url)
    if response.status_code == 200:
        for app in response.json()["results"]:
            waiting_urls.add(app["url"])
    else:
        exit("Request Error")


    with open("app_ids_record_audio.txt", "w", buffering=1) as target:
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
            except KeyError:
                print("All linked applications are traversed")
                break

            #add apk id if it is free, popular, and has longer description than 100 characters
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
                                    target.write(url.split('/')[-1]+"\n")

            #add similar app urls
            response = requests.get(url + '/similar')
            if response.status_code == 200:
                for app in response.json()["results"]:
                    if app["appId"] not in app_ids:
                        waiting_urls.add(app["url"])
            counter += 1

        for url in waiting_urls:
            response = requests.get(url)
            if response.status_code == 200:
                json = response.json()
                if json["minInstalls"] > 10000 and json["priceText"] == "Free" and len(json["description"]) > 500:
                    if url.split('/')[-1] not in app_ids:
                        app_ids.add(url.split('/')[-1] )
                        target.write(url.split('/')[-1]+"\n")


def get_descriptions(id_file, out_file):
    #Get application ids beforehand
    app_infos = {}
    with open(id_file, "r") as target:
        for app_id in target:
            app_infos[app_id.rstrip()] = None

    #Get application descriptions
    base_url = "http://localhost:3000/api/apps/"
    lang = "/?lang=en"
    with open(out_file, "w") as out:
        writer = csv.writer(out)
        writer.writerow(["application_id", "description"])
        for app_id in app_infos:
            response = requests.get(base_url+app_id+lang)
            if response.status_code == 200:
                if "description" in response.json():
                    writer.writerow([app_id, response.json()["description"]])


if __name__ == "__main__":
    ID_FILE = "record_audio_applist.txt"
    OUT_FILE = "record_audio_applist.csv"
    get_descriptions(ID_FILE, OUT_FILE)
