import csv

IN = "/home/huseyinalecakir/Security/data/play_store_data/app_ids_record_audio_2228.txt"
OLD = "/home/huseyinalecakir/Security/data/play_store_data/record_audio_selected_2000_first.csv"
EK_2 = "/home/huseyinalecakir/Security/data/play_store_data/app_ids_record_audio_ek2.txt"

def get_prev_versions(versions):
    id_list = set()
    with open(versions) as stream:
        reader = csv.reader(stream)
        next(reader)
        for row in reader:
            if row[1].startswith("##"):
                id_list.add(row[1].split("##")[1])
    return id_list
	

old_id_list = get_prev_versions(OLD)

new_id_list = set()

with open(IN) as target:
	for line in target:
		if line.strip() not in old_id_list:
			new_id_list.add(line.strip())

with open(EK_2, "w") as target:
	for app in new_id_list:
		target.write(app + "\n")


