
import requests

import json
import csv

f = open(r"C:\Users\filiz\Desktop\YapayZekaProje\YapayZekaProje.txt")

sites = f.readlines()

header = True
csv_file = open("osuDataset.csv", 'w', newline='') # 3. arguman windows için \r\n\n oayından linuxda kaldırılacak
writer = csv.writer(csv_file)
for site in sites:
    res = requests.get(site)
    if (res.status_code == 200):
        beatmapsets = res.json()['beatmapsets']
        for i in range(len(beatmapsets)):
            beatmaps=beatmapsets[i]['beatmaps']
            for j in range(len(beatmaps)):
                if(header == True):
                    header=False
                    writer.writerow(list(beatmaps[j].keys())[:-2])
                writer.writerow(list(beatmaps[j].values())[:-2])
                