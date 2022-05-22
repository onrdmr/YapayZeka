import csv
import pandas as pd

df = pd.read_csv('osuDataset.csv')

df.head(3)
# with open('osuDataset.csv', 'r') as csv_file:

#     csv_reader = csv.reader(csv_file, delimiter=',')

#     for row in csv_reader:
#         print(row)