from glob import glob
import os
import shutil

import pandas as pd
import numpy as np
import datetime



# df = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')

df = pd.read_csv('2023_tags.csv', sep=';')
print(len(df))
# # df = pd.read_csv('comm_2023-01-01_2023-12-31.csv', sep=';')

# df = df[df['Дата публикации отзыва'] == '2023-03-04']

# print(df['Время публикации отзыва'].sample(7))

# print(min(df['Время публикации отзыва']))
# print(max(df['Время публикации отзыва']))

# # print(df.head())
# print(np.unique(df['Время публикации отзыва']))


def merge_all_csv(csv_dir, result_csv):

    first_hdr = True

    # all .csv files in the directory have the same header
    with open(result_csv, "w", newline="") as result_file:
        for filename in glob(os.path.join(csv_dir, "*.csv")):
            with open(filename) as in_file:
                header = in_file.readline()
                if first_hdr:
                    result_file.write(header)
                    first_hdr = False
                shutil.copyfileobj(in_file, result_file)


if __name__ == "__main__":
    print('ok')
    # csv_dir = 'reviews/2024'
    # result_csv = '2024_tags.csv'

    # merge_all_csv(csv_dir, result_csv)

    