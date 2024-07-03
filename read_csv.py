from glob import glob
import os
import shutil

import pandas as pd
import numpy as np
import datetime


# tags_2024 = pd.read_csv('2024_tags.csv', sep=';')
tags_2023 = pd.read_csv('2023_tags.csv', sep=';')
# tags_2022 = pd.read_csv('2022_tags.csv', sep=';')

parser_id = set(tags_2023['id'].unique())

# tags_df = pd.concat([tags_2024, tags_2023, tags_2022], axis=0)

# print(f'2024: {len(tags_2024)}')
# print(f'2023: {len(tags_2023)}')
# print(f'2022: {len(tags_2022)}')

df = pd.read_csv('Tanya_file.csv')

print(max(df['Дата публикации отзыва'])) # 2023-09-22
print(min(df['Дата публикации отзыва'])) # 2021-09-01

1/0
print(f'len df {len(df)}')
df = df.drop_duplicates(subset='ID отзыва')
print(len(df))

print(f'len tags {len(tags_2023)}')
# tags_2023 = tags_2023.drop_duplicates(subset='id')
# print(len(tags_2023))

print('DUPLICATES')
ids = tags_2023["id"]
print(tags_2023[ids.isin(ids[ids.duplicated()])].sort_values("id"))

# file_id = set(df['ID отзыва'].unique())

# print('intersection')
# print(len(file_id.intersection(parser_id)))


# dt_date = '2023-03-25'
# df_2023 = df[(df['Дата публикации отзыва'] >= dt_date) & (df['Дата публикации отзыва'] <= dt_date)]

# df_2023.to_excel(f'file_{dt_date}.xlsx')

# df_2023 = df_2023[df_2023['Теги'].notnull()]
# print(f'2023: {len(df_2023)}')

# df_2022 = df[(df['Дата публикации отзыва'] >= '2022-01-01') & (df['Дата публикации отзыва'] <= '2022-12-31')]
# df_2022 = df_2022[df_2022['Теги'].notnull()]
# print(f'2022: {len(df_2022)}')

# print('MAX')
# print(max(df['Дата публикации отзыва']))


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

    