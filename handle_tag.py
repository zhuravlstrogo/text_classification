import pandas as pd
import numpy as np
import re


# def extract_tag(tag):
#     if len(tag) > 5:
#         print(tag)
#         return tag.split("value': '",1)[1][:-3]

df = pd.read_csv('comm_400_days.csv', sep=';')
print(df.columns)

# print(df[['published_at_author', 'text_tonality', 'text_author']].sample(5))

# 2024-06-07
print(max(df['published_at_author']))

# 2023-05-04
print(min(df['published_at_author']))

def save_range(st, en, save_file_name):

    data = df[(df['published_at_author'] >= st)& (df['published_at_author'] <= en)]
    # df.to_csv(save_file_name, encoding='utf-8-sig')
    data.to_csv(save_file_name, encoding='utf-8-sig')
    print(f'shape: {df.shape}')

save_range(st='2023-05-01', en='2023-11-01', save_file_name='part_1.csv')
save_range(st='2023-11-01', en='2024-02-01', save_file_name='part_2.csv')
save_range(st='2024-02-01', en='2024-04-01', save_file_name='part_3.csv')
save_range(st='2024-04-01', en='2024-06-08', save_file_name='part_4.csv')

# df['tags_handled'] = df['tags'].apply(extract_tag)
# print(df['tags_handled'].value_counts())

# df.to_csv('new_comm_400_days.csv', sep=';')


