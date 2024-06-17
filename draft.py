import pandas as pd
import numpy as np

from utils import rename_tags, join_words, extract_tag, process_classes


df_2024 = pd.read_csv('2024_tags.csv',sep=';')
# df_headline = pd.concat([df_2023, df_2024], axis=0)
df_2024.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва'}, inplace=True)
df_2024['Теги'] = df_2024['Теги'].apply(extract_tag) 

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')
df_headline = pd.concat([df_headline, df_2024], axis=0)

# оценка 4-5, тональгнось негативная - сколько таких в %

# print(df_headline.columns)

# print(df_headline[['Оценка', 'Тональность отзыва']].sample(7))

sample1 = df_headline[(df_headline['Оценка'] > 3) & (df_headline['Тональность отзыва'] =='Негативная') ]
print(sample1[['Оценка', 'Тональность отзыва']].sample(7))

print('%')
print(len(sample1)/len(df_headline))


sample2 = df_headline[(df_headline['Оценка'] < 3) & (df_headline['Тональность отзыва'] =='Позитивная') ]
print(sample2[['Оценка', 'Тональность отзыва']].sample(7))

print('%')
print(len(sample2)/len(df_headline))

print()
print(len(df_headline))