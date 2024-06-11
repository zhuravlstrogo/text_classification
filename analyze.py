import pandas as pd
import numpy as np

from utils import extract_tag

def analyze_text(text):
    # Разделение текста на слова
    words = text.split()

    # Создание словаря для подсчета частоты слов
    word_frequency = {}

    for word in words:
        # Убираем знаки препинания и приводим к нижнему регистру
        cleaned_word = word.strip('.,!?()[]').lower()

        # Подсчет частоты слов
        if cleaned_word in word_frequency:
            word_frequency[cleaned_word] += 1
        else:
            word_frequency[cleaned_word] = 1

    # Вывод частоты слов
    # for word, frequency in word_frequency.items():
    #     print(f"{word}: {frequency}")
    return pd.DataFrame(word_frequency.items(), columns=['word', 'freq'])

df_2024 = pd.read_csv('2024_tags.csv',sep=';')
# df_headline = pd.concat([df_2023, df_2024], axis=0)
df_2024.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва'}, inplace=True)
df_2024['Теги'] = df_2024['Теги'].apply(extract_tag) 

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')
df_headline = pd.concat([df_headline, df_2024], axis=0)
df_headline = df_headline.dropna(subset='Текст отзыва')

input_text = ' '.join(df_headline['Текст отзыва'].to_list())

# Вызов функции для анализа текста
word_frequency = analyze_text(input_text)
# print(word_frequency.sort_values('freq', ascending=False)[:10])
print(word_frequency[word_frequency['freq'] < 10000].sort_values('freq', ascending=False)[:20])

# TODO: добавить в стоп словарь  < 6000?