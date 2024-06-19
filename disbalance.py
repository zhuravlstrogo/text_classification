import pandas as pd
import numpy as np
import sklearn
import re
import random

import nltk
from nltk import tokenize
from nltk.tokenize import RegexpTokenizer

def extract_tag(tag):
    if len(tag) > 5:
        return tag.split("value': '",1)[1][:-3]


def get_text_for_label(df, label): # Получим список с текстами для каждого класса
    # label = 'label == ' + str(label)
    # df_label = df.query(label).drop(['label'], axis=1)
    df_label = df[df['tags'] == label]
    return df_label.text_author.values.tolist()

def tokenize_sentences(text_corp): # Функцию разбиения текстов корпуса на токены с разбивкой по предложениям
    token_corp = []
    for text in text_corp:
        text = re.sub(r'\s+', ' ', text, flags=re.M)
        for sent in re.split(r'(?<=[.!?…])\s+', text):
            sent = sent.replace('\n',' ')
            for word in sent.split():
                token = re.search(r'[а-яёА-ЯЁa-zA-Z]+', word, re.I)
                if token is None:
                    continue
                token_corp.append(token.group().lower())
            token_corp.append('END_SENT_START') # В конце каждого предложения добавляем фиктивный токен
    return token_corp


from collections import Counter, defaultdict

def get_bigramms(token_list):
    bigramm_corp = []
    for i in range(len(token_list)-1):
        bigramm = token_list[i] + ' ' + token_list[i+1]
        bigramm_corp.append(bigramm) # Получим список биграмм
    
    unique_token_count = len(set(bigramm_corp)) # кол-во уникальных биграмм
    bigramm_proba = {} # Создаю словарик для результата: Ключ - биграмма. Значение - вероятность
     
    count_bigramm = Counter(bigramm_corp) # Создаю словарь для хранения частот биграмм
    count_token = Counter(token_list) # Создаю словарь для хранения частот токенов        
        
    # Создаю словарь с группированными биграммами: ключи – отдельные токены, 
    # в значение – список слов которые следуют за ними в корпусе 
    # с собственной частотой (  { 'отель': [('отличным', 4.116e-06), ('вобщем', 2.058e-06), …)
    grouped_bigramms = defaultdict(list)
    for bigramm in set(bigramm_corp):
        first_word, second_word = bigramm.split()
        proba = (count_bigramm[bigramm] + 1) / (count_token[first_word] + unique_token_count) # Формула Лапласа
        grouped_bigramms[first_word].append((second_word, proba))
    return grouped_bigramms

def generate_texts(token_label, grouped_bigramm, label, count_text, count_sent, count_word):

    # Создаём словарь для подсчёта биграмм "исключений"
    exceptions_bigramm = defaultdict(int)
    # Создаём список уникальных токенов для старта предложения
    unique_token = list(set(token_label))

    texts = []
    for it_text in range(count_text):  # Цикл с диапазоном кол-ва текстов
        text = ''
        unique_word = set()

        # Цикл с диапазоном кол-ва предложений в тексте
        for it_sent in range(count_sent):
            len_sent = count_word
            # Генерим случайное слово для начала предложения для обеспечения стохастического процесса генерации предложения
            start_word = random.choice(unique_token)
            # Записываем в строку с финальным предложением первое стартовое слово
            final_sent = start_word
            # Множество уникальных слов, которые уже сгенерились в предложение (чтобы геенерация не зацикливалась)
            unique_word.add(start_word)

            for step in range(count_word):
                next_word = None  # Создаём переменную для нового слова
                frequency = 0  # Переменная-счётчик для частоты каждого нового слова

                # Проходим циклом по словарю с ключом биграммы и значением её частоты
                for second_word, freq in grouped_bigramm[start_word]:
                    bigramm = start_word + ' ' + second_word
                    # Устанавливаем значение максимального повторения слова в одном тексте
                    if exceptions_bigramm[bigramm] > 3:
                        continue
                    if freq > frequency and second_word not in unique_word and second_word != 'END_SENT_START':
                        next_word = second_word
                        frequency = freq  # Если второе слово проходит условие запоминаем его
                if next_word is None:  # Если подходящего по условиям слова не найдено, перезаписываем стартовое слово и начинаем поиск заново
                    start_word = random.choice(unique_token)
                    final_sent += ', ' + start_word
                    unique_word.add(start_word)
                else:
                    # Если после цикла нашли подходящее слова (которое запомнили в цикле) - записываем его в предложение
                    exceptions_bigramm[start_word + ' ' + next_word] += 1
                    start_word = next_word
                    final_sent += ' ' + next_word
                    unique_word.add(start_word)
            final_sent += '. '
            text += final_sent
        texts.append(text)

    generation_text_df = pd.DataFrame(texts, columns=['text_author'])  # Формируем фрейм из списка
    generation_text_df['tags'] = label
    return generation_text_df[['tags', 'text_author']]


def balance_text(corp, low_thresh, high_thresh, low_remove_tresh):
    new_corp, temp = [], []
    for text in corp:
        if low_thresh <= len(text) <= high_thresh: # Если длина текста в пределах диапазона - оставляем текст без изменения
            new_corp.append(text)
        elif len(text) < low_thresh: # Если длина текста меньше нижнего порога - запоминаем, затем склеиваем с таким же текстом
            if len(temp) >= low_thresh:
                new_corp.append(temp)
                temp = text
            else:
                temp.extend(text)
        else:
            # Если длина текста больше верхнего порога - сплитим на меньшие тексты в рамках диапазона
            # Совсем мелкие хвостовые части не добавляем
            for j in range(0, len(text) - low_remove_tresh, high_thresh):
                new_corp.append(text[j:min(len(text), j + high_thresh)])
 
    if len(temp) > low_remove_tresh:
        new_corp.append(temp)
 
    return new_corp



def generate(df_headline, lablel, count_text=5000):
    """label 'сотрудники' """
    print(f'label: {label}')
    feedback_label_1 = get_text_for_label(df_headline, lablel) # К примеру, список для первого класса
    print(f'input len: {len(feedback_label_1)}')

    token_label_1 = tokenize_sentences(feedback_label_1) # К примеру, список для первого класса
    grouped_bigramm_1 = get_bigramms(token_label_1) # На примере первого класса
    label_1_df = generate_texts(token_label_1, grouped_bigramm_1, label=1, count_text=count_text, count_sent=4, count_word=15)

    label_1_df['tags'] = lablel

    print(f'output len: {len(label_1_df)}')

    return label_1_df

 
# TODO:
# list_feedback_label_1 = get_text_for_label(df_headline, 'сотрудники') # Снова получим список документов из фрейма (пример для первого класса)
# label_1_feedback_balance = balance_text(list_feedback_label_1, 50, 100, 10) # Получаем сбалансированный текст для первого класса
# print(label_1_feedback_balance[:3])
# Всё тоже проделываем для всех 5-ти классов.


# загрузка

df_headline = pd.read_csv('processed.csv')
df_headline.rename(columns={'Теги':'tags', 'Текст отзыва':'text_author'}, inplace=True)



frames = []
for label in ['ипотека', 'карта','кассовое_обслуживание', 'контактный_центр', 
'мобайл/онлайн' , 'счет', 'навязывание_продуктов', 'кредит_наличными', 
'мошенничество', 'график_работы', 'парковка','банкоматы',
'счет','комфорт','скорость_работы' ]:
    df = generate(df_headline=df_headline, lablel=label)
    frames.append(df)

if all(v is None for v in frames):
    final_df = None
else:
    final_df = pd.concat(frames, axis=0)

final_df['text_author'] = final_df['text_author'].str.replace(',', '')
final_df.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва'}, inplace=True)
print(final_df.sample(7))
final_df.to_csv('generated_data.csv', index=False)


print(final_df.shape)