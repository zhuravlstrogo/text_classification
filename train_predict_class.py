import pandas as pd
import numpy as np

import fasttext
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_colwidth = 1000

from utils import extract_tag, rename_tags, join_words, process_classes, replace_emoji

# TODO: очереди, сотрудники, скорость, комфорт регуляркой помечать?
# TODO: посмотреть топ слов по встречаемости, добавить их в словарь стоп слов
# TODO: исправлять опечатки 

# TODO: добавлять веса к отдельным значимым словам, чтобы их влияние в тексте было больше 
# TODO: razdel
# TODO: убрать ошибки
# tag_by_review = df.groupby(['Номер филиала', 'Текст отзыва', 'Дата написания отзыва', 'Автор отзыва'], as_index=False).agg(
#     tag_count=('Теги', 'count'))
# tag_by_review[tag_by_review['tag_count'] == 2]
# TODO: несколько меток - - навязывание продуктов, страховка 
# TODO: мошенничество + навязывание продуктов

# TODO: в модель определения темы добавить предсказание тональности 





# with open('stop-words-ru.txt') as f:
#     russian_stopwords = [x.strip('\n') for x in f]


# посмотри ещё
# https://medium.com/@hseymakoc/fasttext-doc2vec-for-text-classification-d07fde87fe10

# параметры модели 
# https://sysblok.ru/nlp/kak-rabotaet-fasttext-i-gde-ee-primenjat/

# исходный код
# https://proglib.io/p/prakticheskoe-rukovodstvo-po-nlp-izuchaem-klassifikaciyu-tekstov-s-pomoshchyu-biblioteki-fasttext-2021-08-28

# Загружаем данные

df_headline = pd.read_csv('comm_3650.csv',sep=';')
df_headline = df_headline.dropna(subset=['id'])

df_headline = df_headline[df_headline['published_at_author'] >= '2017-01-01']

df_headline = df_headline[df_headline['tags'] != '[]']
df_headline.drop(['text', 'published_at'], axis=1, inplace=True)

df_headline.rename(columns={
'author':'Автор',
'company_uuid':'Номер филиала',
'id':'ID отзыва',
'provider_id':'ID платформы',
'published_at_author':'Дата публикации отзыва',
'rating':'Оценка',
'text_author':'Текст отзыва',
'reply':'Ответ',
'text_tonality':'Тональность отзыва',
'tags':'Теги'}, inplace=True)

df_headline['Теги'] = df_headline['Теги'].apply(extract_tag)

val = df_headline['Теги'].apply(len).max()
cols = ['Тег ' + str(i) for i in range(1, val+1)]
df_headline[cols] = df_headline['Теги'].apply(pd.Series).fillna('')



df_headline = df_headline[df_headline['Тег 2'] == '']
X_processed = process_classes(df_headline)

print('columns')
print(df_headline.columns)

cols = ['Дата публикации отзыва', 'text_processed', 'Тег 1']
skip = ['качество_обслуживания', 'благодарность_общая', 'без_тематики']

X_processed = X_processed[~X_processed['Теги'].isin(skip)][cols]

# TODO: добавить код из create tags

# TODO: X_processed = pd.concat([X_processed, tagged_reviews], axis=0)


X_processed = process_text(X_processed)

X_processed = X_processed.drop_duplicates(subset=['Текст отзыва', 'Теги'])
X_processed = X_processed.dropna(subset=['text_processed', 'Теги'])
X_processed = X_processed[X_processed['Теги'].notnull()]
X_processed = X_processed[X_processed['text_processed'].notnull()]

vc = X_processed['Теги'].value_counts()
selected_classes = list(vc[vc > 500].keys())
print(f'selected_classes {len(selected_classes)}')

X_processed['Теги'] = np.where(X_processed['Теги'] .isin(selected_classes), X_processed['Теги'], 'без_тематики')


# стратификация по дате
X = X_processed[['text_processed', 'Теги', 'Дата публикации отзыва']]

split_date = '2024-01-20'
target_col = 'Теги'

X_train = X[X['Дата публикации отзыва'] < split_date]['text_processed']
y_train = X[X['Дата публикации отзыва'] < split_date][target_col]

X_test = X[X['Дата публикации отзыва'] >= split_date]['text_processed']
y_test = X[X['Дата публикации отзыва'] >= split_date][target_col]

# TODO: print len test 

# # TODO: отложить честный test
# X = df_headline['Текст отзыва']
# y = df_headline['Теги']
# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)

# for i, (train_index, test_index) in enumerate(sss.split(X, y)):
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# print(f'train size: {len(X_train)}, {len(y_train)}')
# print(f'test size: {len(X_test)}, {len(y_test)}')

# TODO: сплит не работает 
# with open("test_index.txt", 'w') as output:
#     for row in list(X_test.index):
#         output.write(str(row) + '\n')

# df_headline = df_headline.loc[~df_headline.index.isin(test_index)]
# N = len(df_headline)
# print(f'CHECK: {N == len(X_train)}')

generated_data = pd.read_csv('generated_data.csv')
generated_data = process_classes(generated_data)
generated_data = process_texr(generated_data)
generated_data = generated_data[generated_data['Теги'].isin(selected_classes)]
# generated_data = generated_data[generated_data['Теги'] != 'очередь']

# Я ТУТ 
X_train  = pd.concat([X_train, generated_data['text_processed']], axis=0)
y_train  = pd.concat([y_train, generated_data['Теги']], axis=0)
# print(f'{len(df_headline) - N} generated data added')
# print(f'TOTAL LEN {len(df_headline)} ')


print('value_counts')
print(df_headline['Теги'].value_counts())

intersection_list = list(set(X_train.to_list()).intersection(set(X_test.to_list())))
print(len(intersection_list))

for text in intersection_list:
    idx = X_train.to_list().index(text)
    X_train = X_train.drop(X_train.index[idx])
    y_train = y_train.drop(y_train.index[idx])

intersection_list = list(set(X_train.to_list()).intersection(set(X_test.to_list())))
print(len(intersection_list))


# Создадим текстовые файля для обучения модели с лейблом и текстом
with open('train.txt', 'w') as f:
    for each_text, each_label in zip(X_train, y_train):
        f.writelines(f'__label__{each_label} {each_text}\n')
        

# модель
model1 = fasttext.train_supervised('train.txt', epoch=750, lr=1.0, wordNgrams =2)
model_name = "model_4_july.bin"
model1.save_model("models/" + model_name)
print(f' {model_name} saved')

s = "при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно"
print(s)
labels = model1.predict(s, k = 1)
print(labels)
