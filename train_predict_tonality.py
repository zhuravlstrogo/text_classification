import pandas as pd
import numpy as np
import fasttext
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import re

from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings("ignore")

from utils import extract_tag, join_words, process_tonality

pd.options.display.max_colwidth = 1000



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

tonality_map = {'pos' : 1, 'neg' : 3, 'neu' : 2}
df_headline['Тональность отзыва'] = df_headline['Тональность отзыва'].map(tonality_map)



X_processed = process_tonality(df_headline)
X_processed = X_processed.dropna(subset=['Тональность отзыва', 'text_processed'])
X_processed = X_processed.drop_duplicates(subset=['Тональность отзыва', 'text_processed'])
X_processed = X_processed[X_processed['text_processed'].notnull()]




# стратификация по дате
X = X_processed[['text_processed', 'Тональность отзыва', 'Дата публикации отзыва', 'Оценка']]
# TODO: убирать, где оценка не матчится с тональностью
X = X[((X['Тональность отзыва'] == 'neg') & (X['Оценка'] < 3)) | ((X['Тональность отзыва'] == 'pos') & (X['Оценка'] > 3)) ]

split_date = '2024-01-20'
target_col = 'Тональность отзыва'

X_train = X[X['Дата публикации отзыва'] < split_date]['text_processed']
y_train = X[X['Дата публикации отзыва'] < split_date][target_col]

X_test = X[X['Дата публикации отзыва'] >= split_date]['text_processed']
y_test = X[X['Дата публикации отзыва'] >= split_date][target_col]

# TODO: print len test 

# отложить честный test
#  стратификация 
# X = df_headline['Текст отзыва']
# y = df_headline['Тональность отзыва']

# sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)

# for i, (train_index, test_index) in enumerate(sss.split(X, y)):
#     # print(f"Fold {i}:")
#     # print(f"  Train: index={train_index}")
#     # print(f"  Test:  index={test_index}")
    
#     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
#     y_train, y_test = y.iloc[train_index], y.iloc[test_index]

# print(f'train size: {len(X_train)}, {len(y_train)}')
# print(f'test size: {len(X_test)}, {len(y_test)}')

# with open("test_index_tonality.txt", 'w') as output:
#     for row in list(X_test.index):
#         output.write(str(row) + '\n')

generated_data = pd.read_csv('generated_tonality.csv')
generated_data = process_tonality(generated_data)

# generated_data = generated_data[generated_data['Теги'] != 'очередь']

# Я ТУТ 
X_train  = pd.concat([X_train, generated_data['text_processed']], axis=0)
y_train  = pd.concat([y_train, generated_data['Тональность отзыва']], axis=0)

intersection_text= list(set(X_train.to_list()).intersection(set(X_test.to_list())))
print(len(intersection_text))


dual_train_idx = []
dual_test_idx = []

for i in intersection_text:
    X_train_idx = X_train[X_train == i].index[0]
    dual_train_idx.append(X_train_idx)

    X_test_idx = X_test[X_test == i].index[0]
    dual_test_idx.append(X_test_idx)

X_train = X_train.loc[~X_train.index.isin(dual_train_idx)]
y_train = y_train.loc[~y_train.index.isin(dual_train_idx)]

X_test = X_test.loc[~X_test.index.isin(dual_test_idx)]
y_test = y_test.loc[~y_test.index.isin(dual_test_idx)]


intersection_text= list(set(X_train.to_list()).intersection(set(X_test.to_list())))
print(len(intersection_text))
# TODO: assert 

# Создадим текстовые файля для обучения модели с лейблом и текстом
with open('train_tonality.txt', 'w') as f:
    for each_text, each_label in zip(X_train, y_train):
        f.writelines(f'__label__{each_label} {each_text}\n')
        

# модель
model1 = fasttext.train_supervised('train_tonality.txt', epoch=500, lr=1.0, wordNgrams =2)
model1.save_model("models/tonality_model_july.bin")
print('model saved')

s = "при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно"

print('SINGLE PREDICT')
print(s)
labels = model1.predict(s, k = 2)
print(labels)