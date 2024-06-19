import pandas as pd
import numpy as np

import fasttext
from sklearn.metrics import classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import warnings
warnings.filterwarnings("ignore")

pd.options.display.max_colwidth = 1000

from utils import extract_tag, create_tags, rename_tags, join_words, process_classes

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

# df_2023 = pd.read_csv('2023_tags.csv',sep=';')
df_2024 = pd.read_csv('2024_tags.csv',sep=';')
# df_headline = pd.concat([df_2023, df_2024], axis=0)
df_2024.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва'}, inplace=True)
df_2024['Теги'] = df_2024['Теги'].apply(extract_tag) 

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')

df_headline = pd.concat([df_headline, df_2024], axis=0)
df_headline = process_classes(df_headline)


# label = 'счет'
# l = df_headline[df_headline['Теги'] == label]
# print(f'before: {len(l)}')

# отложить честный test
#  стратификация 
X = df_headline['Текст отзыва']
y = df_headline['Теги']
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)

for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(f'train size: {len(X_train)}, {len(y_train)}')
print(f'test size: {len(X_test)}, {len(y_test)}')

# with open("train_index.txt", 'w') as output:
#     for row in list(X_train.index):
#         output.write(str(row) + '\n')

with open("test_index.txt", 'w') as output:
    for row in list(X_test.index):
        output.write(str(row) + '\n')

df_headline = df_headline.loc[~df_headline.index.isin(test_index)]
N = len(df_headline)
print(f'CHECK: {N == len(X_train)}')

generated_data = pd.read_csv('generated_data.csv')
generated_data = process_classes(generated_data)
generated_data = generated_data[generated_data['Теги'] != 'очередь']

df_headline = pd.concat([df_headline, generated_data], axis=0)
print(f'{len(df_headline) - N} generated data added')
print(f'TOTAL LEN {len(df_headline)} ')

# df_headline.to_csv('train.csv')
# 1/0

# l = df_headline[df_headline['Теги'] == label]
# print(f'after: {len(l)}')

print('value_counts')
print(df_headline['Теги'].value_counts())

X_train = df_headline['Текст отзыва']
y_train = df_headline['Теги']

# print('X_train')
# print(X_train.head())

# print('y_train')
# print(y_train.head())


# Создадим текстовые файля для обучения модели с лейблом и текстом
with open('train.txt', 'w') as f:
    for each_text, each_label in zip(X_train, y_train):
        f.writelines(f'__label__{each_label} {each_text}\n')
        

# модель
model1 = fasttext.train_supervised('train.txt', epoch=750, lr=1.0, wordNgrams =2)
model_name = "model_17_june_1.bin"
model1.save_model("models/" + model_name)
print(f' {model_name} saved')

s = "при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно"
labels = model1.predict(s, k = 1)

# # Создадим функцую для отображения результатов обучения модели
# def print_results(sample_size, precision, recall):
#     precision   = round(precision, 2)
#     recall      = round(recall, 2)
#     print(f'{sample_size=}')
#     print(f'{precision=}')
#     print(f'{recall=}')

# # Применяем функцию
# # print_results(*model1.test('test.txt'))


# # print(*model1.test('test.txt'))
# print('SINGLE PREDICT')
# print(model1.predict("при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно", k = 3))




# def predict(model, test_data):

#     predictions = []
#     for idx, v in test_data.items():
#     # for idx, sample in test_data.iterrows():
#         prediction = model.predict(v)
#         # prediction = prediction[0][0]
#         prediction = prediction[0][0].replace("__label__", "")
#         predictions.append(prediction)
        
#     return predictions

# print('*************************')



# # print(X_test)
# model_predictions = predict(model1, X_test)

# print(f'classification_report')
# model1_report = classification_report(y_test, model_predictions)
# print(model1_report)
# print()

# # print('X_test')
# # print(X_test[:3])
# indexes = X_test.index
# # print('indexes ', indexes[:3])
# # print(df_headline.loc[indexes]["Текст отзыва"].values)

# df = pd.DataFrame({'pred' : model_predictions, 'true' : y_test, 'text' : X_test})
# # df['text'] = df_headline.loc[indexes]["Текст отзыва"]
# # print("RESULTS")
# # print(df.sample(7))



# error = df[df['pred'] != df['true']]
# print('%')
# print(len(error)/len(df))

# print('error')
# print(error['true'].value_counts())
# error.to_csv('error.csv', index=False)

