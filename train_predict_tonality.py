import pandas as pd
import numpy as np
import fasttext
from sklearn.metrics import precision_recall_fscore_support, classification_report
from sklearn.model_selection import StratifiedShuffleSplit
import re

from gensim.utils import simple_preprocess
import warnings
warnings.filterwarnings("ignore")

from utils import extract_tag, join_words

pd.options.display.max_colwidth = 1000




def process(df_headline):
    # Проверяем количество переменных и наблюдений
    df_headline = df_headline.drop_duplicates()
    df_headline = df_headline.dropna(subset=['Текст отзыва', 'Тональность отзыва'])
    print(df_headline.shape)

    df_headline = df_headline[df_headline['Тональность отзыва'] != 0]

    # df_headline['Тональность отзыва'] = df_headline['Тональность отзыва'].astype('int')


    # Отобразим примеры заголовков
    # print('SOURCE')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))


    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(simple_preprocess) 
    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(join_words) 
    # print('PROCESSED')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))

    return df_headline

# TODO: удалять
with open('stop-words-ru.txt') as f:
    russian_stopwords = [x.strip('\n') for x in f]


# Загружаем данные
# df_2023 = pd.read_csv('2023_tags.csv',sep=';')
df_2024 = pd.read_csv('2024_tags.csv',sep=';')
# df_headline = pd.concat([df_2023, df_2024], axis=0)

df_2024.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва', 'text_tonality' : 'Тональность отзыва'}, inplace=True)
df_2024['Теги'] = df_2024['Теги'].apply(extract_tag) 

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')

tonality_map = {'Позитивная' : 1, 'Негативная' : 3, 'Нейтральная' : 2}
df_headline['Тональность отзыва'] = df_headline['Тональность отзыва'].map(tonality_map)

df_headline = pd.concat([df_headline, df_2024], axis=0)
df_headline = process(df_headline)


print(df_headline['Тональность отзыва'].value_counts())


# отложить честный test
#  стратификация 
X = df_headline['Текст отзыва']
y = df_headline['Тональность отзыва']
sss = StratifiedShuffleSplit(n_splits=5, test_size=0.25, random_state=0)

for i, (train_index, test_index) in enumerate(sss.split(X, y)):
    # print(f"Fold {i}:")
    # print(f"  Train: index={train_index}")
    # print(f"  Test:  index={test_index}")
    
    X_train, X_test = X.iloc[train_index], X.iloc[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

print(f'train size: {len(X_train)}, {len(y_train)}')
print(f'test size: {len(X_test)}, {len(y_test)}')

with open("test_index_tonality.txt", 'w') as output:
    for row in list(X_test.index):
        output.write(str(row) + '\n')


# X_test.to_csv('X_test_tonality.csv')
# y_test.to_csv('y_test_tonality.csv')


X_train = df_headline['Текст отзыва']
y_train = df_headline['Тональность отзыва']


# Создадим текстовые файля для обучения модели с лейблом и текстом
with open('train_tonality.txt', 'w') as f:
    for each_text, each_label in zip(X_train, y_train):
        f.writelines(f'__label__{each_label} {each_text}\n')
        

# модель
model1 = fasttext.train_supervised('train_tonality.txt', epoch=500) #, lr=1.0, wordNgrams =2)
model1.save_model("models/tonality_model_11_june.bin")
print('model saved')

s = "при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно"

print('SINGLE PREDICT')
labels = model1.predict(s, k = 2)

# # Создадим функцую для отображения результатов обучения модели
# def print_results(sample_size, precision, recall):
#     precision   = round(precision, 2)
#     recall      = round(recall, 2)
#     print(f'{sample_size=}')
#     print(f'{precision=}')
#     print(f'{recall=}')

# # Применяем функцию
# # print_results(*model1.test('test_tonality.txt'))


# print(*model1.test('test_tonality.txt'))




def predict(model, test_data):

    predictions = []
    for idx, v in test_data.items():
    # for idx, sample in test_data.iterrows():
        prediction = model.predict(v)
        # prediction = prediction[0][0]
        prediction = prediction[0][0].replace("__label__", "")
        predictions.append(prediction)
        
    return predictions

print('*************************')



# print(X_test)
model_predictions = predict(model1, X_test)

print(f'classification_report')
model1_report = classification_report(y_test, model_predictions)
print(model1_report)
print()

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

