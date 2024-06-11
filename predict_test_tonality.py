import pandas as pd
import numpy as np
import fasttext
from gensim.utils import simple_preprocess
from sklearn.metrics import precision_recall_fscore_support, classification_report

import warnings
warnings.filterwarnings("ignore")

fasttext.FastText.eprint = lambda x: None


def join_words(text):
    interm =  ','.join(text)
    return interm.replace(",", " ")

def extract_tag(tag):
    if len(tag) > 5:
        return tag.split("value': '",1)[1][:-3]



def process_tonality(df_headline):
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
df_headline = process_tonality(df_headline)


model = fasttext.load_model('models/tonality_model_13_june.bin')

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

# X_test = pd.read_csv('X_test_tonality.csv',header=None)
# # X_test['Текст отзыва'] = X_test['Текст отзыва'].values
# print('X_test')
# print(X_test.head())
# y_test = pd.read_csv('y_test_tonality.csv',  header=None)
# # y_test['Тональность отзыва'] = y_test['Тональность отзыва'].astype(int)
# # y_test = y_test['Тональность отзыва'].values
# print('y_test')
# print(y_test.head())

with open('test_index_tonality.txt') as f:
    test_index = [int(x.strip('\n')) for x in f]

tonality_map = {1 : 'pos', 2 : 'neu', 3 : 'neg'}
df_headline['Тональность отзыва'] = df_headline['Тональность отзыва'].map(tonality_map)

X_test = df_headline[df_headline.index.isin(test_index)]['Текст отзыва']
y_test = df_headline[df_headline.index.isin(test_index)]['Тональность отзыва']

print('TEST')
print(df_headline['Тональность отзыва'].value_counts())

s = "при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно"

print('SINGLE PREDICT')
print(s)
labels = model.predict(s, k = 1)
print("PREDICTED LABEL")
print(labels)

# print(X_test)
model_predictions = predict(model, X_test)
model1_report = classification_report(y_test, model_predictions)
print(model1_report)
print()


# print('X_test')
# print(X_test[:3])
indexes = X_test.index
# print('indexes ', indexes[:3])
# print(df_headline.loc[indexes]["Текст отзыва"].values)

df = pd.DataFrame({'pred' : model_predictions, 'true' : y_test, 'text' : X_test})
# df['text'] = df_headline.loc[indexes]["Текст отзыва"]
# print("RESULTS")
# print(df.sample(7))


error = df[df['pred'] != df['true']]
print("% of errors in the test")
print(len(error)/len(df))

# print('error')
# print(error['true'].value_counts())
error.to_csv('errors/error_tonality.csv', index=False)


for s in np.unique(error['true']):
    try:
        sample = error[error['true'] == s]
        sample.to_csv(f'errors/{s}_error.csv')
    except:
        pass
