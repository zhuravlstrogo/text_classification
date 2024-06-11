import pandas as pd
import numpy as np
import fasttext
from gensim.utils import simple_preprocess
from sklearn.metrics import precision_recall_fscore_support, classification_report


def rename_tags(tag):
    if type(tag) == str:
        if 'навязыван' in tag:
            return 'навязывание_продуктов'
        elif 'благодарность общая' in tag:
            return 'благодарность_общая'
        elif 'комфорт' in tag:
            return 'комфорт'
        elif 'банкомат' in tag:
            return 'банкоматы'
        elif 'очеред' in tag:
            return 'очередь'
        elif 'ВТБ' in tag:
            return 'мобайл/онлайн'
        elif 'сотрудн' in tag:
            return 'сотрудники'
        elif 'график' in tag:
            return 'график_работы'
        elif 'карт' in tag:
            return 'карта'
        elif 'парков' in tag:
            return 'парковка'
        else:
            return tag
    else:
        return tag

def join_words(text):
    interm =  ','.join(text)
    return interm.replace(",", " ")

def extract_tag(tag):
    if len(tag) > 5:
        return tag.split("value': '",1)[1][:-3]



df_2024 = pd.read_csv('2024_tags.csv',sep=';')
# df_headline = pd.concat([df_2023, df_2024], axis=0)
df_2024.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва'}, inplace=True)
df_2024['Теги'] = df_2024['Теги'].apply(extract_tag) 

generated_data = pd.read_csv('generated_data.csv')

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')

#, generated_data
df_headline = pd.concat([df_headline, df_2024], axis=0)

skip = ['качество_обслуживания', 'благодарность_общая', 'без_тематики']
df_headline = df_headline[~df_headline['Теги'].isin(skip)]

df_headline = df_headline.drop_duplicates()
df_headline = df_headline.dropna(subset=['Текст отзыва', 'Теги'])
print(df_headline.shape)



# Отобразим примеры заголовков
# print('SOURCE')
# print(df_headline[['Текст отзыва', 'Теги']].head(3))


df_headline['Теги'] = df_headline['Теги'].apply(rename_tags) 
df_headline['Теги'] = df_headline['Теги'].str.replace("благодарность - " ,"") 
df_headline['Теги'] = df_headline['Теги'].str.replace(" " ,"_") 

df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(simple_preprocess) 
df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(join_words) 

with open('test_index.txt') as f:
    test_index = [int(x.strip('\n')) for x in f]

X_test = df_headline[df_headline.index.isin(test_index)]['Текст отзыва']
y_test = df_headline[df_headline.index.isin(test_index)]['Теги']


model = fasttext.load_model('models/model_13_june_1.bin')
# model = fasttext.load_model(''models/model-10_june_3.bin')

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
print('%')
print(len(error)/len(df))

print('error')
print(error['true'].value_counts())
error.to_csv('error.csv', index=False)


for s in np.unique(error['true']):
    try:
        sample = error[error['true'] == s]
        sample.to_csv(f'{s}_error.csv')
    except:
        pass
