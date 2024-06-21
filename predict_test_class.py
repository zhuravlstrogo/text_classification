import pandas as pd
import numpy as np
import fasttext
from gensim.utils import simple_preprocess
from sklearn.metrics import precision_recall_fscore_support, classification_report

import warnings
warnings.filterwarnings("ignore")

fasttext.FastText.eprint = lambda x: None

from utils import rename_tags, join_words, extract_tag, process_classes


df_2024 = pd.read_csv('2024_tags.csv',sep=';')
# df_headline = pd.concat([df_2023, df_2024], axis=0)
df_2024.rename(columns={'tags' : 'Теги', 'text_author': 'Текст отзыва'}, inplace=True)
df_2024['Теги'] = df_2024['Теги'].apply(extract_tag) 

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')
df_headline = pd.concat([df_headline, df_2024], axis=0)

df_headline = process_classes(df_headline)
print(df_headline.shape)

# print(df_headline['Теги'].value_counts())

with open('test_index.txt') as f:
    test_index = [int(x.strip('\n')) for x in f]

print(np.unique(df_headline['Теги']))
print('***************')
X_test = df_headline[df_headline.index.isin(test_index)]['Текст отзыва']
y_test = df_headline[df_headline.index.isin(test_index)]['Теги']



print('X_test')
print(X_test.sample(3))

print('y_test')
print(y_test[:3])
# print(X_test['Теги'].value_counts())

model_name = 'models/model_17_june_1.bin'
# model_name = 'models/model_13_june_3.bin'
# model_name =  'models/model-10_june_3.bin'
print(f"I will inference {model_name}")
model = fasttext.load_model(model_name)

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

df.to_csv('test_with_predict.csv')


error = df[df['pred'] != df['true']]
print("% of errors in the test")
print(len(error)/len(df))

# print('error')
# print(error['true'].value_counts())
error.to_csv('error.csv', index=False)


for s in np.unique(error['true']):
    try:
        sample = error[error['true'] == s]
        sample.to_csv(f'errors/{s}_error.csv')
    except:
        pass
