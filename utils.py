import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess


def extract_tag(tag):
    if len(tag) > 5:
        return tag.split("value': '",1)[1][:-3]


# def create_tags(review):
#     if type(review) == str:

#         if "очеред" in review or "долго" in review or "медлен" in review or "жда" in review or 'ожидан' in review:
#             return 'очередь'
#         elif 'быстр' in review or 'скорость' in review or 'времен' in review or \
#         'минут' in review or 'чac' in review:
#             return 'скорость работы'
#         elif 'навяз' in review or 'впарив' in review or 'самовольно' in review :
#             return 'навязывание_продуктов'
#         elif 'мошенн' in review or 'обман' in review or 'утечк' in review or 'утекл' in review \
#         or 'врань' in review or 'врать' in review or 'воровств' in review \
#         or 'заблужд' in review or 'развод' in review :
#             return 'мошенничество'
#         elif 'сотрудн' in review or 'грамотн' in review or 'вежлив' in review or \
#         'персонал' in review оr 'коллектив' in review or 'менеджер' in review or 'обслужив' in review оr 'работник' in review or 'специалист' in review \
#         or 'девочк' in review or 'оперативно' in review or 'помorли' in review \
#         or 'ребят' in review or 'груб' in review or 'администр' in review \
#         or 'отношен' in review or 'неуважен' in review or 'xaмск' in review or \
#         'хамств' in review or 'хамят' in review or "надмен" in review or \
#         'женщин' in review or 'руководств' in review or 'высокомерн' in review \
#         or 'девушк' in review or 'консультант' in review or 'компетентн' in review \
#         or 'общаться' in review or 'клиент' in review:
#             return "сотрудники"
#         elif 'пенс' in review:
#             return 'пенсонеры'
#         elif 'вклад' in review:
#             return 'вклады'
#         elif 'приложен' in review:
#             return 'приложение'
#         elif 'график' in review or 'суббот' in review or 'воскресен' in review or 'будни' in review \
#         or 'круглосуточн' in review:
#             return "график работы"
#         elif "кредит" in review:
#             return "кредит"
#         elif 'ипоте' in review:
#             return "ипотека"
#         elif 'карт' in review:
#             return "карта"
#         elif 'касс' in review or 'валют' in review:
#             return 'кассовое_обслуживаниеэ'
#         elif 'комф' in review or 'удобн' in review or 'уютн' in review or 'площадь' in review \
#         or 'охран' in review or 'офис' in review or 'кондицион' in review \
#         or 'здани' in review or 'ремонт' in review or 'доступност' in review \
#          or 'расположен' in review:
#             return None
#         # TODO: дописать 

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
        elif 'скорость' in tag:
            return 'скорость работы'
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

        # elif 'благодарность общая' in tag or 'качество обслуживания' in tag or 'кассовое обслуживание' in tag or 'сотрудник' in tag:
        # # elif all(x in tag for x in ['благодарность общая', 'качество обслуживания', 'кассовое обслуживание', 'сотрудник']):
        #     return 'обслуживание сотрудников'
        # elif 'перевод_пенсии' in tag or 'автокредит' in tag or 'вклад' in tag or 'инвестиции' in tag or 'страхование' in tag or 'переводы' in tag:
        # # elif all(x in tag for x in ['перевод_пенсии', 'автокредит', 'вклад', 'инвестиции', 'страхование', 'переводы']):
        #     return 'продукты'

        else:
            return tag
    else:
        return tag

def join_words(text):
    interm =  ','.join(text)
    return interm.replace(",", " ")


def process_classes(df_headline):

    # TODO: удалять
    with open('stop-words-ru.txt') as f:
        russian_stopwords = [x.strip('\n') for x in f]

    # Проверяем количество переменных и наблюдений
    df_headline = df_headline.drop_duplicates()
    df_headline = df_headline.dropna(subset=['Текст отзыва', 'Теги'])
    print(df_headline.shape)


    # Отобразим примеры заголовков
    # print('SOURCE')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))

    df_headline['Теги'] = df_headline['Теги'].apply(rename_tags) 
    df_headline['Теги'] = df_headline['Теги'].str.replace("благодарность - " ,"") 
    df_headline['Теги'] = df_headline['Теги'].str.replace(" " ,"_") 

    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].str.replace('\n', ' ')
    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(simple_preprocess) 
    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(join_words) 
    # print('PROCESSED')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))

    vc = df_headline['Теги'].value_counts()
    selected_classes = list(vc[vc > 100].keys())

    print(f'selected {len(selected_classes)} classes')


    # print(f"ALL CLASSES LEN {len(selected_classes)}")
    # print(selected_classes)
    # df_headline = df_headline[df_headline['Теги'].isin(selected_classes)]

    df_headline['Теги']  = np.where(df_headline['Теги'].isin(selected_classes), df_headline['Теги'], 'без_тематики')


    skip = ['качество_обслуживания', 'благодарность_общая'] # , 'без_тематики'
    df_headline = df_headline[~df_headline['Теги'].isin(skip)]

    return df_headline




def process(df_headline):

    # TODO: удалять
    with open('stop-words-ru.txt') as f:
        russian_stopwords = [x.strip('\n') for x in f]
    # Проверяем количество переменных и наблюдений
    df_headline = df_headline.drop_duplicates()
    df_headline = df_headline.dropna(subset=['Текст отзыва', 'Тональность отзыва'])
    print(df_headline.shape)

    df_headline = df_headline[df_headline['Тональность отзыва'] != 0]

    # df_headline['Тональность отзыва'] = df_headline['Тональность отзыва'].astype('int')


    # Отобразим примеры заголовков
    # print('SOURCE')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))

    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].str.replace('\n', ' ')
    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(simple_preprocess) 
    df_headline['Текст отзыва'] = df_headline['Текст отзыва'].apply(join_words) 
    # print('PROCESSED')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))

    return df_headline

