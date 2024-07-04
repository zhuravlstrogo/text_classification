import pandas as pd
import numpy as np

from gensim.utils import simple_preprocess


# emojis = pd.read_csv('emoji_vc_mean_rating.csv')
# emojis = emojis[emojis['mark'].notnull()]
# emojis_dict = dict(emojis[['Smiley', 'mark']].values)

# def replace_emoji(text):
#    indexes = []
#    text_emoji = ''
#    for match in re.finditer(r'[^\w\s,]', text):
#       indexes.append(match.start())
#    for i in indexes:
#       try:
#          emoji =text[i]
#          text_emoji += ' ' + emojis_dict[emoji]
#       except:
#          pass 
#    text = "".join([char for idx, char in enumerate(text) if idx not in indexes]) + text_emoji 

#    return text 

def replace_emoji(text):
   indexes = []
   text_emoji = ''
   for match in re.finditer(r'[^\w\s,]', text):
      indexes.append(match.start())
   for i in indexes:
      try:
         emoji =text[i]
         text_emoji += ' ' + unicodedata.name(emoji)
      except:
         pass 
   text = "".join([char for idx, char in enumerate(text) if idx not in indexes]) + text_emoji 

   return text 


def extract_tag(tag):
    if len(tag) > 5:
        word_list = tag.split("'")[1::2]
        return list(filter(lambda a: (a!='id') & (a!='value'), word_list))


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
        if 'навязыван' in tag or 'мошеннич' in tag:
            return 'навязывание_продуктов_и_мошенничество'
        elif 'очеред' in tag or 'скорость' in tag:
            return 'загруженность_офиса'
        elif 'благодарность общая' in tag:
            return 'благодарность_общая'
        elif 'комфорт' in tag or 'парков' in tag:
            return 'комфорт'
        elif 'кассовое' in tag or 'валют' in tag:
            return 'кассовое_обслуживание'
        elif 'банкомат' in tag:
            return 'банкоматы'
        elif 'ВТБ' in tag:
            return 'мобайл/онлайн'
        elif 'сотрудн' in tag:
            return 'сотрудники'
        elif 'график' in tag:
            return 'график_работы'
        elif 'карт' in tag or 'кредит' in tag or 'ипотек' in tag or 'вклад' in tag or 'страхов' in tag:
            return 'продукты'
        else:
            return tag
    else:
        return tag

def join_words(text):
    interm =  ','.join(text)
    return interm.replace(",", " ")


def process_text(df_headline, text_col='Текст отзыва'):
    """для тем"""
    # TODO: удалять
    with open('stop-words-ru.txt') as f:
        russian_stopwords = [x.strip('\n') for x in f]

    df_headline['text_processed'] = df_headline[text_col].astype(str)
    df_headline['text_processed'] = df_headline['text_processed'].str.replace('\n', ' ')
    df_headline['text_processed'] = df_headline['text_processed'].apply(simple_preprocess) 
    df_headline['text_processed'] = df_headline['text_processed'].apply(replace_emoji) 
    df_headline['text_processed'] = df_headline['text_processed'].apply(join_words) 

    return df_headline


def process_classes(df_headline, tags_col = 'Тег 1'):
    """для тем"""
    df_headline[tags_col] = df_headline[tags_col].astype(str)
    df_headline[tags_col] = df_headline[tags_col].apply(rename_tags) 
    df_headline[tags_col] = df_headline[tags_col].str.replace("благодарность - " ,"") 
    df_headline[tags_col] = df_headline[tags_col].str.replace(" " ,"_") 

    return df_headline


def process_tonality(df_headline):
    """для тональности"""
    # TODO: удалять
    with open('stop-words-ru.txt') as f:
        russian_stopwords = [x.strip('\n') for x in f]
    # Проверяем количество переменных и наблюдений
    df_headline = df_headline.drop_duplicates()
    df_headline = df_headline.dropna(subset=['Текст отзыва', 'Тональность отзыва'])
    print(df_headline.shape)

    df_headline = df_headline[df_headline['Тональность отзыва'] != 0]

    df_headline['text_processed'] = df_headline['Текст отзыва'].astype(str)
    df_headline['text_processed'] = df_headline['text_processed'].str.replace('\n', ' ')
    df_headline['text_processed'] = df_headline['text_processed'].apply(replace_emoji) 
    df_headline['text_processed'] = df_headline['text_processed'].apply(simple_preprocess) 
    df_headline['text_processed'] = df_headline['text_processed'].apply(join_words) 

    df_headline = df_headline[df_headline['text_processed'] != '']
    # print('PROCESSED')
    # print(df_headline[['Текст отзыва', 'Теги']].head(3))

    return df_headline


def predict_by_rating(rating):
    try:
        rating = int(rating)
        if rating > 3:
            return 'pos'
        elif rating == 3:
            return 'neu'
        elif rating < 3:
            return 'neg'
        else:
            return None
    except:
        return None