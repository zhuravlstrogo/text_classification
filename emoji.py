import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
from utils import rename_tags, join_words, extract_tag, process_classes
import emoji
import emojis
import math
import time
import ast
import re
from deep_translator import GoogleTranslator
from emosent import get_emoji_sentiment_rank

def convert_size(size_bytes):
   if size_bytes == 0:
       return "0B"
#    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
   size_name = ("MB")
   i = int(math.floor(math.log(size_bytes, 1024)))
   p = math.pow(1024, i)
   s = round(size_bytes / p, 2)
   return "%s %s" % (s, size_name[i])

# text = '😊 отлично'
# # text = '👍🏽'

# clean_text = emoji.demojize(text, delimiters=(" ", " "))
# print(text, clean_text)

# e = pd.read_csv('emoji_list.csv', sep=';')
# emoji_dict = {}
# # print('1')

# unique_emoji = list(set(e['symbol'].to_list()))

# for row in unique_emoji:

#    k = re.findall(r'\(.*?\)', row)[0][1]
#    v = row.split('#')[0].split(":")[0]
#    # print('2')
#    v = GoogleTranslator(source='auto', target='ru').translate(v)
#    # time.sleep(1)
#    # print('3')
#    emoji_dict[k] = v
# print('emoji_dict created ')

# print(emoji_dict['🚫'])

# 💗

# df = pd.read_csv('comments_255000.csv')


# size_bytes = df.memory_usage(deep=True).sum()
# print(convert_size(size_bytes))


def mark(rating):
   if rating < 3:
      return 'ужасно'
   elif rating > 3:
      return 'отлично'
   else:
      return None



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


def emoji_to_unicode(emoji):
   try:
      e = unicodedata.name(emoji)
   except:
      e = emoji
   return e 


def get_emoji(text):

   e = emojis.get(text)
   if len(e) > 0:
      return list(e)
   else:
      return None


def handle(e):
   try:
      e = ast.literal_eval(e)
   except:
      e = e 
   return e 

def make_emoji_vc():

   df = pd.read_csv('comments_255000.csv')
   df = df[df['text'].notnull()]

   text = df['text'].to_list()
   text = ' '.join(text)

   out = (pd.DataFrame(emoji.emoji_list(text)).value_counts('emoji')
            .rename_axis('Smiley').rename('Count').reset_index()
            .assign(Type=lambda x: x['Smiley'].apply(emoji.demojize)))

   print(f'emoji value counts {len(out)}')

   out['emoji_text'] = out['Smiley'].apply(emoji_to_unicode)
   file_name = 'emoji_value_counts.csv'
   out.to_csv(file_name, index=False)
   print(f'{file_name} saved')
   

def add_rating_to_vc_emoji():
   emoji_rating = pd.read_csv('emoji_mean_rating.csv')
   print('emoji_mean_rating')
   # print(emoji_rating.sample())

   emoji_value_counts = pd.read_csv('emoji_value_counts.csv')
   print('emoji_value_counts')
   # print(emoji_value_counts.sample())

   # data = emoji_value_counts.join ( emoji_rating.set_index( [ 'emoji' ], verify_integrity=True ),
   #                on=[ 'Smiley' ], how='left' )

   data = emoji_value_counts.join ( emoji_rating.set_index( [ 'emoji_text' ], verify_integrity=True ),
                  on=[ 'emoji_text' ], how='left' )


   # data = pd.merge(emoji_rating, emoji_value_counts, left_on='emoji', right_on='Smiley')

   print(data.sample(7))
   data['emoji_mean_rating'] = round(data['emoji_mean_rating'], 2)
   data = data[['Smiley', 'Count', 'emoji_mean_rating']]

   data['mark'] = data['emoji_mean_rating'].apply(mark)
   data.to_csv('emoji_vc_mean_rating.csv')


def groupby_emoji_by_rating():
   df = pd.read_csv('comments_255000.csv')
   # df = df[df['published_at'] > '2024-01-01']

   df['text'] = df['text'].astype(str)
   df['emoji'] = df['text'].apply(get_emoji)

   df1 = df[df['emoji'].notnull()]
   # df1.to_csv('rows_with_emoji.csv')
   # df1 = pd.read_csv('rows_with_emoji.csv')
   df1 = df1[['emoji', 'rating']]
   df1['emoji'] = df1['emoji'].apply(handle)
   df2 = df1.explode('emoji')
   # print(df2.sample(10))


   df2['emoji_text'] = df2['emoji'].apply(emoji_to_unicode)
   # emoji_rating = df2.groupby('emoji', as_index=False)['rating'].mean()
   emoji_rating = df2.groupby('emoji_text', as_index=False).agg(
         emoji_mean_rating=('rating', 'mean')

   )
   emoji_rating.to_csv('emoji_mean_rating.csv')

# make_emoji_vc()
# groupby_emoji_by_rating()
add_rating_to_vc_emoji()



# print(len(data))
# print(len(emoji_rating))
# print(len(emoji_value_counts))

# text = df['text'].to_list()
# text = ' '.join(text)
# text = text[:1270]
# print(text)


# emoji_chars = emoji.EMOJI_ALIAS_UNICODE.values()

def _emoji(char):
   if char in emoji_chars:
      return unicodedata.name(char)

# text = '💗 error 🚫 🚫🚫' 
# new_list = emojis.get(text)
# print(new_list)







# my_dict = {}
# for e in new_list:
#    # if not isinstance(e, str):
#    try:
#       decoded = unicodedata.name(e).lower()
#       # decoded = _emoji(e)
#       my_dict[e] = GoogleTranslator(source='auto', target='ru').translate(decoded)
#    # else: 
#    except Exception as error:
#       # print(error)
#       print(e)
#    #    e = e.encode("unicode_escape")
#    #    decoded = unicodedata.name(e).lower()
#    #    my_dict[e] = GoogleTranslator(source='auto', target='ru').translate(decoded)

# emoji_df = pd.DataFrame(my_dict.items(), columns=['symbol', 'meaning'])


# def get_emotion_rating(e):
#    print(e)
#    dict1 = get_emoji_sentiment_rank(str(e))

#    if dict1 is not None:

#       keys = ['negative', 'neutral', 'positive']
#       if dict1 is not None:
#          dict2 = {x:dict1[x] for x in keys}
#          emotion = max(dict2, key=dict2.get)
#       else:
#          emotion = None
#    else:
#       emotion = None

#    return None

# emoji_df['emotion'] = emoji_df['symbol'].apply(get_emotion_rating)

# print(emoji_df.head())

# emoji_df.to_csv('emoji_dict.csv', index=False)

# e = '🤷'
# print(unicodedata.name(e))