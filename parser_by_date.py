import requests
import json
import pandas as pd
import numpy as np
import time
from datetime import datetime, timedelta, date
import re
import warnings
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', None)
pd.options.display.expand_frame_repr = False

from config import token

# кол-во дней, за к-ые перегружаем данные 
N = 1825
# 1825 - 5 лет


# cur_day = date.today()
# cur_day = cur_day - timedelta(days=N) # поставить нужное кол-во дней для пересчёта - 400
# cur_day = str(pd.to_datetime(cur_day))[:10]

# 37 min 2019-2020
# 31 min 2021
# 33 min 2022
# min 2023-24

# 1:51   2024-06-08 - 2024-06-08ч

# забирает данные, не включая указанные даты. по дням не влазиет всё
# from_date = '2024-06-03'
# to_date = '2024-06-05'





def get_reviews(from_date, to_date, true_date):
    headers = {
        'Accept': 'application/json'
        ,'Authorization': f'Bearer {token}'
        ,'Accept-Charset': 'UTF-8'
            }

    #meta
    url = f'https://api.pntr.io/v1/companies?with_ratings=true'
    # url_comments = f'https://api.pntr.io/v1/reviews?published_at_from={cur_day}&with_tags=true'
    url_comments = f'https://api.pntr.io/v1/reviews?published_at_from={from_date}&published_at_to={to_date}&with_tags=true'

    r = requests.get(url, headers=headers)
    try:
        r_dict = json.loads(r.text)

        r_meta = r_dict['meta']['total']
        r_int = r_meta // 50
        r_ost = r_meta % 50

        time.sleep(1)

        comments = requests.get(url_comments, headers=headers)
        # TODO: 
        try:
            comments_meta = json.loads(comments.text)['meta']['total']
            c_int = comments_meta // 50
            c_ost = comments_meta % 50
        
            time.sleep(1)

            url_providers = 'https://api.pntr.io/v1/providers'
            providers = requests.get(url_providers, headers=headers)
            providers = pd.DataFrame.from_dict(json.loads(providers.text)['items'])[['id', 'name_en']]

            keys_col = providers.query('name_en != "Reviews book"')['id'].to_list()
            values_col = providers.query('name_en != "Reviews book"')['name_en'].to_list()
            providers = {keys_col[i]: values_col[i] for i in range(len(keys_col))}

            time.sleep(1)
            print('start import reit')
            df = []
            limit = 50
            i = 0
            for offset in range(0, 50*r_int+1, 50):
                if len(df) == r_int:
                    url = f'https://api.pntr.io/v1/companies?with_ratings=true&limit={r_ost}&offset={50*r_int}'
                else:
                    url = f'https://api.pntr.io/v1/companies?with_ratings=true&limit={limit}&offset={offset}'
                r = requests.get(url, headers=headers)
                time.sleep(1)
                r_dict = json.loads(r.text)

                if i == 0:
                    df = pd.DataFrame.from_dict(r_dict['items'])
                    df = df[[
                        'full_address',
                        'lat',
                        'lng',
                        'number',
                        'ratings',
                        'uuid'
                    ]]
                else:
                    buf = pd.DataFrame.from_dict(r_dict['items'])
                    buf = buf[[
                        'full_address',
                        'lat',
                        'lng',
                        'number',
                        'ratings',
                        'uuid'
                    ]]
                    df = pd.concat([df, buf])
                i+= 1

            df['level'] = df['ratings'].apply(lambda x: len(x))

            df = pd.concat([df, df['ratings'].apply(pd.Series)], axis=1)
            level = df.level.max()

            if 1 <= level <=3:
                df = pd.concat([df, df[0].apply(pd.Series)], axis=1)
                df.drop(columns=['ratings', 0, 'date', 'ratings_count'], inplace=True)
                df.rename(columns={'avg_rating':'avg_rating_1', 'provider_id':'provider_id_1'}, inplace=True)

            if 2 <= level <=3:
                df = pd.concat([df, df[1].apply(pd.Series)], axis=1)
                df.drop(columns=[1, 'date', 'ratings_count'], inplace=True)
                df.rename(columns={'avg_rating':'avg_rating_2', 'provider_id':'provider_id_2'}, inplace=True)

            if level == 3:
                df = pd.concat([df, df[2].apply(pd.Series)], axis=1)
                df.drop(columns=[2, 0, 'date', 'ratings_count'], inplace=True)
                df.rename(columns={'avg_rating':'avg_rating_3', 'provider_id':'provider_id_3'}, inplace=True)

            if level == 1:
                df['provider_id_2'] = -1
                df['avg_rating_2'] = -1
                df['provider_id_3'] = -1
                df['avg_rating_3'] = -1

            if level == 2:
                df['provider_id_3'] = -1
                df['avg_rating_3'] = -1

            for i in providers.keys():
                df[providers[i]] = np.where(df['provider_id_1'] == i,
                                            df['avg_rating_1'],
                                            np.where(
                                                df['provider_id_2'] == i,
                                                df['avg_rating_2'],
                                                np.where(
                                                    df['provider_id_3'] == i,
                                                    df['avg_rating_3'],
                                                    np.nan
                                                        )
                                                    )
                                            )

            df.drop(columns=['avg_rating_1', 'provider_id_1', 'avg_rating_2', 'provider_id_2', 'avg_rating_3', 'provider_id_3', 'level'], inplace=True)
            # df.to_csv('reit.csv', sep=';', encoding = 'utf-8')

            time.sleep(1)

            df = []
            limit = 50
            i = 0
            for offset in range(0, 50*c_int+1, 50):
                if len(df) == c_int:
                    # url_comments = f'https://api.pntr.io/v1/reviews?limit={c_ost}&offset={50 * с_int}&published_at_from={cur_day}&with_tags=true'
                    url_comments = f'https://api.pntr.io/v1/reviews?published_at_from={from_date}&published_at_to={to_date}&with_tags=true'
                else:
                    # url_comments = f'https://api.pntr.io/v1/reviews?limit={limit}&offset={offset}&published_at_from={cur_day}&with_tags=true'
                    url_comments = f'https://api.pntr.io/v1/reviews?published_at_from={from_date}&published_at_to={to_date}&with_tags=true'
                r = requests.get(url_comments, headers=headers)
                time.sleep(1)
                c_dict = json.loads(r.text)

                if i == 0:
                    comments = pd.DataFrame.from_dict(c_dict['items'])
                    comments = comments[[
                        'author',
                        'company_uuid',
                        'id',
                        'provider_id',
                        'published_at',
                        'rating',
                        'reply',
                        'text',
                        'text_tonality',
                        'tags'
                    ]]
                else:
                    buf = pd.DataFrame.from_dict(c_dict['items'])
                    buf = buf[[
                        'author',
                        'company_uuid',
                        'id',
                        'provider_id',
                        'published_at',
                        'rating',
                        'reply',
                        'text',
                        'text_tonality',
                        'tags'
                    ]]
                    print(len(comments))
                    comments = pd.concat([comments, buf])
                i+= 1


            comments.rename(columns={'published_at':'published_at_author', 'text':'text_author'}, inplace=True)



            comments = pd.concat([comments, comments['reply'].apply(pd.Series)], axis=1)
            comments.drop(columns=['reply'], inplace=True)
            comments['provider_id'] = comments['provider_id'].apply(lambda x: providers[x])


            # comments['handled_tags'] = comments['tags'].apply(extract_tag)

            # print('tags')
            # print(comments['tags'].value_counts().keys())

            print('min date in reviews ', min(comments['published_at_author']))
            print('max date in reviews ', max(comments['published_at_author']))

            print(comments.sample())

            comments.to_csv(f'reviews/comm_{true_date}.csv', sep=';', encoding = 'utf-8')   
        except Exception as e:
            print(f'request in comments: {comments.text}')
            print(f'error in comments: {e}')
    except Exception as e:
        print(f'request in r_dict: {r_dict}')
        print(f'error in r_dict: {e}')

            





if __name__ == "__main__":
    start = datetime.now()
    print(f'start: {start}')
    # TODO: '2021-12-31' '2022-08-01' 
    # TODO: 2015 год есть с октября  
    date_range = [d for d in pd.date_range(start='2015-10-01', end='2015-10-03', freq='d')]

    for from_date in date_range:
        from_date = pd.to_datetime(from_date, format="%Y-%m-%d")
        true_date = from_date + timedelta(days=1)
        end_date = from_date + timedelta(days=2)
        

        from_date = from_date.strftime("%Y-%m-%d")
        true_date = true_date.strftime("%Y-%m-%d")
        to_date = end_date.strftime("%Y-%m-%d")


        print(f'get reviews for {true_date}')

        get_reviews(from_date, to_date, true_date)
        print(f'Finish: {datetime.now() - start} seconds')

