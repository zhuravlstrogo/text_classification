import pandas as pd
# from navec import Navec

data = pd.read_excel('reviews-vtb-otdeleniya-20231004-112204.xlsx')
print('I read')
# print(data.head())

# 2023-10-04
print(max(data['Дата написания отзыва']))

# 2010-12-26 
print(min(data['Дата написания отзыва']))


def save_range(st, en, save_file_name):

    df = data[(data['Дата написания отзыва'] >= st)& (data['Дата написания отзыва'] <= en)]
    # df.to_csv(save_file_name, encoding='utf-8-sig')
    df.to_csv(save_file_name, encoding='utf-8-sig')
    print(f'shape: {df.shape}')

save_range(st='2010-10-04', en='2018-12-31', save_file_name='part_1.csv')

save_range(st='2019-01-01', en='2019-12-31', save_file_name='part_2019.csv')
save_range(st='2020-01-01', en='2020-12-31', save_file_name='part_2020.csv')

save_range(st='2021-01-01', en='2021-08-30', save_file_name='part_2021_1.csv')
save_range(st='2021-08-30', en='2021-12-31', save_file_name='part_2021_2.csv')

save_range(st='2022-01-01', en='2022-08-30', save_file_name='part_2022_1.csv')
save_range(st='2022-08-30', en='2022-12-31', save_file_name='part_2022_2.csv')

save_range(st='2023-01-01', en='2023-04-30', save_file_name='part_2023_0.csv')
save_range(st='2023-04-30', en='2022-12-31', save_file_name='part_2023_2.csv')



# path = 'navec_news_v1_1B_250K_300d_100q.tar'
# navec = Navec.load(path)

# print(navec['каша'])