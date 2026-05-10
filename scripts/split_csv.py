import pandas as pd
# from navec import Navec

data = pd.read_csv('comm_3650.csv', sep=';')

# comm_3650
print('tags')
print(data['tags'].sample(5))

# st, en = '2024-03-01', '2024-03-02' # 2024-03-25
# data['published_at'] = pd.to_datetime(data['published_at'])

# print(data.info())
# sample = data[(data['published_at'] > st ) & (data['published_at'] <  en)]
# print('LEN ', len(sample))

# # 2024-06-25
print('max')
print(max(data['published_at'].fillna('1800-01-01')))

# 2017-04-17
print('min')
print(min(data['published_at'].fillna('2222-02-02')))

1/0
def save_range(st, en, save_file_name):

    df = data[(data['published_at'] >= st)& (data['published_at'] <= en)]
    # df.to_csv(save_file_name, encoding='utf-8-sig')

    if not df.empty:
        try:
            df.to_excel(save_file_name + '.xlsx')
            print(f'shape: {df.shape}')
            print(f'{save_file_name} saved')
        except:
            df.to_csv(save_file_name + '.csv') 
            print(f'shape: {df.shape}')
            print(f'{save_file_name} saved')

save_range(st='2017-01-01', en='2020-12-31', save_file_name='part_2017_2020')
# save_range(st='2018-01-01', en='2018-12-31', save_file_name='part_2018')
# save_range(st='2019-01-01', en='2019-12-31', save_file_name='part_2019')
# save_range(st='2019-06-01', en='2019-12-31', save_file_name='part_2')
# save_range(st='2020-01-01', en='2020-12-31', save_file_name='part_2020')
# save_range(st='2020-06-01', en='2020-12-31', save_file_name='part_4')

save_range(st='2021-01-01', en='2021-12-31', save_file_name='part_2021')
# save_range(st='2021-06-01', en='2021-12-31', save_file_name='part_6')
save_range(st='2022-01-01', en='2022-12-31', save_file_name='part_2022')
# save_range(st='2022-06-01', en='2022-12-31', save_file_name='part_8')
save_range(st='2023-01-01', en='2023-12-31', save_file_name='part_2023')
# save_range(st='2023-06-01', en='2023-12-31', save_file_name='part_10')
save_range(st='2024-01-01', en='2024-06-01', save_file_name='part_2024')