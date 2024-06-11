 # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import fasttext

import warnings
warnings.filterwarnings("ignore")

fasttext.FastText.eprint = lambda x: None

df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')
null_tags = df_headline[df_headline['Теги'].isnull()]
rndm_index = null_tags.sample().index[0]

s = null_tags.loc[rndm_index, 'Текст отзыва']
# s = 'Хороший банк. Пользуюсь им уже два с половиной года, все устраивает, обмана нет.'
# s = 'Давно пора сравнять с землей. Обслуживание отстой'
# s = 'КЭП Говорят 5 минут По факту 2-4 часа Вот такой вот фан!'
print(s)


# модель без класса обслуживание сотрудников 
# model = fasttext.load_model('models/model-10_june.bin')

model = fasttext.load_model(''models/model-10_june_3.bin')

# s = "при оформлении кредита пытались впарить дополнительную страховку то есть еще одну страховки итого договор не подписал на следующий день деньги за нее все равно списали по горячей линии подтвердили что услуга не обязательна но разбираются ситуации уже месяц даже если вы не подписали договор сотрудники ради премии все равно без вашего ведома впарят вам то что им нужно"

labels = model.predict(s, k = 1)
print(labels)
