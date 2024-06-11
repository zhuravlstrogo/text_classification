 # -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import fasttext

import warnings
warnings.filterwarnings("ignore")

fasttext.FastText.eprint = lambda x: None

# df_headline = pd.read_csv('Tanya_file.csv',encoding='utf-8-sig')
# null_tags = df_headline[df_headline['Теги'].isnull()]
# rndm_index = null_tags.sample().index[0]
# s = null_tags.loc[rndm_index, 'Текст отзыва']


# модель без сборного класса "обслуживание сотрудников" 
model = fasttext.load_model('models/tonality_model_13_june.bin') # рабочая


s = "Отделение работает до 19:00. Если вы приедете даже к 18:00, то вам каждые десять минут будут напоминать, что «в 19:00 отделение закрывается, всех принять мы не сможем».И вопрос: как тогда в банк приходить??На 10 человек работает 2 оператора."

labels = model.predict(s, k = 1)
print(labels)
