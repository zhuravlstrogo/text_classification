import pandas as pd
import numpy as np
from datetime import datetime
import unicodedata
# from utils import rename_tags, join_words, extract_tag, process_classes
import emoji
import emojis
import math
import time
import ast
import re
from deep_translator import GoogleTranslator
from emosent import get_emoji_sentiment_rank



df = pd.read_csv('comm_3650.csv' ,sep=';')

print(max(df['published_at']))

sample1 = df[(df['published_at'] >= '2024-06-01') & (df['published_at'] <= '2024-06-30')]
print(len(sample1))

sample2 = df[(df['published_at'] >= '2024-05-01') & (df['published_at'] <= '2024-05-31')]
print(len(sample2))