import pandas as pd
import numpy as np
from datetime import datetime
import time



big_df = pd.read_csv('comm_3650.csv' ,sep=';')


# TODO: код из батчей 
start = 0
end = big_df.shape[0]
step = 50000

counter = 0
for i in range(start, end, step):
    fn_i = f'geoservices_part_{counter}'
    x = i
    chunk = big_df.iloc[x:x+step,:]
    print(f"chunk from {x} to {x+step}")
    try:
        chunk.to_excel(fn_i +  '.xlsx', index=False)
    except:
        chunk.to_csv(fn_i + '.csv', index=False)
    counter += 1

print(f'{counter} files saved')