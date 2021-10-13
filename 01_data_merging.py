# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/Wojtek/Desktop/Uczelnia/Praca magisterska")

# %% Wczytanie pierwszej czesci danych
raw_data = pd.read_excel('./dane/2017/DATA1.xlsx', header = 7, index_col = 0,
                         nrows = 1000)

# %% Skopiowanie danych do nowej zmiennej
df_2017 = raw_data.copy()

# %% Wczytanie kolejnych danych
for i in range(2, 19):
    df_temp = pd.read_excel('./dane/2017/DATA' + str(i) + '.xlsx', header = 7,
                            index_col = 0, nrows = 1000)
    df_2017 = df_2017.append(df_temp)
df_2017 = df_2017[:-8]

# %% Zapisanie do nowego pliku
df_2017.to_excel('./dane/DATA2017.xlsx')