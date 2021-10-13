# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os

os.chdir("C:/Users/Wojtek/Desktop/Uczelnia/Praca magisterska")

# %% Wczytanie zbioru danych
raw_data = pd.read_excel('./dane/DATA2017.xlsx')

# %% Utworzenie kopii i zmiana typu zmiennej "NIP" na tekstową
df = raw_data.copy()
df['NIP'] = df['NIP'].astype(str)

# %% Wczytanie zbioru danych z upadłosciami i zmiana typu danych na tekstowe
data_upadlosci = pd.read_excel('./dane/2018 upadlosci/upadlosci.xlsx')
data_upadlosci['NIP'] = data_upadlosci['NIP'].astype(str)

# %% Połączenie zbiorów danych na podstawie zmiennej "NIP"
df = df.merge(data_upadlosci, how = 'left')

# %% Dodanie w pustych miejscach "0" w zmiennej "upadlosc"
df['upadlosc'] = df['upadlosc'].fillna(0)

# %% usunięcie wielkich liter, polskich liter, spacji oraz innych znaków z nazw zmiennych
df.columns = [col.lower().replace(" ", "_").replace("(", "").replace(")", "")\
              for col in df.columns]

# %% Podstawowe informacje o zmiennych
df.info(verbose = True, show_counts = True)

# %% Usunięcie niepotrzebnych zmiennych
df = df.drop(df.columns[0:52], axis = 1)
df = df.drop(df.columns[-19:-1], axis = 1)
df.info(verbose = True, show_counts = True)

# %% Analiza braków danych w poszczególnych obserwacjach
df[(df.isnull().sum(axis = 1) / df.shape[1] > 0.3) & (df["upadlosc"] == 0)]

# %% Usunięcie obserwacji, w których występuje więcej niż 30% braków danych
df = df.drop(df[(df.isnull().sum(axis = 1) / df.shape[1] > 0.3) & (df["upadlosc"] == 0)].index,
             axis = 0).reset_index(drop = True)

# %% Analiza braków danych w poszczególnych zmiennych
braki_danych_zm = pd.DataFrame({"zmienne": df.columns[:-1],
                                "braki_pct": df.iloc[:,:-1].isna().sum() / np.shape(df)[0]})
braki_danych_zm.reset_index(drop = True, inplace = True)

braki_danych_zm[braki_danych_zm["braki_pct"] > 0.2]

# %% Usunięcie zmiennych, w których występuje więcej niż 20% braków danych
df.drop(df.columns[braki_danych_zm[braki_danych_zm["braki_pct"] > 0.2].index],
        axis = 1,
        inplace = True)

# %% Podstawowe informacje o zmiennych
df.info(verbose = True, show_counts = True)

# %% Podstawowe statystki zmiennych objasniających, zapisanie do pliku z wynikami
df.describe().T.to_excel('Wyniki.xlsx', sheet_name = 'Podstawowe statystyki',
                         startrow = 1, startcol = 1)
df.describe().T

# %% Podział zbioru danych wg zmiennej dotyczącej upadlosci
X0 = df[df["upadlosc"] == 0].reset_index(drop = True)
X1 = df[df["upadlosc"] == 1].reset_index(drop = True)

# %% Zastąpienie brakow danych medianą
for i in X0.iloc[:,:-1].columns:
    X0[i].fillna(X0[i].median(), inplace = True)

for i in X1.iloc[:,:-1].columns:
    X1[i].fillna(X1[i].median(), inplace = True)

# %% Złączenie danych
df = X0.append(X1).reset_index(drop = True)

# %% Zapisanie danych do pliku
df.to_excel('df_preprocessed.xlsx')

# %% Ponowny podział zbioru danych wg zmiennej dotyczącej upadlosci
X0 = df[df["upadlosc"] == 0].reset_index(drop = True)
X1 = df[df["upadlosc"] == 1].reset_index(drop = True)

# %% Ograniczenie obserwacji klasy "0"
X0 = X0.sample(int(9 * len(X1)), axis = 0).reset_index(drop = True)

# %% Dobranie danych do zbioru treningowego i walidacyjnego
X0_sample = X0.sample(int(round(0.6 * len(X0), 0)), axis = 0)
X1_sample = X1.sample(int(round(0.6 * len(X1), 0)), axis = 0)
X1_train_add = X1_sample.append([X1_sample] * 8, ignore_index = True).reset_index(drop = True)
X_train = pd.concat([X0_sample, X1_train_add], axis = 0).sample(frac = 1).reset_index(drop = True)

# %% Stworzenie zbioru walidacyjnego
X0_temp = X0.drop([i for i in X0_sample.index], axis = 0)
X1_temp = X1.drop([i for i in X1_sample.index], axis = 0)
X0_validation = X0_temp.sample(int(round(0.2 * len(X0), 0)), axis = 0)
X1_validation = X1_temp.sample(int(round(0.2 * len(X1), 0)), axis = 0)
X_validation = pd.concat([X0_validation, X1_validation], axis = 0).sample(frac = 1).reset_index(drop = True)

# %% Utworzenie z pozostałych danych zbioru testowego
X0_test = X0_temp.drop([i for i in X0_validation.index], axis = 0)
X1_test = X1_temp.drop([i for i in X1_validation.index], axis = 0)
X_test = pd.concat([X0_test, X1_test], axis = 0).sample(frac = 1).reset_index(drop = True)

# %% Oddzielenie zmiennej objasnianej od zbiorow danych
y_train = X_train.pop('upadlosc')
y_validation = X_validation.pop('upadlosc')
y_test = X_test.pop('upadlosc')

# %% Zapisanie zbiorów do plików
X_train.to_excel('x_train.xlsx')
X_validation.to_excel('x_validation.xlsx')
X_test.to_excel('x_test.xlsx')
y_train.to_excel('y_train.xlsx')
y_validation.to_excel('y_validation.xlsx')
y_test.to_excel('y_test.xlsx')