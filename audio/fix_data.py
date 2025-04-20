import pandas as pd
import numpy as np

df = pd.read_csv('final_data_set_copy.csv')


def fix_year(year):
    if year < 10:
        return int('201' + str(year))
    elif 10 <= year < 25:
        return int('20' + str(year))
    elif 25 <= year < 100:
        return int('201' + str(year)[1])
    elif 1980 <= year <= 2024:
        return year
    else:
        return None


def replace(rep):
    if rep in ['-', '---', '-----', 0.0, '- мест']:
        return None
    else:
        return rep


df = df.drop(columns=['liters_per_km', 'places', 'trans'], axis=1)

df['year_of_creation'] = df['year_of_creation'].apply(fix_year)

for col in df.columns:
    df[col] = df[col].apply(replace)

for col in ['distance_travelled', 'price']:
    df[col] = df[col].str.replace(',', '').astype(float)

for col in ['distance_travelled', 'engine_capacity', 'year_of_creation', 'price', 'hp']:
    df[col] = df[col].astype(float)
    df[col] = df[col].fillna(df[col].median())

df['engine_type'] = df['engine_type'].astype('category')


df.to_csv('final_data_set.csv', index=False)

for col in df.columns:
    pct_missing = np.mean(df[col].isnull())
    print('{} - {}%'.format(col, round(pct_missing*100)))


