import numpy as np
import pandas as pd

def add_seasonal_features(df):
    df['Calendar_Week'] = pd.to_datetime(df['Calendar_Week'])
    df['month'] = df['Calendar_Week'].dt.month
    df['weekofyear'] = df['Calendar_Week'].dt.isocalendar().week
    df['year'] = df['Calendar_Week'].dt.year
    df['week_sin'] = np.sin(2 * np.pi * df['weekofyear'] / 52)
    df['week_cos'] = np.cos(2 * np.pi * df['weekofyear'] / 52)
    df['is_black_friday'] = df['weekofyear'].isin([47, 48]).astype(int)
    df['is_christmas'] = ((df['Calendar_Week'].dt.month == 12) & (df['Calendar_Week'].dt.day >= 20)).astype(int)

    easter_dates = {
        2018: pd.Timestamp('2018-04-01'),
        2019: pd.Timestamp('2019-04-21'),
        2020: pd.Timestamp('2020-04-12'),
    }

    def is_easter(row):
        return int(row['year'] in easter_dates and abs((row['Calendar_Week'] - easter_dates[row['year']]).days) <= 7)

    df['is_easter'] = df.apply(is_easter, axis=1)
    df['is_summer_sale'] = df['month'].isin([6, 7, 8]).astype(int)
    return df

def apply_adstock(series, decay=0.5):
    adstocked = []
    accumulated = 0
    for x in series:
        accumulated = x + decay * accumulated
        adstocked.append(accumulated)
    return adstocked

def add_division_dummies(df):
    if 'Division' in df.columns:
        division_dummies = pd.get_dummies(df['Division'], prefix='div', drop_first=True)
        return pd.concat([df, division_dummies], axis=1)
    return df
