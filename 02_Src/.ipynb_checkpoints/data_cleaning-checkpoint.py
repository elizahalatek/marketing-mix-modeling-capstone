# src/data_cleaning.py

import pandas as pd

import pandas as pd

def clean_data(df):
    df['Calendar_Week'] = pd.to_datetime(df['Calendar_Week'], format='%m/%d/%Y')
    df = df.dropna()
    df = df[df['Sales'] < df['Sales'].quantile(0.99)]
    return df

