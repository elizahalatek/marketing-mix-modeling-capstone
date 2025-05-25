import pandas as pd

def calculate_roi(coef_df, df):
    roi_list = []

    CPM = {
    'Facebook_Impressions_adstock': 10,
    'Google_Impressions_adstock': 8,
    'Email_Impressions_adstock': 5,
    'Affiliate_Impressions_adstock': 6,
    'Paid_Views_adstock': 7,
    'Organic_Views_adstock': 0,
    'Overall_Views_adstock': 0
    }


    for _, row in coef_df.iterrows():
        feature = row['feature']
        coef = row['coefficient']

        if feature not in CPM:
            continue

        impressions = df[feature].sum()
        spend = (impressions / 1000) * CPM[feature]
        contribution = coef * impressions
        roi = (contribution - spend) / spend if spend > 0 else 0

        roi_list.append({
            'channel': feature,
            'roi': roi,
            'spend_estimate': spend,
            'contribution': contribution
        })

    return pd.DataFrame(roi_list)
