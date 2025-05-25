import numpy as np
import pandas as pd

def simulate_budget_shift(
    df_sim,
    df_full,
    original_feature,
    target_feature,
    shift_percent,
    static_features,
    model,
    original_feature_list
):

    df_sim = df_sim.copy()

    # Total impressions to shift
    total_shift = df_sim[original_feature].sum() * shift_percent

    # Evenly redistribute
    df_sim[original_feature] -= total_shift / len(df_sim)
    df_sim[target_feature] += total_shift / len(df_sim)

    # Avoid negative inputs
    df_sim[original_feature] = np.maximum(0, df_sim[original_feature])
    df_sim[target_feature] = np.maximum(0, df_sim[target_feature])

    # Predict
    X_sim = df_sim[original_feature_list].copy()
    X_sim = X_sim.fillna(0).astype(np.float64)

    preds = model.predict(X_sim)
    preds = np.clip(preds, -10, 15)  # prevent overflow in expm1
    y_sim = np.expm1(preds)

    # Add back division means
    division_means = df_full.groupby('Division')['Sales'].mean()
    df_sim['division_mean'] = df_sim['Division'].map(division_means)
    y_sim += df_sim['division_mean']

    return y_sim.sum(), None
