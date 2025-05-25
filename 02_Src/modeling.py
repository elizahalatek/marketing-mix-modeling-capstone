# src/modeling.py

import numpy as np
import pandas as pd
import statsmodels.api as sm
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from feature_engineering import apply_adstock


# Linear Regression with Statsmodels

def run_linear_model(df, features, target='Sales'):
    X = df[features].copy()
    X = X.astype(float)
    y = np.log1p(df[target])
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    return model, model.summary2().tables[1]


# Full Grid Search: Adstock + XGBoost

def full_grid_search_with_adstock(
    df,
    base_cols,
    static_features,
    decay_grid,
    xgb_param_grid,
    use_log=True,
    n_splits=3
):

    from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
    from xgboost import XGBRegressor
    from sklearn.metrics import r2_score
    import numpy as np
    import pandas as pd

    from feature_engineering import apply_adstock

    best_score = -np.inf
    best_result = None

    for decay in decay_grid:
        df_temp = df.copy()

        # Apply adstock
        for col in base_cols:
            df_temp[f'{col}_adstock'] = apply_adstock(df_temp[col], decay)
            if use_log:
                df_temp[f'{col}_adstock'] = np.log1p(df_temp[f'{col}_adstock'])

        # Choose feature set
        feature_suffix = '_adstock'
        features = [f"{col}{feature_suffix}" for col in base_cols] + static_features

        # Define target
        y = np.log1p(df_temp['Sales'])  # modeling log(Sales)

        # Define inputs
        X = df_temp[features]

        # Model & time series CV
        model = XGBRegressor(random_state=42)
        tscv = TimeSeriesSplit(n_splits=n_splits)

        grid = GridSearchCV(model, xgb_param_grid, cv=tscv, scoring='r2', n_jobs=-1)
        grid.fit(X, y)

        if grid.best_score_ > best_score:
            best_score = grid.best_score_
            best_result = {
                'decay': decay,
                'params': grid.best_params_,
                'score': grid.best_score_,
                'features': features,
                'model': grid.best_estimator_
            }

    log_note = "log(adstock)" if use_log else "raw adstock"
    print(f" Best decay: {best_result['decay']}")
    print(f" Best XGBoost Params: {best_result['params']}")
    print(f" Best CV R² using {log_note}: {best_result['score']:.4f}")

    return best_result



# Adstock Decay Tuning Grid Only

def test_adstock_grid(df, base_cols, decay_values, static_features=[]):
    results = []

    for decay in decay_values:
        temp_df = df.copy()

        for col in base_cols:
            temp_df[f'{col}_adstock'] = apply_adstock(temp_df[col], decay)

        model_features = [f'{col}_adstock' for col in base_cols] + static_features
        X = temp_df[model_features]
        y_log = np.log1p(temp_df['Sales'])

        model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
        model.fit(X, y_log)

        y_pred_log = model.predict(X)
        y_pred = np.expm1(y_pred_log)
        y_true = temp_df['Sales']

        results.append({
            'decay': decay,
            'R2': r2_score(np.log1p(y_true), y_pred_log),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'MAE': mean_absolute_error(y_true, y_pred)
        })

    return pd.DataFrame(results).sort_values(by='R2', ascending=False)


# XGBoost Model Runner (Simple)

def run_xgboost_model(df, features, target='Sales'):
    X = df[features].copy()
    y = np.log1p(df[target])

    model = XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1, random_state=42)
    model.fit(X, y)

    y_pred_log = model.predict(X)
    y_pred = np.expm1(y_pred_log)
    y_true = df[target]

    metrics = {
        'R2': r2_score(np.log1p(y_true), y_pred_log),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred)
    }

    feature_importance = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values(by='importance', ascending=False)

    return model, metrics, feature_importance
