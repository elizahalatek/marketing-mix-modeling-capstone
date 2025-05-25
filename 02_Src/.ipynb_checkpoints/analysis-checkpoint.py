import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_feature_importance(importance_df, top_n=10):
    top_df = importance_df.head(top_n)
    plt.figure(figsize=(10, 6))
    sns.barplot(x='importance', y='feature', data=top_df)
    plt.title("Top Media Drivers (XGBoost Feature Importance)")
    plt.tight_layout()
    plt.show()

def plot_decay_tuning(adstock_results):
    plt.figure(figsize=(8, 5))
    plt.plot(adstock_results['decay'], adstock_results['R2'], marker='o')
    plt.title("R² vs Adstock Decay")
    plt.xlabel("Adstock Decay Factor")
    plt.ylabel("R²")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
