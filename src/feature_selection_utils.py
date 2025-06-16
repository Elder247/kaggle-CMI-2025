from typing import Literal
import pandas as pd
import matplotlib.pyplot as plt

import shap
import phik
from phik.report import plot_correlation_matrix
from sklearn.inspection import permutation_importance

from src.constants import RANDOM_STATE


def get_phik_corr(
    df: pd.DataFrame, 
    label_name: str, 
    mode: Literal['significance', 'matrix'] = 'significance'
) -> None:
    """
    Calculate the correlation matrix using the phik library.
    """
    if mode == 'significance':
        corr = df.significance_matrix() # type: ignore
        title = r"correlation $\phi_K$ (significance)"
    elif mode == 'matrix':
        corr = df.phik_matrix() # type: ignore
        title = r"correlation $\phi_K$ (matrix)"
    else:
        raise ValueError("Mode must be either 'significance' or 'matrix'.")

    corr = corr.fillna(0).round(2).sort_values(by=label_name)

    plot_correlation_matrix(
        corr.values,
        x_labels=corr.columns,
        y_labels=corr.index,
        vmin=0, vmax=1, color_map="Greens",
        title=title,
        fontsize_factor=0.8, figsize=(11, 6)
    )


def get_shap_catboost(model, val_pool):
    """
    Calculate SHAP values for a CatBoost model and plot the summary.
    """
    explainer = shap.TreeExplainer(model)
    
    shap_values = explainer.shap_values(val_pool).transpose(1, 0, 2)
    shap.summary_plot(list(shap_values[:, :-1, :-1]),)


def get_permutation_importance(model, X_val, y_val):
    """
    Calculate permutation importance for a model.
    """
    result = permutation_importance(
        model, 
        X_val, y_val, 
        n_repeats=10, 
        random_state=RANDOM_STATE
    )
    sorted_idx = result.importances_mean.argsort() # type: ignore
    
    plt.figure(figsize=(10, 6))
    plt.barh(
        X_val.columns[sorted_idx], 
        result.importances_mean[sorted_idx],  # type: ignore
        xerr=result.importances_std[sorted_idx] # type: ignore
    )
    plt.xlabel("Permutation Importance")
    plt.title("Permutation Importance of Features")
    plt.show()
