import warnings
warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score


# ==========================================================
# Data Loading
# ==========================================================

def load_data(
    species_file="species.xlsx",
    selected_species_file="c1.txt",
    group_a_file="c_sample_tax.csv",
    group_b_file="o_sample_tax.csv"
):
    """
    Load feature matrix and labels.

    Returns
    -------
    X : np.ndarray
        Feature matrix (samples x features)
    y : np.ndarray
        Binary labels
    feature_names : list
        Feature names
    """

    species_df = pd.read_excel(species_file, index_col=0)

    selected_species_df = pd.read_csv(selected_species_file, sep="\t")
    selected_species = selected_species_df["species"].tolist()

    group_a_df = pd.read_csv(group_a_file, header=None)
    group_b_df = pd.read_csv(group_b_file, header=None)

    group_a_samples = group_a_df.iloc[1].dropna().tolist()
    group_b_samples = group_b_df.iloc[1].dropna().tolist()

    filtered_species = species_df.loc[
        species_df.index.isin(selected_species)
    ]

    X_df = filtered_species.T

    all_samples = group_a_samples + group_b_samples
    labels = [0] * len(group_a_samples) + [1] * len(group_b_samples)

    existing_samples = [s for s in all_samples if s in X_df.index]
    existing_labels = [
        labels[all_samples.index(s)] for s in existing_samples
    ]

    X = X_df.loc[existing_samples].values
    y = np.array(existing_labels)
    feature_names = X_df.columns.tolist()

    return X, y, feature_names


# ==========================================================
# Model Training
# ==========================================================

def train_random_forest(
    X,
    y,
    param_grid=None,
    cv_splits=5,
    random_state=42
):
    """
    Perform GridSearchCV for RandomForest.
    """

    if param_grid is None:
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [None, 10, 20, 30],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf": [1, 2, 4],
            "max_features": ["sqrt", "log2", 0.5],
            "min_weight_fraction_leaf": [0.0, 0.1],
            "bootstrap": [True, False],
        }

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    model = RandomForestClassifier(
        random_state=random_state,
        n_jobs=-1
    )

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring="roc_auc",
        n_jobs=-1
    )

    grid_search.fit(X, y)

    return (
        grid_search.best_estimator_,
        grid_search.best_params_,
        grid_search.best_score_,
    )


# ==========================================================
# Recursive Feature Elimination
# ==========================================================

def recursive_feature_elimination(
    X,
    y,
    feature_names,
    cv_splits=5,
    random_state=42
):
    """
    Iteratively remove least important features
    and record cross-validated AUC.
    """

    best_model, _, _ = train_random_forest(
        X, y,
        cv_splits=cv_splits,
        random_state=random_state
    )

    best_model.fit(X, y)

    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": best_model.feature_importances_,
    }).sort_values("importance", ascending=False)

    importance_df.to_csv("initial_feature_importance.csv", index=False)

    sorted_features = importance_df["feature"].tolist()
    removal_order = sorted_features[::-1]

    current_features = feature_names.copy()
    current_X = X.copy()

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    auc_scores = cross_val_score(
        best_model,
        current_X,
        y,
        cv=cv,
        scoring="roc_auc"
    )

    results.append({
        "n_features": len(current_features),
        "removed_feature": None,
        "mean_auc": auc_scores.mean(),
        "std_auc": auc_scores.std(),
        "remaining_features": ",".join(current_features)
    })

    best_auc = -1
    best_model_saved = None

    for feature in removal_order:

        if feature not in current_features:
            continue

        idx = current_features.index(feature)
        current_features.remove(feature)
        current_X = np.delete(current_X, idx, axis=1)

        if len(current_features) == 0:
            break

        model, _, _ = train_random_forest(
            current_X,
            y,
            cv_splits=cv_splits,
            random_state=random_state
        )

        auc_scores = cross_val_score(
            model,
            current_X,
            y,
            cv=cv,
            scoring="roc_auc"
        )

        mean_auc = auc_scores.mean()

        if mean_auc > best_auc:
            best_auc = mean_auc
            best_model_saved = model
            joblib.dump(best_model_saved, "best_model.joblib")

        results.append({
            "n_features": len(current_features),
            "removed_feature": feature,
            "mean_auc": mean_auc,
            "std_auc": auc_scores.std(),
            "remaining_features": ",".join(current_features)
        })

    return pd.DataFrame(results)


# ==========================================================
# Main
# ==========================================================

def main():

    X, y, feature_names = load_data()

    results_df = recursive_feature_elimination(
        X, y, feature_names
    )

    results_df.to_csv("rfe_results.csv", index=False)

    best_idx = results_df["mean_auc"].idxmax()
    best_features = results_df.loc[
        best_idx, "remaining_features"
    ].split(",")

    pd.DataFrame({
        "feature": best_features
    }).to_csv("best_features.csv", index=False)


if __name__ == "__main__":
    main()