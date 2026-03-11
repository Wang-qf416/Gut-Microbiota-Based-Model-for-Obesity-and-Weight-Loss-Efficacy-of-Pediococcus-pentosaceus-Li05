import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score
from model import train_random_forest


def recursive_feature_elimination(
    X,
    y,
    feature_names,
    cv_splits=5,
    random_state=42,
    initial_importance_file="initial_feature_importance.csv",
    best_model_file="best_model.joblib"
):
    """
    Iteratively remove least important features, record cross-validated AUC,
    and save the best model.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Labels.
    feature_names : list
        List of feature names.
    cv_splits : int
        Number of cross-validation folds.
    random_state : int
        Random seed.
    initial_importance_file : str
        Output path for initial feature importance CSV.
    best_model_file : str
        Output path for the best model (joblib format).

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: n_features, removed_feature, mean_auc, std_auc, remaining_features.
    """
    # Train initial model and compute feature importance
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
    importance_df.to_csv(initial_importance_file, index=False)

    sorted_features = importance_df["feature"].tolist()
    removal_order = sorted_features[::-1]  # remove least important first

    current_features = feature_names.copy()
    current_X = X.copy()

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state
    )

    results = []

    # Initial full-feature AUC
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

    # Iteratively remove features
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
            joblib.dump(best_model_saved, best_model_file)

        results.append({
            "n_features": len(current_features),
            "removed_feature": feature,
            "mean_auc": mean_auc,
            "std_auc": auc_scores.std(),
            "remaining_features": ",".join(current_features)
        })

    return pd.DataFrame(results)
