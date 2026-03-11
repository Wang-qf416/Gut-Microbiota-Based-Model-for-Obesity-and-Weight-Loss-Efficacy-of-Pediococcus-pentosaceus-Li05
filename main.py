import warnings
warnings.filterwarnings("ignore")

import argparse
import pandas as pd
from data_loader import load_data
from rfe import recursive_feature_elimination


def main():
    parser = argparse.ArgumentParser(description="Recursive feature elimination with random forest classifier")
    # Input file arguments
    parser.add_argument("--species-file", default="species.xlsx",
                        help="Excel file with species abundance (default: species.xlsx)")
    parser.add_argument("--selected-species-file", default="c1.txt",
                        help="Text file with selected species list (default: c1.txt)")
    parser.add_argument("--group-a-file", default="c_sample_tax.csv",
                        help="CSV file with group A sample names (default: c_sample_tax.csv)")
    parser.add_argument("--group-b-file", default="o_sample_tax.csv",
                        help="CSV file with group B sample names (default: o_sample_tax.csv)")
    # Output file arguments
    parser.add_argument("--initial-importance", default="initial_feature_importance.csv",
                        help="Output file for initial feature importance (default: initial_feature_importance.csv)")
    parser.add_argument("--best-model", default="best_model.joblib",
                        help="Output file for best model (default: best_model.joblib)")
    parser.add_argument("--rfe-results", default="rfe_results.csv",
                        help="Output CSV for RFE results (default: rfe_results.csv)")
    parser.add_argument("--best-features", default="best_features.csv",
                        help="Output CSV for best feature set (default: best_features.csv)")
    # Other arguments
    parser.add_argument("--cv-splits", type=int, default=5,
                        help="Number of cross-validation folds (default: 5)")
    parser.add_argument("--random-state", type=int, default=42,
                        help="Random seed (default: 42)")

    args = parser.parse_args()

    # Load data
    X, y, feature_names = load_data(
        species_file=args.species_file,
        selected_species_file=args.selected_species_file,
        group_a_file=args.group_a_file,
        group_b_file=args.group_b_file
    )

    # Run recursive feature elimination
    results_df = recursive_feature_elimination(
        X, y, feature_names,
        cv_splits=args.cv_splits,
        random_state=args.random_state,
        initial_importance_file=args.initial_importance,
        best_model_file=args.best_model
    )

    # Save RFE process results
    results_df.to_csv(args.rfe_results, index=False)

    # Extract best feature set
    best_idx = results_df["mean_auc"].idxmax()
    best_features = results_df.loc[best_idx, "remaining_features"].split(",")
    pd.DataFrame({"feature": best_features}).to_csv(args.best_features, index=False)


if __name__ == "__main__":
    main()
