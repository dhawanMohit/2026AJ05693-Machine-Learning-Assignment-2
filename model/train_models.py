"""
Training and Evaluation Pipeline for BITS Pilani M.Tech (AIML) - Machine Learning Assignment 2.
Trains all 5 required classification models on the Breast Cancer Wisconsin (Diagnostic) dataset,
saves the preprocessing scaler and trained models (.pkl) inside model/, and computes all 6 evaluation metrics.
"""
import os
import sys
import json
import pickle
import numpy as np
import pandas as pd

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

from model.ml_models import (
    StandardScalerArtifact,
    LogisticRegressionModel,
    DecisionTreeModel,
    KNearestNeighborsModel,
    GaussianNaiveBayesModel,
    SupportVectorMachineModel,
    compute_all_metrics,
)


def main():
    full_csv_path = os.path.join(PROJECT_DIR, "breast_cancer_wisconsin.csv")
    test_csv_path = os.path.join(PROJECT_DIR, "test_data.csv")

    df_full = pd.read_csv(full_csv_path)
    df_test = pd.read_csv(test_csv_path)

    # Identify training rows (complement of held-out test set)
    # Using row hash matching or reload split deterministically
    feature_cols = [c for c in df_full.columns if c != "target"]

    # Deterministic 80/20 stratified split (seed=42) matching generate_data.py
    y_all = df_full["target"].values
    idx_0 = np.where(y_all == 0)[0]
    idx_1 = np.where(y_all == 1)[0]
    np.random.seed(42)
    np.random.shuffle(idx_0)
    np.random.shuffle(idx_1)

    n_test_0 = int(round(0.20 * len(idx_0)))
    n_test_1 = 114 - n_test_0

    test_idx = np.concatenate([idx_0[:n_test_0], idx_1[:n_test_1]])
    train_idx = np.concatenate([idx_0[n_test_0:], idx_1[n_test_1:]])
    np.random.shuffle(test_idx)
    np.random.shuffle(train_idx)

    df_train = df_full.iloc[train_idx].reset_index(drop=True)
    df_test = df_full.iloc[test_idx].reset_index(drop=True)
    df_test.to_csv(test_csv_path, index=False)

    X_train = df_train[feature_cols].values
    y_train = df_train["target"].values
    X_test = df_test[feature_cols].values
    y_test = df_test["target"].values

    # Fit and save StandardScaler preprocessing artifact
    scaler = StandardScalerArtifact()
    X_train_scaled = scaler.fit_transform(X_train, feature_names=feature_cols)
    X_test_scaled = scaler.transform(X_test)

    scaler_path = os.path.join(CURRENT_DIR, "scaler.pkl")
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)

    models_config = [
        ("Logistic Regression", "logistic_regression.pkl", LogisticRegressionModel(C=1.0, random_state=42)),
        ("Decision Tree", "decision_tree.pkl", DecisionTreeModel(max_depth=4, min_samples_split=6, random_state=42)),
        ("KNN", "knn.pkl", KNearestNeighborsModel(n_neighbors=7, weights="distance")),
        ("Naive Bayes", "naive_bayes.pkl", GaussianNaiveBayesModel(var_smoothing=1e-9)),
        ("Support Vector Machine", "svm.pkl", SupportVectorMachineModel(C=1.5, gamma=0.035, n_components=250, random_state=42)),
    ]

    summary_results = {}
    print(f"{'ML Model Name':<25} | {'Accuracy':<8} | {'AUC':<8} | {'Precision':<9} | {'Recall':<8} | {'F1':<8} | {'MCC':<8}")
    print("-" * 88)

    for model_name, filename, clf in models_config:
        clf.fit(X_train_scaled, y_train)
        model_path = os.path.join(CURRENT_DIR, filename)
        with open(model_path, "wb") as f:
            pickle.dump(clf, f)

        y_pred = clf.predict(X_test_scaled)
        y_prob = clf.predict_proba(X_test_scaled)[:, 1]
        metrics = compute_all_metrics(y_test, y_pred, y_prob)
        summary_results[model_name] = metrics

        print(
            f"{model_name:<25} | "
            f"{metrics['Accuracy']:<8.4f} | "
            f"{metrics['AUC']:<8.4f} | "
            f"{metrics['Precision']:<9.4f} | "
            f"{metrics['Recall']:<8.4f} | "
            f"{metrics['F1']:<8.4f} | "
            f"{metrics['MCC']:<8.4f}"
        )

    metrics_json_path = os.path.join(CURRENT_DIR, "evaluation_summary.json")
    with open(metrics_json_path, "w") as f:
        json.dump(summary_results, f, indent=2)

    print("\nSaved all 5 trained models, scaler.pkl, and evaluation_summary.json to:", CURRENT_DIR)


if __name__ == "__main__":
    main()
