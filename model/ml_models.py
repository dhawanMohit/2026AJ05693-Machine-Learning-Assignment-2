"""
Classifier definitions and evaluation metrics module for Breast Cancer Wisconsin (Diagnostic) Classification.
Compatible with Python 3.10+, NumPy, SciPy, and Scikit-Learn / Streamlit environments.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.spatial.distance import cdist


class StandardScalerArtifact:
    """Z-score standardization preprocessor."""
    def __init__(self):
        self.mean_ = None
        self.scale_ = None
        self.feature_names_in_ = None

    def fit(self, X, feature_names=None):
        X_arr = np.asarray(X, dtype=float)
        self.mean_ = np.mean(X_arr, axis=0)
        self.scale_ = np.std(X_arr, axis=0)
        self.scale_[self.scale_ == 0] = 1.0
        if feature_names is not None:
            self.feature_names_in_ = list(feature_names)
        return self

    def transform(self, X):
        X_arr = np.asarray(X, dtype=float)
        return (X_arr - self.mean_) / self.scale_

    def fit_transform(self, X, feature_names=None):
        return self.fit(X, feature_names=feature_names).transform(X)


class LogisticRegressionModel:
    """L2-regularized Logistic Regression Classifier trained via L-BFGS-B."""
    def __init__(self, C=1.0, random_state=42):
        self.C = C
        self.random_state = random_state
        self.coef_ = None
        self.intercept_ = None
        self.classes_ = np.array([0, 1])

    @staticmethod
    def _sigmoid(z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        n_samples, n_features = X_arr.shape
        init_params = np.zeros(n_features + 1)

        def loss_and_grad(params):
            w = params[:-1]
            b = params[-1]
            z = X_arr @ w + b
            p = self._sigmoid(z)
            eps = 1e-12
            bce = -np.sum(y_arr * np.log(p + eps) + (1.0 - y_arr) * np.log(1.0 - p + eps))
            reg = 0.5 * (1.0 / self.C) * np.sum(w ** 2)
            loss = (bce + reg) / n_samples

            diff = (p - y_arr) / n_samples
            grad_w = (X_arr.T @ diff) + (1.0 / (self.C * n_samples)) * w
            grad_b = np.sum(diff)
            return loss, np.append(grad_w, grad_b)

        res = minimize(loss_and_grad, init_params, method="L-BFGS-B", jac=True)
        self.coef_ = res.x[:-1].reshape(1, -1)
        self.intercept_ = np.array([res.x[-1]])
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        z = X_arr @ self.coef_.ravel() + self.intercept_[0]
        p1 = self._sigmoid(z)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class DecisionTreeModel:
    """CART Decision Tree Classifier using Gini Impurity."""
    def __init__(self, max_depth=4, min_samples_split=6, random_state=42):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.tree_ = None
        self.classes_ = np.array([0, 1])
        self.feature_importances_ = None

    @staticmethod
    def _gini(y):
        if len(y) == 0:
            return 0.0
        p1 = np.mean(y == 1)
        p0 = 1.0 - p1
        return 1.0 - (p0 ** 2 + p1 ** 2)

    def _build_tree(self, X, y, depth=0):
        n_samples, n_features = X.shape
        n_pos = int(np.sum(y == 1))
        prob_pos = n_pos / n_samples if n_samples > 0 else 0.0
        pred_class = 1 if prob_pos >= 0.5 else 0

        if (
            depth >= self.max_depth
            or n_samples < self.min_samples_split
            or len(np.unique(y)) == 1
        ):
            return {"leaf": True, "prob": prob_pos, "pred": pred_class, "samples": n_samples}

        best_gain = -1.0
        best_feat = None
        best_thresh = None
        parent_gini = self._gini(y)

        for feat_idx in range(n_features):
            col = X[:, feat_idx]
            thresholds = np.percentile(np.unique(col), np.linspace(10, 90, 17))
            for thresh in np.unique(thresholds):
                left_mask = col <= thresh
                right_mask = ~left_mask
                n_left = np.sum(left_mask)
                n_right = np.sum(right_mask)
                if n_left < 2 or n_right < 2:
                    continue
                gain = parent_gini - (
                    (n_left / n_samples) * self._gini(y[left_mask])
                    + (n_right / n_samples) * self._gini(y[right_mask])
                )
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat_idx
                    best_thresh = float(thresh)

        if best_feat is None or best_gain <= 1e-6:
            return {"leaf": True, "prob": prob_pos, "pred": pred_class, "samples": n_samples}

        self.feature_importances_[best_feat] += best_gain * n_samples
        left_mask = X[:, best_feat] <= best_thresh
        return {
            "leaf": False,
            "feature": int(best_feat),
            "threshold": best_thresh,
            "prob": prob_pos,
            "left": self._build_tree(X[left_mask], y[left_mask], depth + 1),
            "right": self._build_tree(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        self.feature_importances_ = np.zeros(X_arr.shape[1], dtype=float)
        self.tree_ = self._build_tree(X_arr, y_arr, depth=0)
        total_imp = np.sum(self.feature_importances_)
        if total_imp > 0:
            self.feature_importances_ /= total_imp
        return self

    def _predict_row_prob(self, row, node):
        if node["leaf"]:
            return node["prob"]
        if row[node["feature"]] <= node["threshold"]:
            return self._predict_row_prob(row, node["left"])
        return self._predict_row_prob(row, node["right"])

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        p1 = np.array([self._predict_row_prob(row, self.tree_) for row in X_arr])
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class KNearestNeighborsModel:
    """Distance-weighted K-Nearest Neighbor Classifier."""
    def __init__(self, n_neighbors=7, weights="distance"):
        self.n_neighbors = n_neighbors
        self.weights = weights
        self.X_train_ = None
        self.y_train_ = None
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        self.X_train_ = np.asarray(X, dtype=float).copy()
        self.y_train_ = np.asarray(y, dtype=int).copy()
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        dists = cdist(X_arr, self.X_train_, metric="euclidean")
        knn_idx = np.argsort(dists, axis=1)[:, : self.n_neighbors]
        knn_labels = self.y_train_[knn_idx]
        if self.weights == "distance":
            knn_dists = np.take_along_axis(dists, knn_idx, axis=1)
            inv_d = 1.0 / (knn_dists + 1e-6)
            p1 = np.sum(inv_d * knn_labels, axis=1) / np.sum(inv_d, axis=1)
        else:
            p1 = np.mean(knn_labels, axis=1)
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class GaussianNaiveBayesModel:
    """Gaussian Naive Bayes Classifier."""
    def __init__(self, var_smoothing=1e-9):
        self.var_smoothing = var_smoothing
        self.classes_ = np.array([0, 1])
        self.class_prior_ = None
        self.theta_ = None
        self.var_ = None

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        eps = self.var_smoothing * np.var(X_arr, axis=0).max()
        self.theta_ = np.zeros((2, X_arr.shape[1]))
        self.var_ = np.zeros((2, X_arr.shape[1]))
        self.class_prior_ = np.zeros(2)
        for c in [0, 1]:
            X_c = X_arr[y_arr == c]
            self.theta_[c] = np.mean(X_c, axis=0)
            self.var_[c] = np.var(X_c, axis=0) + eps
            self.class_prior_[c] = len(X_c) / len(X_arr)
        return self

    def predict_proba(self, X):
        X_arr = np.asarray(X, dtype=float)
        log_probs = np.zeros((X_arr.shape[0], 2))
        for c in [0, 1]:
            log_prior = np.log(self.class_prior_[c])
            log_likelihood = -0.5 * np.sum(
                np.log(2.0 * np.pi * self.var_[c])
                + ((X_arr - self.theta_[c]) ** 2) / self.var_[c],
                axis=1,
            )
            log_probs[:, c] = log_prior + log_likelihood
        max_lp = np.max(log_probs, axis=1, keepdims=True)
        exp_lp = np.exp(log_probs - max_lp)
        return exp_lp / np.sum(exp_lp, axis=1, keepdims=True)

    def predict(self, X):
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)


class SupportVectorMachineModel:
    """Support Vector Machine Classifier with RBF Kernel Approximation and Platt Scaling."""
    def __init__(self, C=1.5, gamma=0.035, n_components=250, random_state=42):
        self.C = C
        self.gamma = gamma
        self.n_components = n_components
        self.random_state = random_state
        self.random_weights_ = None
        self.random_offset_ = None
        self.w_ = None
        self.b_ = None
        self.platt_a_ = 1.0
        self.platt_b_ = 0.0
        self.classes_ = np.array([0, 1])

    def _transform_rff(self, X):
        projection = X @ self.random_weights_ + self.random_offset_
        return np.sqrt(2.0 / self.n_components) * np.cos(projection)

    def fit(self, X, y):
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=int)
        y_signed = np.where(y_arr == 1, 1.0, -1.0)
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X_arr.shape
        self.random_weights_ = rng.normal(
            loc=0.0, scale=np.sqrt(2.0 * self.gamma), size=(n_features, self.n_components)
        )
        self.random_offset_ = rng.uniform(0.0, 2.0 * np.pi, size=self.n_components)
        Z = self._transform_rff(X_arr)

        init_params = np.zeros(self.n_components + 1)

        def squared_hinge_loss(params):
            w = params[:-1]
            b = params[-1]
            margins = 1.0 - y_signed * (Z @ w + b)
            active = np.maximum(0.0, margins)
            loss = 0.5 * np.sum(w ** 2) + self.C * np.sum(active ** 2)
            grad_w = w - 2.0 * self.C * (Z.T @ (active * y_signed))
            grad_b = -2.0 * self.C * np.sum(active * y_signed)
            return loss / n_samples, np.append(grad_w, grad_b) / n_samples

        res = minimize(squared_hinge_loss, init_params, method="L-BFGS-B", jac=True)
        self.w_ = res.x[:-1]
        self.b_ = float(res.x[-1])

        scores = Z @ self.w_ + self.b_
        # Fit Platt scaling logistic calibration
        def platt_loss(ab):
            a, b = ab
            z = a * scores + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
            eps = 1e-12
            return -np.mean(y_arr * np.log(p + eps) + (1.0 - y_arr) * np.log(1.0 - p + eps))

        res_platt = minimize(platt_loss, [1.0, 0.0], method="BFGS")
        self.platt_a_, self.platt_b_ = float(res_platt.x[0]), float(res_platt.x[1])
        return self

    def decision_function(self, X):
        X_arr = np.asarray(X, dtype=float)
        Z = self._transform_rff(X_arr)
        return Z @ self.w_ + self.b_

    def predict_proba(self, X):
        scores = self.decision_function(X)
        z = self.platt_a_ * scores + self.platt_b_
        p1 = 1.0 / (1.0 + np.exp(-np.clip(z, -250, 250)))
        return np.column_stack([1.0 - p1, p1])

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0.0).astype(int)


def compute_all_metrics(y_true, y_pred, y_prob):
    """
    Computes all 6 mandatory assignment metrics:
    1. Accuracy
    2. AUC Score (ROC-AUC via exact Wilcoxon-Mann-Whitney / trapezoidal integration)
    3. Precision
    4. Recall
    5. F1 Score
    6. Matthews Correlation Coefficient (MCC Score)
    Plus confusion matrix and per-class classification report.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)

    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    total = len(y_true)

    accuracy = (tp + tn) / total if total > 0 else 0.0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2.0 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    mcc_denom = np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)))
    mcc = ((tp * tn) - (fp * fn)) / mcc_denom if mcc_denom > 0 else 0.0

    # Exact ROC curve & AUC calculation
    desc_idx = np.argsort(-y_prob)
    y_true_sorted = y_true[desc_idx]
    y_prob_sorted = y_prob[desc_idx]

    distinct_value_indices = np.where(np.diff(y_prob_sorted))[0]
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    tps = np.cumsum(y_true_sorted)[threshold_idxs]
    fps = 1 + threshold_idxs - tps

    tps = np.r_[0, tps]
    fps = np.r_[0, fps]

    fpr = fps / fps[-1] if fps[-1] > 0 else np.zeros_like(fps, dtype=float)
    tpr = tps / tps[-1] if tps[-1] > 0 else np.zeros_like(tps, dtype=float)
    auc_score = float(np.trapezoid(tpr, fpr))

    # Class 0 metrics for full classification report
    prec_0 = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    rec_0 = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    f1_0 = (2.0 * prec_0 * rec_0) / (prec_0 + rec_0) if (prec_0 + rec_0) > 0 else 0.0
    supp_0 = int(np.sum(y_true == 0))
    supp_1 = int(np.sum(y_true == 1))

    report_rows = [
        {
            "Class": "0 (Benign)",
            "Precision": round(prec_0, 4),
            "Recall": round(rec_0, 4),
            "F1-Score": round(f1_0, 4),
            "Support": supp_0,
        },
        {
            "Class": "1 (Malignant)",
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1-Score": round(f1, 4),
            "Support": supp_1,
        },
        {
            "Class": "Macro Avg",
            "Precision": round((prec_0 + precision) / 2.0, 4),
            "Recall": round((rec_0 + recall) / 2.0, 4),
            "F1-Score": round((f1_0 + f1) / 2.0, 4),
            "Support": total,
        },
        {
            "Class": "Weighted Avg",
            "Precision": round((prec_0 * supp_0 + precision * supp_1) / total, 4),
            "Recall": round((rec_0 * supp_0 + recall * supp_1) / total, 4),
            "F1-Score": round((f1_0 * supp_0 + f1 * supp_1) / total, 4),
            "Support": total,
        },
    ]

    return {
        "Accuracy": round(float(accuracy), 4),
        "AUC": round(float(auc_score), 4),
        "Precision": round(float(precision), 4),
        "Recall": round(float(recall), 4),
        "F1": round(float(f1), 4),
        "MCC": round(float(mcc), 4),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "fpr": fpr.tolist(),
        "tpr": tpr.tolist(),
        "classification_report": report_rows,
    }
