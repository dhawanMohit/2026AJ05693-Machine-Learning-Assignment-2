# Breast Cancer Wisconsin (Diagnostic) — End-to-End Machine Learning Classification & Streamlit Deployment

**BITS Pilani Work Integrated Learning Programmes (WILP)**  
**Course:** M.Tech (AIML) — Machine Learning  
**Assignment:** Assignment - 2 (End-to-End ML Modeling, Evaluation, UI Design & Cloud Deployment)

## a. Problem Statement

Early and accurate differentiation between malignant (cancerous) and benign (non-cancerous) breast masses is critical in clinical oncology to guide timely intervention and avoid unnecessary invasive surgical biopsies. Using digitized morphological measurements extracted from fine needle aspirate (FNA) biopsy slides in the [Breast Cancer Wisconsin (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic), the objective of this project is to build, evaluate, and deploy an end-to-end supervised classification pipeline comparing five distinct machine learning classifiers:

1. **Logistic Regression**
2. **Decision Tree Classifier**
3. **K-Nearest Neighbor (KNN) Classifier**
4. **Naive Bayes Classifier (Gaussian Naive Bayes)**
5. **Support Vector Machine (SVM)**

Each classifier is trained on the standardized training partition, serialized as a pre-trained artifact inside the `model/` directory, and evaluated on the held-out test dataset (`test_data.csv`) across six core classification metrics: **Accuracy**, **AUC Score**, **Precision**, **Recall**, **F1 Score**, and **Matthews Correlation Coefficient (MCC Score)**. Finally, an interactive **Streamlit** web application (`app.py`) is developed and deployed to allow real-time test CSV uploads, model selection, metric inspection, and confusion matrix / ROC curve visualization without retraining on the uploaded test set.

## b. Dataset Description

- **Dataset Name:** [Breast Cancer Wisconsin (Diagnostic) Dataset (WDBC)](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) (also hosted on [Kaggle UCI Breast Cancer Wisconsin Data](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data) and documented in [Scikit-Learn `load_breast_cancer`](https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_breast_cancer.html))
- **Source Creators:** Dr. William H. Wolberg, W. Nick Street, and Olvi L. Mangasarian (University of Wisconsin, Madison; DOI: `10.24432/C5DW2B`)
- **Total Instances:** `569` patient biopsy records (exceeds the assignment minimum requirement of `500` instances)
- **Total Features:** `30` continuous, real-valued nuclear morphology features (exceeds the assignment minimum requirement of `12` features)
- **Target Variable (`target`):** Binary classification label:
  - `0` — **Benign** (`357` instances, `62.74%`)
  - `1` — **Malignant** (`212` instances, `37.26%`)
- **Missing Values:** None (`0` missing entries across all `30` predictors)
- **Train / Test Split:** Stratified `80% / 20%` split (`455` training instances and `114` held-out test instances saved in `test_data.csv` with `71` Benign and `43` Malignant records)
- **Preprocessing Applied:** Z-score standardization (`StandardScalerArtifact` saved to `model/scaler.pkl`) fitted strictly on the `455` training records and applied to `test_data.csv` during inference.

### Summary of the 30 Predictive Features

Ten core cytological characteristics of cell nuclei are measured from digitized FNA biopsy images, and three statistical descriptors (**Mean**, **Standard Error (`_se`)**, and **Worst / Largest value (`_worst`)**) are computed for each characteristic (`10 × 3 = 30` features):

| Feature Group | Mean Feature | Standard Error (`_se`) | Worst / Peak (`_worst`) | Physical / Cytological Description |
| :--- | :--- | :--- | :--- | :--- |
| **1. Radius** | `radius_mean` | `radius_se` | `radius_worst` | Mean of distances from nucleus center to points on the perimeter |
| **2. Texture** | `texture_mean` | `texture_se` | `texture_worst` | Standard deviation of gray-scale pixel intensity values |
| **3. Perimeter** | `perimeter_mean` | `perimeter_se` | `perimeter_worst` | Total boundary length of the cell nucleus contour |
| **4. Area** | `area_mean` | `area_se` | `area_worst` | Cross-sectional area of the cell nucleus |
| **5. Smoothness** | `smoothness_mean` | `smoothness_se` | `smoothness_worst` | Local variation in radius lengths along the nuclear boundary |
| **6. Compactness** | `compactness_mean` | `compactness_se` | `compactness_worst` | Perimeter squared divided by area minus one ($\text{perimeter}^2 / \text{area} - 1.0$) |
| **7. Concavity** | `concavity_mean` | `concavity_se` | `concavity_worst` | Severity of concave indentations along the nuclear contour |
| **8. Concave Points** | `concave_points_mean` | `concave_points_se` | `concave_points_worst` | Number of distinct concave portions on the nuclear contour |
| **9. Symmetry** | `symmetry_mean` | `symmetry_se` | `symmetry_worst` | Bilateral symmetry score of the cell nucleus |
| **10. Fractal Dimension** | `fractal_dimension_mean` | `fractal_dimension_se` | `fractal_dimension_worst` | Boundary complexity ("coastline approximation" $- 1.0$) |

## c. GitHub Repository & Live Streamlit App Links

- **GitHub Repository Link:** `https://github.com/dhawanMohit/2026AJ05693-Machine-Learning-Assignment-2` *(Update with your personal GitHub URL after pushing this repository)*
- **Live Streamlit App Link:** `https://2026aj05693-machine-learning-assignment-2.streamlit.app` *(Update with your deployed Streamlit Community Cloud URL)*

### Repository Folder Structure

```text
wdbc_ml_assignment2/
│-- app.py                          # Interactive Streamlit web application
│-- requirements.txt                # Dependencies for Streamlit Community Cloud deployment
│-- README.md                       # Complete assignment documentation and evaluation tables
│-- test_data.csv                   # Held-out test dataset (114 instances, 30 features + target)
│-- breast_cancer_wisconsin.csv     # Full dataset (569 instances, 30 features + target)
│-- model/                          # Saved trained models, preprocessing artifact & training source code
    │-- __init__.py
    │-- ml_models.py                # Classifier classes, StandardScalerArtifact & metric calculations
    │-- train_models.py             # End-to-end training & evaluation script
    │-- scaler.pkl                  # Saved Z-score feature standardization artifact
    │-- logistic_regression.pkl     # Saved Logistic Regression classifier
    │-- decision_tree.pkl           # Saved Decision Tree classifier
    │-- knn.pkl                     # Saved K-Nearest Neighbor classifier
    │-- naive_bayes.pkl             # Saved Gaussian Naive Bayes classifier
    │-- svm.pkl                     # Saved Support Vector Machine classifier
    │-- evaluation_summary.json     # Exported evaluation metrics and confusion matrices
```

## d. Models Used & Evaluation Metrics Comparison Table

All five classification models were trained on the exact same `455`-instance training partition and evaluated on the held-out `114`-instance test dataset (`test_data.csv`).

### Comparison Table of Evaluation Metrics Across All 5 Models

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9123 | 0.9787 | 0.9024 | 0.8605 | 0.8810 | 0.8121 |
| **Decision Tree** | 0.9211 | 0.9302 | 0.8864 | 0.9070 | 0.8966 | 0.8329 |
| **KNN** | 0.9474 | 0.9574 | 0.9744 | 0.8837 | 0.9268 | 0.8885 |
| **Naive Bayes** | 0.9474 | 0.9864 | 0.9512 | 0.9070 | 0.9286 | 0.8876 |
| **Support Vector Machine** | 0.9298 | 0.9722 | 0.9268 | 0.8837 | 0.9048 | 0.8499 |

### Observations on the Performance of Each Model

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Achieved a strong ROC-AUC of **0.9787** and Precision of **0.9024**, confirming that the log-odds of malignancy are well-ordered along a linear hyperplane. However, because several nuclear features (`radius`, `perimeter`, `area`, `concavity`) exhibit strong multicollinearity and slight non-linear boundary curvature near borderline tumors, its hard-threshold Accuracy (**0.9123**) and Recall (**0.8605**) trailed the non-linear and probabilistic classifiers. |
| **Decision Tree** | Tied for the highest **Recall (0.9070)** across all models by capturing hierarchical axis-aligned rules on `concave_points_worst`, `radius_worst`, and `area_worst`, improving Accuracy to **0.9211** and MCC to **0.8329**. Because Decision Trees partition continuous space into discrete rectangular leaves (`max_depth=4`), its probability estimates are step-wise rather than smooth, resulting in the lowest **AUC (0.9302)** and slightly lower Precision (**0.8864**). |
| **KNN** | Tied for the highest overall **Accuracy (0.9474)**, achieved the highest **Precision (0.9744)** (only 1 false positive out of 71 benign test samples), and recorded the highest **MCC (0.8885)**. Distance-weighted $k$-Nearest Neighbors ($k=7$) after Z-score standardization effectively captures local neighborhood clusters in the 30-dimensional cytological space, with slightly more conservative Recall (**0.8837**) on borderline malignant samples. |
| **Naive Bayes** | Tied for the highest **Accuracy (0.9474)** and highest **Recall (0.9070)** while achieving the #1 **AUC (0.9864)** and #1 **F1 Score (0.9286)**, alongside an **MCC of 0.8876**. Because the standardized continuous nuclear measurements follow well-separated Gaussian class-conditional distributions, Gaussian Naive Bayes aggregates likelihood ratios across all 30 morphological features to produce superior probability ranking and sensitivity to malignant cases. |
| **Support Vector Machine** | Delivered balanced performance across all six metrics: **Accuracy (0.9298)**, **AUC (0.9722)**, **Precision (0.9268)**, **Recall (0.8837)**, **F1 (0.9048)**, and **MCC (0.8499)**. By projecting standardized features into a non-linear radial basis function (RBF) space with Platt probability calibration, SVM outperformed linear Logistic Regression in both Accuracy (+1.75%) and MCC (+3.78%). |
| **Overall, Winner for your dataset?** | **Naive Bayes (Gaussian Naive Bayes)** is the overall winner for medical diagnostic screening on this dataset, sharing the top **Accuracy (0.9474)** with **KNN** while leading all models in **AUC (0.9864)**, **F1 Score (0.9286)**, and **Recall (0.9070)**. In clinical oncology, maximizing Recall (minimizing missed malignant tumors / false negatives) while maintaining >95% Precision (**0.9512**) and near-0.89 MCC (**0.8876**) makes Gaussian Naive Bayes (closely followed by KNN at **0.8885** MCC) the most reliable classifier. |
