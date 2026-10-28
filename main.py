"""
Clustering and Model Evaluation Using Scikit-learn
Run: python main.py
Outputs are written to the results/ directory.
"""

from pathlib import Path
import pandas as pd
from sklearn.datasets import load_iris, load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score, adjusted_rand_score, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# 1. Clustering experiment: Iris dataset
iris = load_iris()
X_iris = iris.data
y_iris = iris.target
X_scaled = StandardScaler().fit_transform(X_iris)

cluster_rows = []
cluster_labels = {}
models = {
    "K-Means (k=3)": KMeans(n_clusters=3, random_state=42, n_init=10),
    "Agglomerative (k=3)": AgglomerativeClustering(n_clusters=3),
}
for name, model in models.items():
    labels = model.fit_predict(X_scaled)
    cluster_labels[name] = labels
    cluster_rows.append({
        "method": name,
        "clusters": len(set(labels)),
        "silhouette_score": silhouette_score(X_scaled, labels),
        "adjusted_rand_index": adjusted_rand_score(y_iris, labels),
    })
pd.DataFrame(cluster_rows).to_csv(OUT / "clustering_results.csv", index=False)

# 2. PCA: reduce standardized Iris features to two components
pca = PCA(n_components=2)
iris_pca = pca.fit_transform(X_scaled)
pd.DataFrame({
    "principal_component_1": iris_pca[:, 0],
    "principal_component_2": iris_pca[:, 1],
    "species": [iris.target_names[i] for i in y_iris],
}).to_csv(OUT / "pca_results.csv", index=False)

pd.DataFrame({
    "component": ["PC1", "PC2"],
    "explained_variance_ratio": pca.explained_variance_ratio_,
}).to_csv(OUT / "pca_variance.csv", index=False)

# 3. Classification: Wisconsin Diagnostic Breast Cancer dataset
cancer = load_breast_cancer()
X_train, X_test, y_train, y_test = train_test_split(
    cancer.data, cancer.target, test_size=0.20,
    random_state=42, stratify=cancer.target
)

pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(max_iter=10000, solver="liblinear")),
])
param_grid = {"model__C": [0.01, 0.1, 1, 10, 100]}
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
search = GridSearchCV(
    pipeline, param_grid, cv=cv, scoring="f1", n_jobs=-1,
    return_train_score=True
)
search.fit(X_train, y_train)
best_model = search.best_estimator_

y_pred = best_model.predict(X_test)
y_prob = best_model.predict_proba(X_test)[:, 1]

test_metrics = {
    "accuracy": accuracy_score(y_test, y_pred),
    "precision": precision_score(y_test, y_pred, zero_division=0),
    "recall": recall_score(y_test, y_pred, zero_division=0),
    "f1_score": f1_score(y_test, y_pred, zero_division=0),
    "roc_auc": roc_auc_score(y_test, y_prob),
}
pd.DataFrame([test_metrics]).to_csv(OUT / "classification_results.csv", index=False)

cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
pd.DataFrame(
    cm,
    index=["actual_class_0", "actual_class_1"],
    columns=["predicted_class_0", "predicted_class_1"],
).to_csv(OUT / "confusion_matrix.csv", index=True)

# 4. Five-fold cross-validation using the selected hyperparameter
cv_results = cross_validate(
    best_model, X_train, y_train, cv=cv,
    scoring=["accuracy", "precision", "recall", "f1", "roc_auc"],
    n_jobs=-1
)
cv_summary = {
    metric.replace("test_", ""): scores.mean()
    for metric, scores in cv_results.items()
    if metric.startswith("test_")
}
pd.DataFrame([cv_summary]).to_csv(OUT / "cross_validation_results.csv", index=False)

# 5. Grid search results
grid_results = pd.DataFrame(search.cv_results_)
grid_results[[
    "param_model__C", "mean_test_score", "std_test_score",
    "rank_test_score", "mean_train_score"
]].to_csv(OUT / "grid_search_results.csv", index=False)

print("Project complete. Results saved to:", OUT)
print("\nClustering results:\n", pd.DataFrame(cluster_rows).round(4).to_string(index=False))
print("\nBest parameter:", search.best_params_)
print("Held-out test metrics:", {k: round(v, 4) for k, v in test_metrics.items()})
print("Cross-validation means:", {k: round(v, 4) for k, v in cv_summary.items()})
