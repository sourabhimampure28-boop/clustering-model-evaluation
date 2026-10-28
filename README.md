# Clustering and Model Evaluation Using Scikit-learn

A beginner-friendly machine-learning project demonstrating unsupervised clustering, dimensionality reduction, classification, model evaluation, cross-validation, and hyperparameter tuning.

## Project objectives

- Compare K-Means and Agglomerative (hierarchical) clustering on the Iris dataset.
- Reduce four standardized Iris features to two principal components using PCA.
- Train Logistic Regression on the Wisconsin Diagnostic Breast Cancer dataset.
- Evaluate the classifier using accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Use five-fold stratified cross-validation and `GridSearchCV` to tune the regularization parameter `C`.

## Project structure

```text
clustering_model_evaluation_github/
├── main.py
├── requirements.txt
├── README.md
├── Clustering_and_Model_Evaluation_Report.docx
└── results/
    ├── clustering_results.csv
    ├── pca_results.csv
    ├── pca_variance.csv
    ├── classification_results.csv
    ├── confusion_matrix.csv
    ├── cross_validation_results.csv
    └── grid_search_results.csv
```

## Requirements

- Python 3.10 or later
- pandas
- scikit-learn

## Run the project

1. Download or clone this repository.
2. Open a terminal in the project folder.
3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the script:

   ```bash
   python main.py
   ```

5. Check the generated CSV files in the `results/` directory.

## Datasets

The Iris and Wisconsin Diagnostic Breast Cancer datasets are loaded directly from Scikit-learn. Separate raw data files are not required.

## Evaluation notes

Clustering quality is summarized with the silhouette score and Adjusted Rand Index (ARI). PCA output and explained variance are exported to CSV. The classification metrics are calculated on a held-out 20% test split, with stratification and a fixed random seed. `GridSearchCV` uses five-fold stratified cross-validation to select `C` by F1-score.

The exact numeric results may differ from a separately prepared report if the random seed, library versions, preprocessing, or model settings differ. Run `main.py` to generate results for this repository's code.

## Report

See `Clustering_and_Model_Evaluation_Report.docx` for the accompanying project report.

## Disclaimer

This is an educational machine-learning exercise. It is not a medical diagnostic tool and should not be used to make clinical decisions.
