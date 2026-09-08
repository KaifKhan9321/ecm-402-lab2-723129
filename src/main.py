import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

from data import load_metadata, create_grouped_split
from features import extract_all_features, FEATURE_NAMES
from models import compare_logistic_class_weights, tune_rbf_svm, sweep_random_forest, build_hard_voting_ensemble
from evaluate import evaluate_predictions, run_paired_ttest, analyze_errors, plot_conf_matrix

def main():
    # 1. Load Data & Split
    print("Loading metadata and splitting...")
    meta_df = load_metadata("free-spoken-digit-dataset/recordings")
    train_mask, test_mask = create_grouped_split(meta_df)
    
    # 2. Extract Features
    print("Extracting features (this may take a moment)...")
    X_all = extract_all_features(meta_df["path"].tolist())
    y_all = meta_df["digit"].to_numpy()
    groups_all = meta_df["speaker"].to_numpy()
    
    X_train, y_train, groups_train = X_all[train_mask], y_all[train_mask], groups_all[train_mask]
    X_test, y_test, groups_test = X_all[test_mask], y_all[test_mask], groups_all[test_mask]
    
    # 3. Tune Logistic Regression
    print("\n--- Tuning Logistic Regression ---")
    lr_results = compare_logistic_class_weights(X_train, y_train, groups_train)
    best_lr_scores = lr_results[None] 
    
    # 4. Tune RBF SVM
    print("\n--- Tuning RBF SVM ---")
    svm_grid = tune_rbf_svm(X_train, y_train, groups_train)
    print(f"Best SVM Params: {svm_grid.best_params_}")
    
    # 5. Tune Random Forest
    print("\n--- Tuning Random Forest ---")
    rf_results = sweep_random_forest(X_train, y_train, groups_train, [10, 25, 50, 100, 200])
    best_n_est = max(rf_results, key=lambda k: np.mean(rf_results[k]))
    best_rf_scores = rf_results[best_n_est]
    print(f"Best n_estimators: {best_n_est}")
    
    # 6. Statistical Comparison (Validation Folds)
    print("\n--- Model Comparison (Validation Folds) ---")
    best_svm_idx = svm_grid.best_index_
    best_svm_scores = np.array([
        svm_grid.cv_results_[f'split{i}_test_score'][best_svm_idx] 
        for i in range(5)
    ])
    
    # Save the t-test results to the tables folder
    run_paired_ttest(best_svm_scores, best_rf_scores, ("RBF SVM", "Random Forest"), 
                     save_path="results/tables/paired_ttest_results.txt")
    
    # 7. Final Test Evaluation (Touched Once!)
    print("\n--- FINAL TEST SET EVALUATION ---")
    best_svm = svm_grid.best_estimator_
    
    # Retrain on full train set
    best_svm.fit(X_train, y_train)
    test_preds = best_svm.predict(X_test)
    
    # Save metrics to tables folder
    evaluate_predictions(y_test, test_preds, dataset_name="Final Test (RBF SVM)", save_dir="results/tables")
    
    # Save confusion matrix to figures folder
    plot_conf_matrix(y_test, test_preds, classes=range(10), title="Test Confusion Matrix (SVM)", 
                     save_path="results/figures/test_confusion_matrix.png")
    
    # Save Error analysis visualizations to figures folder
    test_paths = np.array(meta_df["path"].tolist())[test_mask]
    analyze_errors(test_paths, y_test, test_preds, num_examples=5, save_dir="results/figures/errors")

    # 8. Bonus: Hard-Voting Ensemble
    print("\n--- BONUS: HARD-VOTING ENSEMBLE ---")
    lr_pipe = Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=None))])
    svm_pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf", C=0.1, gamma=0.01))])
    rf_model = RandomForestClassifier(n_estimators=best_n_est, max_depth=10, random_state=42)
    
    ensemble = build_hard_voting_ensemble(lr_pipe, svm_pipe, rf_model)
    ensemble.fit(X_train, y_train)
    ens_preds = ensemble.predict(X_test)
    
    # Save ensemble metrics to tables folder
    evaluate_predictions(y_test, ens_preds, dataset_name="Final Test (Voting Ensemble)", save_dir="results/tables")

if __name__ == "__main__":
    main()