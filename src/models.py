from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import f1_score
import numpy as np

def get_cv_splitter():
    """Returns the 5-fold grouped cross-validation splitter."""
    return StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=0)

def compare_logistic_class_weights(X_train, y_train, groups):
    """Compares None vs 'balanced' class weights using 5-fold grouped CV."""
    cv = get_cv_splitter()
    results = {}
    for cw in [None, "balanced"]:
        pipe = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(solver="lbfgs", max_iter=2000, class_weight=cw))
        ])
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train, groups=groups):
            pipe.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(f1_score(y_train[va_idx], pipe.predict(X_train[va_idx]), average="macro"))
        results[cw] = np.array(scores)
    return results

def tune_rbf_svm(X_train, y_train, groups):
    """Grid search for RBF SVM over C and gamma inside a scaling pipeline."""
    pipe = Pipeline([("scaler", StandardScaler()), ("svc", SVC(kernel="rbf"))])
    param_grid = {
        "svc__C": [0.01, 0.1, 1, 10, 100],
        "svc__gamma": [0.001, 0.01, 0.1, 1, 10]
    }
    grid = GridSearchCV(pipe, param_grid, cv=get_cv_splitter(), scoring="f1_macro", n_jobs=-1)
    grid.fit(X_train, y_train, groups=groups)
    return grid

def sweep_random_forest(X_train, y_train, groups, estimators_list):
    """Evaluates Random Forest across a sweep of n_estimators using grouped CV."""
    cv = get_cv_splitter()
    rf_cv_scores = {}
    for n_est in estimators_list:
        rf = RandomForestClassifier(n_estimators=n_est, max_depth=10, random_state=42, n_jobs=-1)
        scores = []
        for tr_idx, va_idx in cv.split(X_train, y_train, groups=groups):
            rf.fit(X_train[tr_idx], y_train[tr_idx])
            scores.append(f1_score(y_train[va_idx], rf.predict(X_train[va_idx]), average="macro"))
        rf_cv_scores[n_est] = np.array(scores)
    return rf_cv_scores

def build_hard_voting_ensemble(logreg_pipe, svm_pipe, rf_model):
    """Combines the three tuned classifiers into a hard-voting ensemble (Problem 5e)."""
    return VotingClassifier(
        estimators=[('lr', logreg_pipe), ('svm', svm_pipe), ('rf', rf_model)],
        voting='hard'
    )