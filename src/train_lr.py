import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix, accuracy_score
from imblearn.over_sampling import RandomOverSampler

def compute_score(cm):
    # confusion matrix ke elements extract karo
    # sklearn ka cm: [[TN, FP], [FN, TP]]
    TN, FP, FN, TP = cm.ravel()

    Se = TP / (TP + FN + 1e-6)   # Sensitivity (True Positive Rate)
    Sp = TN / (TN + FP + 1e-6)   # Specificity (True Negative Rate)
    Score = 0.5 * (Se + Sp)      # Balanced score
    return Se, Sp, Score

def train_logreg(X, y, n_splits=10):
    kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    accuracies, scores, conf_matrices = [], [], []
    sensitivities, specificities = [], []

    for fold, (train_index, test_index) in enumerate(kf.split(X, y), 1):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        # class imbalance handle karne ke liye oversampling
        ros = RandomOverSampler(random_state=42)
        X_res, y_res = ros.fit_resample(X_train, y_train)

        clf = LogisticRegression(max_iter=1000, solver="liblinear")
        clf.fit(X_res, y_res)
        y_pred = clf.predict(X_test)

        # confusion matrix banao aur save karo
        cm = confusion_matrix(y_test, y_pred, labels=[0,1])
        conf_matrices.append(cm.tolist())   # list me save for JSON compatibility

        Se, Sp, Score = compute_score(cm)
        sensitivities.append(Se)
        specificities.append(Sp)
        scores.append(Score)
        acc = accuracy_score(y_test, y_pred)
        accuracies.append(acc)

        # optional print har fold ka
        print(f"\n=== Fold {fold} ===")
        print("Confusion Matrix:\n", cm)
        print(f"Accuracy: {acc:.4f}, Sensitivity: {Se:.4f}, Specificity: {Sp:.4f}, Score: {Score:.4f}")

    results = {
        "conf_matrices": conf_matrices,
        "accuracies": accuracies,
        "scores": scores,
        "sensitivities": sensitivities,
        "specificities": specificities,
        "average_accuracy": float(np.mean(accuracies)),
        "average_score": float(np.mean(scores)),
        "average_sensitivity": float(np.mean(sensitivities)),
        "average_specificity": float(np.mean(specificities)),
    }
    return results
