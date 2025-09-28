import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix

def train_logreg(X, y):
    """Train baseline logistic regression with 10-fold CV."""
    skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    scores = []
    for train_idx, test_idx in skf.split(X, y):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        clf = LogisticRegression(max_iter=1000)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        cm = confusion_matrix(y_test, y_pred, labels=[1,0])
        scores.append(cm)
    return scores
