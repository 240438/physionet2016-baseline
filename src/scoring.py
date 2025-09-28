def compute_score(cm, w=0.5):
    """
    cm = confusion matrix [ [TP, FN], [FP, TN] ]
    Return Se, Sp, Score
    """
    TP, FN = cm[0]
    FP, TN = cm[1]

    # In this simple baseline, no 'unsure' => Aq=Nq=0
    Se = TP / (TP + FN + 1e-6)
    Sp = TN / (TN + FP + 1e-6)
    Score = 0.5 * (Se + Sp)
    return Se, Sp, Score
