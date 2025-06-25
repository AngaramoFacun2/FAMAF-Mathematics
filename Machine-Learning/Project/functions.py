import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay,
    classification_report, accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, average_precision_score
)

def results(model, X: np.ndarray, y: np.ndarray, model_name: str) -> None:

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(14, 12))

    fig.suptitle(model_name, fontsize=32, fontweight="bold", y=1)

    ConfusionMatrixDisplay \
        .from_estimator(model, X, y, ax=ax[0, 0], colorbar=False, display_labels=["No", "Yes"])
    ax[0, 0].set_title(f"Confusion Matrix", fontsize=18)
    ax[0, 0].set_xlabel("Predicted Label", fontsize=16)
    ax[0, 0].set_ylabel("True Label", fontsize=16)

    ax[0 ,1].text(
        x=0.5, 
        y=0.5, 
        s=classification_report(y, model.predict(X), target_names=["No", "Yes"]), 
        ha="center", 
        va="center", 
        fontsize=16, 
        fontfamily="monospace"
    )
    ax[0, 1].set_title("Classification Report", fontsize=18)
    ax[0, 1].axis("off")

    RocCurveDisplay \
        .from_estimator(model, X, y, ax=ax[1, 0], plot_chance_level=True, name=model_name, color="tomato")
    ax[1, 0].set_title("ROC Curve", fontsize=18)
    ax[1, 0].set_xlabel("False Positive Rate", fontsize=16)
    ax[1, 0].set_ylabel("True Positive Rate", fontsize=16)

    PrecisionRecallDisplay \
        .from_estimator(model, X, y, ax=ax[1, 1], plot_chance_level=True, name=model_name, color="steelblue")
    ax[1, 1].set_title("Precision-Recall Curve", fontsize=18)
    ax[1, 1].set_xlabel("Recall", fontsize=16)
    ax[1, 1].set_ylabel("Precision", fontsize=16)

    plt.tight_layout()
    plt.show()


def summary(models: dict, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    
    s = []

    for name, model in models.items():
        y_pred = model.predict(X)
        y_proba = model.predict_proba(X)[:, 1]

        s.append({
            "Model": name,
            "Accuracy": accuracy_score(y, y_pred),
            "Precision": precision_score(y, y_pred),
            "Recall": recall_score(y, y_pred),
            "F1": f1_score(y, y_pred),
            "AUC": roc_auc_score(y, y_proba),
            "AP": average_precision_score(y, y_proba)
        })
        
        RocCurveDisplay \
            .from_estimator(model, X, y, ax=ax[0], name=name)
        
        PrecisionRecallDisplay \
            .from_estimator(model, X, y, ax=ax[1], name=name)

    ax[0].plot([0, 1], [0, 1], color="black", linestyle="--")
    ax[0].set_title("ROC Curve", fontsize=18)
    ax[0].set_xlabel("False Positive Rate", fontsize=16)
    ax[0].set_ylabel("True Positive Rate", fontsize=16)

    ax[1].axhline(y=0.22, color="black", linestyle="--")
    ax[1].set_title("Precision-Recall Curve", fontsize=18)
    ax[1].set_xlabel("Recall", fontsize=16)
    ax[1].set_ylabel("Precision", fontsize=16)
    
    return pd.DataFrame(s).round(4)