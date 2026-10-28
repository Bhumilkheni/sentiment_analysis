"""Metric + confusion-matrix helpers used by both notebooks."""
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support)


def evaluate(y_true, y_pred, name="model"):
    """Accuracy + macro-averaged Precision / Recall / F1 (treats all classes equally)."""
    p, r, f, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return {
        "Model": name,
        "Accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "Precision": round(float(p), 4),
        "Recall": round(float(r), 4),
        "F1": round(float(f), 4),
    }


def plot_confusion_matrix(y_true, y_pred, labels, title, save_path=None):
    """Counts (left) and row-normalised percentages (right)."""
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[0])
    axes[0].set_title("Counts")
    sns.heatmap(cm_norm, annot=True, fmt=".0%", cmap="Blues", cbar=False,
                xticklabels=labels, yticklabels=labels, ax=axes[1])
    axes[1].set_title("Row-normalised (recall per class)")
    for ax in axes:
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
