import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from itertools import cycle
import os

from utils import NUM_CLASSES, EMOTIONS_ID2LABEL

def plot_roc_curves(
    y_true_bin,
    y_pred_proba,
    model_name,
    save_dir="plots"
):
    """plot for each class, micro-average, and macro-average."""
    print(f"\nGenerating ROC curves for {model_name}...")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if not isinstance(y_true_bin, np.ndarray):
        y_true_bin = np.array(y_true_bin)
    if not isinstance(y_pred_proba, np.ndarray):
        y_pred_proba = np.array(y_pred_proba)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    print("Calculating per-class ROC AUC...")
    for i in range(NUM_CLASSES):
        if np.sum(y_true_bin[:, i]) > 0:
            fpr[i], tpr[i], _ = roc_curve(y_true_bin[:, i], y_pred_proba[:, i])
            roc_auc[i] = auc(fpr[i], tpr[i])
        else:
            fpr[i], tpr[i], roc_auc[i] = None, None, None

    print("Calculating micro-average ROC AUC...")
    fpr["micro"], tpr["micro"], _ = roc_curve(y_true_bin.ravel(), y_pred_proba.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    if np.isnan(roc_auc["micro"]):
        print("  Warning: NaN AUC calculated for micro-average. Setting to 0.0.")
        roc_auc["micro"] = 0.0

    print("Calculating macro-average ROC AUC...")
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(NUM_CLASSES) if fpr[i] is not None]))
    mean_tpr = np.zeros_like(all_fpr)
    valid_classes_count = 0
    for i in range(NUM_CLASSES):
        if fpr[i] is not None and tpr[i] is not None:
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
            valid_classes_count += 1

    if valid_classes_count > 0:
        mean_tpr /= valid_classes_count
        fpr["macro"] = all_fpr
        tpr["macro"] = mean_tpr
        roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    else:
        print("  Warning: No valid classes found for macro-average ROC. Setting AUC to 0.0.")
        fpr["macro"] = np.array([0, 1])
        tpr["macro"] = np.array([0, 1])
        roc_auc["macro"] = 0.0

    if np.isnan(roc_auc["macro"]):
         print("  Warning: NaN AUC calculated for macro-average. Setting to 0.0.")
         roc_auc["macro"] = 0.0

    # plotting
    plt.figure(figsize=(12, 10))

    plt.plot(
        fpr["micro"],
        tpr["micro"],
        label=f"Micro-average ROC curve (area = {roc_auc.get('micro', 0.0):0.3f})",
        color="deeppink",
        linestyle=":",
        linewidth=4,
    )

    plt.plot(
        fpr["macro"],
        tpr["macro"],
        label=f"Macro-average ROC curve (area = {roc_auc.get('macro', 0.0):0.3f})",
        color="navy",
        linestyle=":",
        linewidth=4,
    )

    cmap = plt.get_cmap('tab20')
    colors = [cmap(i/NUM_CLASSES) for i in range(NUM_CLASSES)]

    for i, color in zip(range(NUM_CLASSES), colors):
        if roc_auc.get(i) is not None and fpr.get(i) is not None and tpr.get(i) is not None:
            plt.plot(
                fpr[i], tpr[i], color=color, lw=1.5,
                label=f"ROC curve of class {i} ({EMOTIONS_ID2LABEL[i]}) (area = {roc_auc[i]:0.3f})",
            )
    plt.plot([0, 1], [0, 1], "k--", lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"Multi-label ROC Curves for {model_name}")
    
    #legend
    plt.legend(loc="center left", bbox_to_anchor=(1, 0.5), fontsize='small')
    plt.grid(True)
    plt.tight_layout(rect=[0, 0, 0.85, 1]) # Adjust layout to make space for legend

    plot_filename = os.path.join(save_dir, f"roc_curves_{model_name.lower().replace(' ', '_')}.png")
    plt.savefig(plot_filename)
    plt.close()

    return roc_auc