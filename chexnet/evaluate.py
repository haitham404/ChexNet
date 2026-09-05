"""Evaluation metrics for multi-label classification."""

import numpy as np
from sklearn.metrics import roc_auc_score, f1_score


class Metrics:
    """Computation of AUROC, F1-score, and Bootstrap CIs."""

    @staticmethod
    def compute_auroc(y_true, y_prob, class_count: int):
        """Compute AUROC per class (one-vs-rest)."""
        y_true = y_true.detach().cpu().numpy()
        y_prob = y_prob.detach().cpu().numpy()
        aurocs = []
        for c in range(class_count):
            gt, prob = y_true[:, c], y_prob[:, c]
            if len(np.unique(gt)) < 2:
                aurocs.append(float("nan"))
            else:
                aurocs.append(roc_auc_score(gt, prob))
        return aurocs

    @staticmethod
    def compute_pneumonia_f1(y_true, y_prob, pneumonia_idx: int = 6):
        """Compute F1-score for pneumonia at optimal threshold."""
        y_true = y_true.detach().cpu().numpy()[:, pneumonia_idx]
        y_prob = y_prob.detach().cpu().numpy()[:, pneumonia_idx]
        thresholds = np.linspace(0.05, 0.9, 50)
        f1_list = [f1_score(y_true, (y_prob >= t).astype(int),
                            zero_division=0)
                   for t in thresholds]
        best_i = int(np.argmax(f1_list))
        return f1_list[best_i], thresholds[best_i]

    @staticmethod
    def bootstrap_auroc(y_true, y_prob, class_count, n_boot: int = 1000, seed: int = 42):
        """Bootstrap confidence intervals for AUROC."""
        y_true = y_true.detach().cpu().numpy()
        y_prob = y_prob.detach().cpu().numpy()
        rng = np.random.default_rng(seed)
        N = len(y_true)
        scores = []
        for _ in range(n_boot):
            idx = rng.choice(N, N, replace=True)
            row = []
            for c in range(class_count):
                gt, prob = y_true[idx, c], y_prob[idx, c]
                if len(np.unique(gt)) < 2:
                    row.append(float("nan"))
                else:
                    row.append(roc_auc_score(gt, prob))
            scores.append(row)
        scores = np.array(scores)
        return {
            "mean": np.nanmean(scores, axis=0),
            "std": np.nanstd(scores, axis=0),
            "lower": np.nanpercentile(scores, 2.5, axis=0),
            "upper": np.nanpercentile(scores, 97.5, axis=0),
        }