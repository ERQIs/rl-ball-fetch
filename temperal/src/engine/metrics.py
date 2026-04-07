import numpy as np


def l2_error(preds, targets):
    d = preds - targets
    return np.linalg.norm(d, axis=1)


def mean_l2_error(preds, targets):
    return float(np.mean(l2_error(preds, targets)))


def median_l2_error(preds, targets):
    return float(np.median(l2_error(preds, targets)))


def success_rate_at_threshold(preds, targets, threshold):
    errs = l2_error(preds, targets)
    return float((errs <= threshold).mean())
