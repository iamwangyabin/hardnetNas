"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.

"""
import numpy as np
def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    # recall_point = 0.95
    # import pdb
    # pdb.set_trace()
    labels = labels[np.argsort(distances)]
    # threshold_index = np.argmax(np.cumsum(labels2) >= recall_point * np.sum(labels2))
    # FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
    # TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
    # TP = np.sum(labels[:threshold_index] == 1)
    TP = np.sum(labels[:50000] == 1)
    TN = np.sum(labels[50000:] == 0)
    return (TP+TN)/len(labels)     # float(FP) / float(FP + TN)
