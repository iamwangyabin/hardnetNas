"""Utility methods for computing evaluating metrics. All methods assumes greater
scores for better matches, and assumes label == 1 means match.

"""
import numpy as np
# def ErrorRateAt95Recall(labels, scores):
#     distances = 1.0 / (scores + 1e-8)
#     # recall_point = 0.95
#     # import pdb
#     # pdb.set_trace()
#     labels = labels[np.argsort(distances)]
#     # threshold_index = np.argmax(np.cumsum(labels2) >= recall_point * np.sum(labels2))
#     # FP = np.sum(labels[:threshold_index] == 0) # Below threshold (i.e., labelled positive), but should be negative
#     # TN = np.sum(labels[threshold_index:] == 0) # Above threshold (i.e., labelled negative), and should be negative
#     # TP = np.sum(labels[:threshold_index] == 1)
#     TP = np.sum(labels[:50000] == 1)
#     TN = np.sum(labels[50000:] == 0)
#     return (TP+TN)/len(labels)     # float(FP) / float(FP + TN)


# 显而易见，这个值越小越好
def ErrorRateAt95Recall(labels, scores):
    distances = 1.0 / (scores + 1e-8)
    recall_point = 0.95
    labels = labels[np.argsort(distances)]
    # Sliding threshold: get first index where recall >= recall_point.
    # This is the index where the number of elements with label==1 below the threshold reaches a fraction of
    # 'recall_point' of the total number of elements with label==1.
    # (np.argmax returns the first occurrence of a '1' in a bool array).
    threshold_index = np.argmax(np.cumsum(labels) >= recall_point * np.sum(labels))

    FP = np.sum(labels[:threshold_index] == 0) # 实际为负例但被分类器划分为正例的实例数
    TN = np.sum(labels[threshold_index:] == 0) # 实际为负例且被分类器划分为负例的实例数

    return float(FP) / float(FP + TN)



