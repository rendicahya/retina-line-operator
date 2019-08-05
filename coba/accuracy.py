import numpy as np
from sklearn.metrics import classification_report

from util.data_util import confusion_matrix

pred = np.zeros((4, 4), np.uint8)
true = pred.copy()

pred[2:, 2:] = 1
true[1:-1, 1:-1] = 1

pred = pred.ravel()
true = true.ravel()

acc = classification_report(true, pred)
tn, fp, fn, tp = confusion_matrix(true, pred).ravel()
tn_, fp_, fn_, tp_ = confusion_matrix(true, pred).ravel()

print(pred)
print(true)
print(tn, fp, fn, tp)
print(tn_, fp_, fn_, tp_)
