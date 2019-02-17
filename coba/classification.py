from sklearn import svm
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

X, y = make_classification(n_samples=10, n_features=4, n_informative=2, n_redundant=0, random_state=0, shuffle=False)

rf = RandomForestClassifier(random_state=0, verbose=1, n_jobs=-1)
svc = svm.SVC()

print(X)
print(y)

rf.fit(X, y)
svc.fit(X, y)

print(rf.predict([
    [0, 0, 0, 0],
    [0, 1, 0, 1]
]))

print(svc.predict([
    [0, 0, 0, 0],
    [0, 1, 0, 1]
]))
