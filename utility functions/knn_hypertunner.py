import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import time


t_start = time.time()

X_train = pd.read_csv("X_train.csv", index_col='index').drop(columns='Unnamed: 0', axis=1)
y_train = pd.read_csv("y_train.csv", index_col='id').drop(columns='Unnamed: 0', axis=1)



scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
knn=KNeighborsClassifier(n_jobs=-1)
parameters={'n_neighbors':[10,100],'weights':['uniform', 'distance']}
clf_knn=GridSearchCV(estimator=knn,param_grid=parameters, scoring=scoring, cv=3, refit='AUC')
clf_knn.fit(X_train, y_train)
print("Best parameters are {}".format(clf_knn.best_params_))
print("Best score is {}".format(clf_knn.best_score_))




t_end = time.time()


print("It took {} seconds to run this test".format(t_end-t_start))