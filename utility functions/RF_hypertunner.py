import pandas as pd
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score
import time


t_start = time.time()

X_train = pd.read_csv("X_train.csv", index_col='index').drop(columns='Unnamed: 0', axis=1)
y_train = pd.read_csv("y_train.csv", index_col='id').drop(columns='Unnamed: 0', axis=1)



scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
rf=RandomForestClassifier(n_jobs=-1)
param_grid = {"max_depth": [7, 15, None],
              "max_features": ["sqrt", "log2", 10],
              "min_samples_split": [100,200,1000],
              "min_samples_leaf": [10, 50, 100],
              "bootstrap": [True, False],
              "criterion": ["gini", "entropy"]}

clf_rf = GridSearchCV(rf, param_grid=param_grid,scoring=scoring, cv=3, refit='AUC')
clf_rf.fit(X_train, y_train)
print("Best parameters are {}".format(clf_rf.best_params_))
print("Best score is {}".format(clf_rf.best_score_))


t_end = time.time()


print("It took {} seconds to run this test".format(t_end-t_start))