from xgboost import XGBClassifier
import pandas as pd
from sklearn.grid_search import GridSearchCV
import time

t_start = time.time()

X_train = pd.read_csv("X_train.csv").drop(columns=["Unnamed: 0","index"], axis=1)
y_train = pd.read_csv("y_train.csv").drop(columns=["Unnamed: 0","id"], axis=1).target.values


parameters = {'min_child_weight': [5, 10],
        'gamma': [0.01, .1,1],
        'subsample': [0.8,1],
        'colsample_bytree': [0.6, 0.8],
        'max_depth': [3, 5,  7]
        }

parameters = {'learning_rate':[0.02],
        'min_child_weight': [5],
        'gamma': [30],
        'subsample': [0.6],
        'colsample_bytree': [0.6],
        'max_depth': [6],
        'n_estimators':[1000]
        }



xgbc = XGBClassifier(objective='binary:logistic',
                    silent=True, verbose=True, scale_pos_weight=8)# 'scale_pos_weight' to set as the ratio for skewed data

clf_xgbc=GridSearchCV(estimator=xgbc,param_grid=parameters, scoring='roc_auc', cv=3, n_jobs=1,verbose=100)

clf_xgbc.fit(X_train, y_train)


print("Best parameters are {}".format(clf_xgbc.best_params_))
print("Best score is {}".format(clf_xgbc.best_score_))


t_end = time.time()
print("It took {} seconds to run this test".format(t_end-t_start))