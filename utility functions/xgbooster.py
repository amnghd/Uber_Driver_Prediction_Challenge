import xgboost as xgb

import pandas as pd

import matplotlib.pyplot as plt

from sklearn.metrics import roc_auc_score, roc_curve, auc

import numpy as np

from sklearn.grid_search import GridSearchCV

 

 

 

 

X_train = pd.read_csv("X_train.csv", index_col='index').drop(columns="Unnamed: 0", axis=1)
y_train = pd.read_csv("y_train.csv", index_col="id").drop(columns="Unnamed: 0", axis=1)

X_test = pd.read_csv("X_test.csv", index_col='index').drop(columns="Unnamed: 0", axis=1)
y_test = pd.read_csv("y_test.csv", index_col="id").drop(columns="Unnamed: 0", axis=1)

params = {}
params["objective"] = "binary:logistic"
params["eval_metric"] = "auc"
params["gamma"] = 30
params["eta"] = 0.02
params["min_child_weight"] = 5
params["subsample"] = 0.6
params["colsample_bytree"] = 0.60
params["scale_pos_weight"] = 1.0
params["silent"] = 1
params["max_depth"] = 6
params["n_jobs"]=-1

plst = list(params.items())

xgtrain = xgb.DMatrix(X_train, label=y_train)
xgval = xgb.DMatrix(X_test, label=y_test)

#train using early stopping and predict

watchlist = [(xgtrain, 'train'),(xgval, 'val')]

num_rounds = 1000


model = xgb.train(plst, xgtrain, num_rounds,watchlist, early_stopping_rounds=10, verbose_eval=False)

preds1 = model.predict(xgval)

for loop in range(100):

    print(loop)

    model = xgb.train(plst, xgtrain, num_rounds,watchlist, early_stopping_rounds=10, verbose_eval=False)

    preds1 = preds1 + model.predict(xgval)

preds = (preds1/101 )

preds = preds[:,1]
fpr, tpr, threshold = roc_curve(y_test, preds)
roc_auc = auc(fpr, tpr)
plt.title('Receiver Operating Characteristic - K-Nearest Neighbors')
plt.plot(fpr, tpr, 'g', label = 'AUC = %0.2f' % roc_auc)
plt.fill_between(fpr, tpr,fpr, facecolor='violet', alpha=0.6)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'b--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()