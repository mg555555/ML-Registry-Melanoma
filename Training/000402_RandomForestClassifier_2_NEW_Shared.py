import _Globalconstants as GB
import time
import pandas as pd
import datetime
import sys
import numpy as np
import pickle


import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score

from sklearn.model_selection import cross_validate
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA

from scipy.stats import norm



print("Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle")
MASTER = pd.read_pickle("/RAWDATA/_099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle").copy()
print("Finished Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle")



MASTER_train = MASTER[MASTER["GROUP_TRAINVALTEST"]=="train"]
MASTER_val   = MASTER[MASTER["GROUP_TRAINVALTEST"]=="val"]


print("Size of MASTER_train: ", MASTER_train.shape)
print("Size of MASTER_val: ",   MASTER_val.shape)


MASTER_train = MASTER_train.drop(["GROUP_TRAINVALTEST"], axis=1).copy()
MASTER_val   = MASTER_val.drop(["GROUP_TRAINVALTEST"], axis=1).copy()

X_train = MASTER_train.loc[:, (MASTER_train.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE')]
X_val   = MASTER_val.loc[:, (MASTER_val.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE')]

y_train = MASTER_train.loc[:, MASTER_train.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']
y_train = y_train.values.ravel()

y_val = MASTER_val.loc[:, MASTER_val.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']
y_val = y_val.values.ravel()


param_max_depth=18
param_n_estimators=1000

hpstring = "_max_depth_" + str("{:.0f}".format(param_max_depth)) + "_n_est_" + str("{:.0f}".format(param_n_estimators)) + "_"



clf = RandomForestClassifier(verbose=True, max_depth=param_max_depth, n_estimators=param_n_estimators)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)

print("Classifier:", clf)


y_pred_proba = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)
print("Val AUC=", auc)

ROWS = MASTER.shape[0]

modelstring = str(ROWS) + "_ALLCols_UPSAMPLED_AllRows_" + ("{:.4f}".format(auc))

filename="/RAWDATA/TRAINEDMODELS/BCC_CENS_LISA06_MIG/RandForest/" + str(clf) + hpstring + modelstring + ".sav"
pickle.dump(clf, open(filename, 'wb'))
