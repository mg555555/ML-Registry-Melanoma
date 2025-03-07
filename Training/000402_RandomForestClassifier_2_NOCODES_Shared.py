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
weight_cases = 1

print(type(MASTER))
print(MASTER.shape)
print(GB.table_nan(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"], MASTER["GROUP_TRAINVALTEST"]))


#-----
# CAN
#-----
index_CAN_start = list(MASTER).index("CAN_ICD10FULL_C300")
index_CAN_end   = list(MASTER).index("CAN_ICD10FULL_C496")
all_vars_CAN    = list(MASTER)[index_CAN_start:(index_CAN_end+1)]

#-------
# OV+SV
#-------
index_OV_SV_start  = list(MASTER).index("OV_ICD103st_")
index_OV_SV_end    = list(MASTER).index("SV_ICD103st_S25")
all_vars_OV_SV = list(MASTER)[index_OV_SV_start:(index_OV_SV_end+1)]

#------
# LMED
#------
index_LMED_start = list(MASTER).index("LMED_ATC_D10AD01")
index_LMED_end   = list(MASTER).index("LMED_ATC_B01AC04")
all_vars_LMED    = list(MASTER)[index_LMED_start:(index_LMED_end+1)]

#------
# BCC
#------
index_BCC_start = list(MASTER).index("CAN_BCC_EXISTS")
index_BCC_end   = list(MASTER).index("CAN_BCC_T02511")
all_vars_BCC    = list(MASTER)[index_BCC_start:(index_BCC_end+1)]


MASTER = MASTER.drop(all_vars_CAN + all_vars_OV_SV + all_vars_LMED + all_vars_BCC, axis=1).copy()

print(type(MASTER))

print(list(MASTER))



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


param_max_depth=12
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

modelstring = str(ROWS) + "_NOCODES_" + ("{:.4f}".format(auc))

filename="/RAWDATA/TRAINEDMODELS/BCC_CENS_LISA06_MIG/RandForest/NOCODES/" + str(clf) + hpstring + modelstring + ".sav"
pickle.dump(clf, open(filename, 'wb'))
