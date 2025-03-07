import _Globalconstants as GB
import time
import pandas as pd
import datetime
import sys
# import random
import numpy as np
import pickle




""
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


print("Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle")
MASTER = pd.read_pickle("/RAWDATA/_099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle").copy()
print("Finished Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle")
weight_cases = 5026682/32505


MASTER["WEIGHTS"] = 1.0
print("MASTER.shape[0] = ", MASTER.shape[0])
print("MASTER weights only ones?:")
print(GB.table_nan(MASTER["WEIGHTS"]))
print("weight_cases = ", weight_cases)
MASTER.loc[MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"], "WEIGHTS"] = weight_cases
print("Test if WEIGHTS is defined correctly:")
print(GB.table_nan(MASTER["WEIGHTS"], MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]))


MASTER = MASTER[["MM_BETWEEN_INDEX_AND_ENDDATE", "GROUP_TRAINVALTEST", "Kon_equals_1", "AGE_INDEX_5_(-inf, 35.0]", "AGE_INDEX_5_(35.0, 40.0]", "AGE_INDEX_5_(40.0, 45.0]", "AGE_INDEX_5_(45.0, 50.0]", "AGE_INDEX_5_(50.0, 55.0]", "AGE_INDEX_5_(55.0, 60.0]", "AGE_INDEX_5_(60.0, 65.0]", "AGE_INDEX_5_(65.0, 70.0]", "AGE_INDEX_5_(70.0, 75.0]", "AGE_INDEX_5_(75.0, 80.0]", "AGE_INDEX_5_(80.0, 85.0]", "AGE_INDEX_5_(85.0, inf]"]].copy()



MASTER_train = MASTER[MASTER["GROUP_TRAINVALTEST"]=="train"]
MASTER_val   = MASTER[MASTER["GROUP_TRAINVALTEST"]=="val"]


print("Size of MASTER_train: ", MASTER_train.shape)
print("Size of MASTER_val: ",   MASTER_val.shape)

MASTER_train = MASTER_train.drop(["GROUP_TRAINVALTEST"], axis=1).copy()
MASTER_val   = MASTER_val.drop(["GROUP_TRAINVALTEST"], axis=1).copy()


X_train = MASTER_train.loc[:, (MASTER_train.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE')]
X_val = MASTER_val.loc[:, (MASTER_val.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE')]

y_train = MASTER_train.loc[:, MASTER_train.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']
y_train = y_train.values.ravel()

y_val = MASTER_val.loc[:, MASTER_val.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']
y_val = y_val.values.ravel()

param_C=0.01
param_penalty="l2"
param_solver="liblinear"

hpstring = "_C_" + str("{:.4f}".format(param_C)) + "_penalty_" + param_penalty + "_solver_" + param_solver + "_"

clf = LogisticRegression(verbose=True, C=param_C, penalty=param_penalty, solver=param_solver)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_val)
    
print("Classifier:", clf)
    
acc = accuracy_score(y_val, y_pred)
precision, recall, _ = precision_recall_curve(y_val, y_pred)
fpr, tpr, _ = roc_curve(y_val, y_pred)
f1 = f1_score(y_val, y_pred)


y_pred_proba = clf.predict_proba(X_val)[:, 1]
auc = roc_auc_score(y_val, y_pred_proba)

ROWS = MASTER.shape[0]
    
modelstring = str(ROWS) + "_ONLY_AGESEX_" + ("{:.4f}".format(auc))

filename="/RAWDATA/TRAINEDMODELS/BCC_CENS_LISA06_MIG/LogisticRegression/AGESEX/" + str(clf) + hpstring + modelstring + ".sav"
pickle.dump(clf, open(filename, 'wb'))
    

print('---------------------------------')
print(str(clf))
print('-----------------------------------')
print('ACC', acc)
print('Precision - Recall', precision, recall)
print('ROC', fpr, tpr)
print('F1_score', f1)
print('---------------------------------')
print("auc=", auc)
print('---------------------------------')














