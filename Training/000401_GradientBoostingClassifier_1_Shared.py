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


print("Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle")
MASTER = pd.read_pickle("/RAWDATA/_099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle").copy()
print("Finished Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG_BALANCED_MM.pickle")
weight_cases = 5026682/32505



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



lr=0.1
n_est=1500
max_d=1

hpstring = "_lr_" + str("{:.2f}".format(lr)) + "_n_est_" + str("{:.0f}".format(n_est)) + "_max_d_" + str("{:.0f}".format(max_d)) + "_"

clf = GradientBoostingClassifier(verbose=True, learning_rate=lr, n_estimators=n_est, max_depth=max_d)
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
    
modelstring = str(ROWS) + "_ALLROWS_UPSAMPLED_ALLCols_" + ("{:.4f}".format(auc))

filename="/RAWDATA/TRAINEDMODELS/BCC_CENS_LISA06_MIG/GradBoost/" + str(clf) + hpstring + modelstring + ".sav"
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














