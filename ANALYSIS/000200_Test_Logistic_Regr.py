import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_auc_score
import numpy as  np


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Read MASTER_TMP file
print("Reading _000110_FRAME_MASTER...")
MASTER = pd.read_pickle("_000110_FRAME_MASTER.pickle").copy()
print("Finished Reading _000110_FRAME_MASTER")

ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX = pd.read_pickle("_000070_ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX.pickle").copy()
# ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_LIST = list(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX)
ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st = list(set(pd.Series(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX).str.slice(start=0, stop=3)))


ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX = pd.read_pickle("_000100_ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX.pickle").copy()
# ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX_LIST = list(ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX)
ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX_3st = list(set(pd.Series(ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX).str.slice(start=0, stop=3)))


# First CAN: "CAN_ICD10FULL_C240"
# LAST SV : "SV_ICD103st_Z12" 3864
index_CAN_OV_SV_start = list(MASTER).index("CAN_ICD10FULL_C240")
index_CAN_OV_SV_end   = list(MASTER).index("SV_ICD103st_Z12")
all_vars_CAN_OV_SV = list(MASTER)[index_CAN_OV_SV_start:index_CAN_OV_SV_end]

Explanatory_Var1 = ["Kon", "AGE_INDEX"]
Explanatory_Var2 = ["Kon", "AGE_INDEX", "CAN_ICD10FULL_C240"]
Explanatory_Var3 = ["Kon", "AGE_INDEX"] + all_vars_CAN_OV_SV

TRAIN = MASTER[MASTER["GROUP_TRAINVALTEST"]=="train"]
TRAIN_1_100 = MASTER[MASTER["GROUP_TRAINVALTEST_100"]=="train"]
VAL =   MASTER[MASTER["GROUP_TRAINVALTEST"]=="val"]

#------------------------------------------------------------------------

X_train1, y_train = TRAIN[Explanatory_Var1], \
                    (TRAIN[["MM_BETWEEN_INDEX_AND_ENDDATE"]]).values.reshape(-1, 1)
X_val1, y_val     = VAL[Explanatory_Var1], \
                    (VAL[["MM_BETWEEN_INDEX_AND_ENDDATE"]]).values.reshape(-1, 1)

clf1 = LogisticRegression(random_state=0).fit(X_train1, y_train)

y_val_pred1 = (clf1.predict(X_val1)).reshape(-1, 1)
y_val_pred_prob1 = clf1.predict_proba(X_val1)[:, 1].reshape(-1, 1)
#y_val_pred_prob1 = (clf1.predict_proba(X_val1)).reshape(-1, 1)
#score1 = clf1.score(y_val_pred, y_val.values.reshape(-1, 1))
acc_score1 = accuracy_score(y_val, y_val_pred1) # not a realistic measure (most are non cases)
auc1 = roc_auc_score(y_val, y_val_pred_prob1)

#------------------------------------------------------------------------

X_train2, y_train = TRAIN[Explanatory_Var2], \
                    (TRAIN[["MM_BETWEEN_INDEX_AND_ENDDATE"]]).values.reshape(-1, 1)
X_val2, y_val     = VAL[Explanatory_Var2], \
                    (VAL[["MM_BETWEEN_INDEX_AND_ENDDATE"]]).values.reshape(-1, 1)

clf2 = LogisticRegression(random_state=0).fit(X_train2, y_train)

y_val_pred2 = (clf2.predict(X_val2)).reshape(-1, 1)
y_val_pred_prob2 = clf2.predict_proba(X_val2)[:, 1].reshape(-1, 1)

#acc_score1 = accuracy_score(y_val, y_val_pred1) # not a realistic measure (most are non cases)
auc2 = roc_auc_score(y_val, y_val_pred_prob2)

#------------------------------------------------------------------------

print("Running clf3...")

X_train3, y_train = TRAIN_1_100[Explanatory_Var3], \
                    (TRAIN_1_100[["MM_BETWEEN_INDEX_AND_ENDDATE"]]).values.reshape(-1, 1)
X_val3, y_val     = VAL[Explanatory_Var3], \
                    (VAL[["MM_BETWEEN_INDEX_AND_ENDDATE"]]).values.reshape(-1, 1)

print("Running LogisticRegression(random_state=0).fit(X_train3, y_train)")
clf3 = LogisticRegression(random_state=0).fit(X_train3, y_train)

print("Validating...")
y_val_pred3 = (clf3.predict(X_val3)).reshape(-1, 1)
y_val_pred_prob3 = clf3.predict_proba(X_val3)[:, 1].reshape(-1, 1)

print("Computing AUC...")
#acc_score1 = accuracy_score(y_val, y_val_pred1) # not a realistic measure (most are non cases)
auc3 = roc_auc_score(y_val, y_val_pred_prob3)


print("Finished")





