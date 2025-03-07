import _Globalconstants as GB
import time
import pandas as pd
import datetime
import sys
# import random
import numpy as np
import pickle

from scipy.stats import norm


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

# load the model from disk
print("Loading model from disk...")

filename="/RAWDATA/TRAINEDMODELS/BCC_CENS_LISA06_MIG/LogisticRegression/NOCODES/LogisticRegression_C_0.0100_penalty_l2_solver_liblinear_11086396_ONLY_NOCODES_0.6797.sav"

model = pickle.load(open(filename, 'rb'))

print("Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG.pickle")
MASTER = pd.read_pickle("/RAWDATA/_099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG.pickle").copy()
print("Finished Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG.pickle")


MASTER_test  = MASTER[MASTER["GROUP_TRAINVALTEST"]=="test"]

print("Size of MASTER_test: ", MASTER_test.shape)


#-----
# CAN
#-----
index_CAN_start = list(MASTER_test).index("CAN_ICD10FULL_C300")
index_CAN_end   = list(MASTER_test).index("CAN_ICD10FULL_C496")
all_vars_CAN    = list(MASTER_test)[index_CAN_start:(index_CAN_end+1)]

#-------
# OV+SV
#-------
index_OV_SV_start  = list(MASTER_test).index("OV_ICD103st_")
index_OV_SV_end    = list(MASTER_test).index("SV_ICD103st_S25")
all_vars_OV_SV = list(MASTER_test)[index_OV_SV_start:(index_OV_SV_end+1)]

#------
# LMED
#------
index_LMED_start = list(MASTER_test).index("LMED_ATC_D10AD01")
index_LMED_end   = list(MASTER_test).index("LMED_ATC_B01AC04")
all_vars_LMED    = list(MASTER_test)[index_LMED_start:(index_LMED_end+1)]

#------
# BCC
#------
index_BCC_start = list(MASTER_test).index("CAN_BCC_EXISTS")
index_BCC_end   = list(MASTER_test).index("CAN_BCC_T02511")
all_vars_BCC    = list(MASTER_test)[index_BCC_start:(index_BCC_end+1)]

# Drop all "codes"
MASTER_test = MASTER_test.drop(all_vars_CAN + all_vars_OV_SV + all_vars_LMED + all_vars_BCC, axis=1).copy()


MASTER_test  = MASTER_test.drop(["GROUP_TRAINVALTEST"], axis=1).copy()

MASTER_test   = MASTER_test.astype(int)

X_test = MASTER_test.loc[:, (MASTER_test.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE')]

y_test = MASTER_test.loc[:, MASTER_test.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']

y_test = y_test.values.ravel()




class Evaluation:
    def auc_ci(self, auc, n0, n1, alpha=0.05):
        q0 = auc / (2 - auc)
        q1 = 2 * auc * auc / (1 + auc)
        se = np.sqrt((auc * (1 - auc) + (n0 - 1) * (q0 - auc * auc) + (n1 - 1) * (q1 - auc * auc)) / (n0 * n1))
        z = norm.ppf(1 - alpha / 2)
        ci = z * se
        return (auc - ci, auc + ci)


    def predict_and_evaluate(self, X_test, y_test, estimator, label):


        pp = estimator.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, pp)

        if isinstance(y_test, pd.DataFrame):
            y_test = y_test.values

        n_y_test_0 = np.sum(y_test == y_test.min())
        n_y_test_1 = np.sum(y_test == y_test.max())
        ci_0, ci_1 = self.auc_ci(auc, n_y_test_0, n_y_test_1, alpha=0.05)

        results = {'auc_' + label: [auc], 'auc_l_' + label: [ci_0], 'auc_u_' + label: [ci_1]}

        return results

eval1 = Evaluation()
print(eval1.predict_and_evaluate(X_test=X_test, y_test=y_test, estimator=model, label=str(model)))



def save_latent_outputs(X_test, y_test, estimator, filename):
    pp = estimator.predict_proba(X_test)[:, 1]
    df_output = pd.DataFrame(data={'Actual': y_test, 'Predicted_Probability': pp})
    df_output.to_csv(filename, index=False)
    print(f"Latent outputs saved to {filename}")


output_filename = "/workspace/public-ml-registry-mm/ANALYSIS/RESULTS/LogReg/LogisticRegression_C_0.0100_penalty_l2_solver_liblinear_11086396_ONLY_NOCODES_0.6797.csv"
save_latent_outputs(X_test, y_test, model, output_filename)



