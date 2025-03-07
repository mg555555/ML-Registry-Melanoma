import _Globalconstants as GB
import time
import pandas as pd
import datetime
import sys
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


print("Loading model from disk...")

filename="/RAWDATA/TRAINEDMODELS/BCC_CENS_LISA06_MIG/RandForest/AGESEX/RandomForestClassifier(max_depth=10, n_estimators=1000, verbose=True)_max_depth_10_n_est_1000_11086396_ONLY_AGESEX_0.6447.sav"


model = pickle.load(open(filename, 'rb'))

print("Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG.pickle")
MASTER = pd.read_pickle("/RAWDATA/_099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG.pickle").copy()
print("Finished Reading _099999_FRAME_MASTER_CENSORED_LISA_2006_CAN_BCC_MIG.pickle")


MASTER_test  = MASTER[MASTER["GROUP_TRAINVALTEST"]=="test"]

print("Size of MASTER_test: ", MASTER_test.shape)


MASTER_test = MASTER_test[["MM_BETWEEN_INDEX_AND_ENDDATE", "Kon_equals_1", "AGE_INDEX_5_(-inf, 35.0]", "AGE_INDEX_5_(35.0, 40.0]", "AGE_INDEX_5_(40.0, 45.0]", "AGE_INDEX_5_(45.0, 50.0]", "AGE_INDEX_5_(50.0, 55.0]", "AGE_INDEX_5_(55.0, 60.0]", "AGE_INDEX_5_(60.0, 65.0]", "AGE_INDEX_5_(65.0, 70.0]", "AGE_INDEX_5_(70.0, 75.0]", "AGE_INDEX_5_(75.0, 80.0]", "AGE_INDEX_5_(80.0, 85.0]", "AGE_INDEX_5_(85.0, inf]"]].copy()


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


output_filename = "/workspace/public-ml-registry-mm/ANALYSIS/RESULTS/RandForest/RandomForestClassifier_max_depth_10_n_est_1000_11086396_ONLY_AGESEX_0.6447.csv"
save_latent_outputs(X_test, y_test, model, output_filename)



