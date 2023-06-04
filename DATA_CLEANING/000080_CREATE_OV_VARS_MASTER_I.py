
import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math
import scipy.stats as stat

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Note! ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX is a list
print("Reading ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX...")
ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX = pd.read_pickle("_000070_ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX.pickle").copy()
print("Finished Reading ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX")


print("Reading OV_MASTER_BEFORE_INDEX...")
OV_MASTER_BEFORE_INDEX = pd.read_pickle("_000070_OV_MASTER_BEFORE_INDEX.pickle").copy()
print("Finished reading OV_MASTER_BEFORE_INDEX")


print("Reading MASTER...")
MASTER = pd.read_pickle("_000050_FRAME_MASTER_True_False.pickle").copy()
print("Finished reading MASTER")


#---------------------------------------------
# Create all unique ICD10 first three letters
#---------------------------------------------
ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st = list(set(pd.Series(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX).str.slice(start=0, stop=3)))

ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st_NAMES = [""]*len(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st)
for i in range(len(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st)):
    ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st_NAMES[i] = "OV_ICD103st_"+ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st[i]

#--------------------------------------------------------------------------

OV_MASTER_BEFORE_INDEX_2COLUMNS = OV_MASTER_BEFORE_INDEX.loc[:, ["LopNr", "DIAGNOS"]].copy()
MASTER_LopNr = MASTER[["LopNr"]].copy()

# Use pandas.Series.str.contains
# Create columns in MASTER for different diagnoses
# MASTER (ever-never innan index) Full length
# The empty string in DIAGNOS must be taken sepoarately
iter=0
LEN1=len(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st)
for icdcode in ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX_3st:
    iter+=1
    MASTER_TMP = MASTER_LopNr.copy()
    if (iter % 1) == 0:
        print("Progress: ", 100 * (iter / LEN1), "%")
    # Only non-empty codes apply to below
    if icdcode!="":
        newcolname = "OV_ICD103st_"+icdcode
        MASTER_TMP[newcolname] = False
        # Check which individuals (a set) is
        # in OV_MASTER_BEFORE_INDEX_2COLUMNS have an icdcode
        OV_TMP_ICDCODE = OV_MASTER_BEFORE_INDEX_2COLUMNS[OV_MASTER_BEFORE_INDEX_2COLUMNS["DIAGNOS"].str.contains(icdcode, na=False)].copy()
        if OV_TMP_ICDCODE.shape[0]>0:
            MASTER_TMP.loc[MASTER_TMP["LopNr"].isin(OV_TMP_ICDCODE["LopNr"]), newcolname] = True
        MASTER[newcolname] = MASTER_TMP[newcolname].copy()
    if icdcode=="":
        newcolname = "OV_ICD103st_"+icdcode
        MASTER_TMP[newcolname] = False
        # Check which individuals (a set) is
        # in OV_MASTER_BEFORE_INDEX_2COLUMNS have an icdcode
        OV_TMP_ICDCODE = OV_MASTER_BEFORE_INDEX_2COLUMNS[OV_MASTER_BEFORE_INDEX_2COLUMNS["DIAGNOS"].isna()].copy()
        if OV_TMP_ICDCODE.shape[0]>0:
            MASTER_TMP.loc[MASTER_TMP["LopNr"].isin(OV_TMP_ICDCODE["LopNr"]), newcolname] = True
        MASTER[newcolname] = MASTER_TMP[newcolname].copy()


print("Saving _000080_FRAME_MASTER.pickle")
MASTER.to_pickle("_000080_FRAME_MASTER.pickle")
print("Finished saving _000080_FRAME_MASTER.pickle")


# Feltesta några koder:
# EMPTY=OV_MASTER_BEFORE_INDEX[OV_MASTER_BEFORE_INDEX["DIAGNOS"].isna()]
# C43=OV_MASTER_BEFORE_INDEX[OV_MASTER_BEFORE_INDEX["DIAGNOS"].str.contains("C43")]
# L40=OV_MASTER_BEFORE_INDEX[OV_MASTER_BEFORE_INDEX["DIAGNOS"].str.contains("L40")]
# J04=OV_MASTER_BEFORE_INDEX[OV_MASTER_BEFORE_INDEX["DIAGNOS"].str.contains("J04")]
# J45=OV_MASTER_BEFORE_INDEX[OV_MASTER_BEFORE_INDEX["DIAGNOS"].str.contains("J45")]
# J06=OV_MASTER_BEFORE_INDEX[OV_MASTER_BEFORE_INDEX["DIAGNOS"].str.contains("J06")]

# GB.table_nan(MASTER["OV_ICD103st_"])
# GB.table_nan(MASTER["OV_ICD103st_C43"])
# GB.table_nan(MASTER["OV_ICD103st_L40"])
# GB.table_nan(MASTER["OV_ICD103st_J04"])
# GB.table_nan(MASTER["OV_ICD103st_J45"])
# GB.table_nan(MASTER["OV_ICD103st_J06"])

# len(set(EMPTY["LopNr"]))
# len(set(C43["LopNr"]))
# len(set(L40["LopNr"]))
# len(set(J04["LopNr"]))
# len(set(J45["LopNr"]))
# len(set(J06["LopNr"]))

#ALL_OCCURING_CAN_ICDO10.to_pickle("_000050_ALL_OCCURING_CAN_ICDO10.pickle")
#ALL_OCCURING_CAN_ICDO10_NAMES.to_pickle("_000050_ALL_OCCURING_CAN_ICDO10_NAMES.pickle")

print("Finished")


