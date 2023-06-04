import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Read MASTER_TMP file
MASTER = pd.read_pickle("_000040_FRAME_MASTER.pickle").copy()
CAN = pd.read_pickle("_000010_FRAME_CAN.pickle").copy()

# Create ages in intervals:
bns5  = [-float("inf")] + list(range(35, 90, 5))  + [float("inf")]
bns10 = [-float("inf")] + list(range(40, 90, 10)) + [float("inf")]

MASTER["AGE_INDEX_5"]  = pd.cut(MASTER["AGE_INDEX"], bins=bns5).copy()
MASTER["AGE_INDEX_10"] = pd.cut(MASTER["AGE_INDEX"], bins=bns10).copy()

# Note! ICDO10 has NA's but ICDO10_nona has "" instead!
CAN_MASTER_BEFORE_INDEX = CAN[(CAN["LopNr"].isin(MASTER["LopNr"])) &
                              (CAN["DIADAT_date"]>=GB.EXPLANATORY_DATA_START_DATE) &
                              (CAN["DIADAT_date"]<=GB.EXPLANATORY_DATA_END_DATE) &
                              (~CAN["ICDO10"].isna())].copy()

MASTER["CAN_BEFORE_INDEX"]   = MASTER["LopNr"].isin(CAN_MASTER_BEFORE_INDEX["LopNr"])

#--------------------------------------------------------------------------

# använd Series.str.slice(start=None, stop=None, step=None) för substrings
ALL_OCCURING_CAN_ICDO10 = list(set(CAN_MASTER_BEFORE_INDEX["ICDO10"]))
ALL_OCCURING_CAN_ICDO10_NAMES = [""]*len(ALL_OCCURING_CAN_ICDO10)
for i in range(len(ALL_OCCURING_CAN_ICDO10)):
    ALL_OCCURING_CAN_ICDO10_NAMES[i] = "CAN_ICD10FULL_"+ALL_OCCURING_CAN_ICDO10[i]

#--------------------------------------------------------------------------

ALL_OCCURING_CAN_ICDO10_3first = list(set(pd.Series(ALL_OCCURING_CAN_ICDO10).str.slice(start=0, stop=3)))
ALL_OCCURING_CAN_ICDO10_3first_NAMES = [""]*len(ALL_OCCURING_CAN_ICDO10_3first)
for i in range(len(ALL_OCCURING_CAN_ICDO10_3first)):
    ALL_OCCURING_CAN_ICDO10_3first_NAMES[i] = "CAN_ICD103st_"+ALL_OCCURING_CAN_ICDO10_3first[i]

#--------------------------------------------------------------------------

CAN_MASTER_BEFORE_INDEX_2COLUMNS = CAN_MASTER_BEFORE_INDEX.loc[:, ["LopNr", "ICDO10"]].copy()

# Create columns in MASTER for different diagnoses
# MASTER (ever-never innan index) Full length
iter=0
LEN1=len(ALL_OCCURING_CAN_ICDO10)
for icdcode in ALL_OCCURING_CAN_ICDO10:
    iter+=1
    if (iter % 1) == 0:
        print("Progress: ", 100 * (iter / LEN1), "%")
    newcolname = "CAN_ICD10FULL_"+icdcode
    MASTER[newcolname] = False
    # Check which individuals (a set) is
    # in CAN_MASTER_BEFORE_INDEX have an icdcode
    CAN_TMP_ICDCODE = CAN_MASTER_BEFORE_INDEX_2COLUMNS[CAN_MASTER_BEFORE_INDEX_2COLUMNS["ICDO10"] == icdcode].copy()
    if CAN_TMP_ICDCODE.shape[0]>0:
        MASTER.loc[MASTER["LopNr"].isin(CAN_TMP_ICDCODE["LopNr"]), newcolname] = True

MASTER.to_pickle("_000050_FRAME_MASTER_True_False.pickle")

ALL_OCCURING_CAN_ICDO10_SERIES = pd.Series(ALL_OCCURING_CAN_ICDO10)
ALL_OCCURING_CAN_ICDO10_SERIES.to_pickle("_000050_ALL_OCCURING_CAN_ICDO10_SERIES.pickle")
ALL_OCCURING_CAN_ICDO10_NAMES_SERIES = pd.Series(ALL_OCCURING_CAN_ICDO10_NAMES)
ALL_OCCURING_CAN_ICDO10_NAMES_SERIES.to_pickle("_000050_ALL_OCCURING_CAN_ICDO10_NAMES_SERIES.pickle")

print("Finished")





