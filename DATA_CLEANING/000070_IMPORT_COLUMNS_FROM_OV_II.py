
import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Read MASTER_TMP file
MASTER = pd.read_pickle("_000050_FRAME_MASTER_True_False.pickle").copy()
OV     = pd.read_pickle("_000060_OV.pickle").copy()

print("OV read")

# Check those that are 8 length but still NA
#print("Check dates that are NA and length 8")
#x=OV.loc[(OV["INDATUMA_date"].isna()) & (OV["INDATUMA_len"]==8), "INDATUMA"]
#print("Finished")


# Note! We must include empty ICD10 as "empty" (still a visit)
# DIAGNOS column is not "", but NA (~18.8 mil / 167 mil)
print("Create OV_MASTER_BEFORE_INDEX")
OV_MASTER_BEFORE_INDEX = OV[(OV["LopNr"].isin(MASTER["LopNr"])) &
                              (~OV["INDATUMA_date"].isna()) &
                              (OV["INDATUMA_date"]>=GB.EXPLANATORY_DATA_START_DATE) &
                              (OV["INDATUMA_date"]<=GB.EXPLANATORY_DATA_END_DATE)].copy()

MASTER["OV_BEFORE_INDEX"]   = MASTER["LopNr"].isin(OV_MASTER_BEFORE_INDEX["LopNr"]).copy()
print("Finished 1")

#--------------------------------------------------------------------------

# använd split för att separera varje DIAGNOS till komponenter (separerade av " ")
# Start with adding the empty DIAGNOS string manually (a list)
# Later make into a set (each step)
ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX = [""]

OV_MASTER_BEFORE_INDEX_NONA = OV_MASTER_BEFORE_INDEX[~OV_MASTER_BEFORE_INDEX["DIAGNOS"].isna()]

# Note! We must save OV_MASTER_BEFORE_INDEX (with NAs so that NA DIAGNOSES gets registered
print("Saving...")
OV_MASTER_BEFORE_INDEX.to_pickle("_000070_OV_MASTER_BEFORE_INDEX.pickle")
print("Finished saving")

iter=0
LEN=OV_MASTER_BEFORE_INDEX_NONA.shape[0]

DIAGNOS_SERIES = OV_MASTER_BEFORE_INDEX_NONA["DIAGNOS"]

# Go through all rows in OV_MASTER_BEFORE_INDEX_NONA
for i in range(0, LEN):
    iter=iter+1
    if (iter % 10000) == 0:
        print("Progress: ", 100 * (iter / LEN), "%")
    ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX = set(list(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX) + (DIAGNOS_SERIES.iloc[i].split(" ")))

print("Finished 2")

print("Saving...")

(pd.Series(list(ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX))).to_pickle("_000070_ALLA_DIAGNOSER_OV_MASTER_BEFORE_INDEX.pickle")

print("Finished saving")

#---------------------- STOPS HERE, CONTINUE with later file ------------------------------



