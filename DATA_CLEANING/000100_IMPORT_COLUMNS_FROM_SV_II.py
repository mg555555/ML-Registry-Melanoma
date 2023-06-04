
import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Read MASTER_TMP file
MASTER = pd.read_pickle("_000080_FRAME_MASTER.pickle").copy()
SV     = pd.read_pickle("_000090_SV.pickle").copy()

print("SV read")

# Check those that are 8 length but still NA
#print("Check dates that are NA and length 8")
#x=SV.loc[(SV["INDATUMA_date"].isna()) & (SV["INDATUMA_len"]==8), "INDATUMA"]
#print("Finished")


# Note! We must include empty ICD10 as "empty" (still a visit)
# DIAGNOS column is not "", but NA
print("Create SV_MASTER_BEFORE_INDEX...")
SV_MASTER_BEFORE_INDEX = SV[(SV["LopNr"].isin(MASTER["LopNr"])) &
                              (~SV["INDATUMA_date"].isna()) &
                              (SV["INDATUMA_date"]>=GB.EXPLANATORY_DATA_START_DATE) &
                              (SV["INDATUMA_date"]<=GB.EXPLANATORY_DATA_END_DATE)].copy()

MASTER["SV_BEFORE_INDEX"]   = MASTER["LopNr"].isin(SV_MASTER_BEFORE_INDEX["LopNr"]).copy()
print("Finished Create SV_MASTER_BEFORE_INDEX")

#--------------------------------------------------------------------------

# använd split för att separera varje DIAGNOS till komponenter (separerade av " ")
# Start with adding the empty DIAGNOS string manually (a list)
# Later make into a set (each step)
ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX = [""]

SV_MASTER_BEFORE_INDEX_NONA = SV_MASTER_BEFORE_INDEX[~SV_MASTER_BEFORE_INDEX["DIAGNOS"].isna()]

# Note! We must save SV_MASTER_BEFORE_INDEX (with NAs so that NA DIAGNOSES gets registered
print("Saving _000100_SV_MASTER_BEFORE_INDEX...")
SV_MASTER_BEFORE_INDEX.to_pickle("_000100_SV_MASTER_BEFORE_INDEX.pickle")
print("Finished saving _000100_SV_MASTER_BEFORE_INDEX")

iter=0
LEN=SV_MASTER_BEFORE_INDEX_NONA.shape[0]

DIAGNOS_SERIES = SV_MASTER_BEFORE_INDEX_NONA["DIAGNOS"]

# Go through all rows in SV_MASTER_BEFORE_INDEX_NONA
for i in range(0, LEN):
    iter=iter+1
    if (iter % 10000) == 0:
        print("Progress: ", 100 * (iter / LEN), "%")
    ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX = set(list(ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX) + (DIAGNOS_SERIES.iloc[i].split(" ")))

print("Finished 2")

print("Saving...")

(pd.Series(list(ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX))).to_pickle("_000100_ALLA_DIAGNOSER_SV_MASTER_BEFORE_INDEX.pickle")

print("Finished saving")

#---------------------- STOPS HERE, CONTINUE with later file ------------------------------



