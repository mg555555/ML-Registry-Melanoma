import _Globalconstants as GB
import time
import pandas as pd
import datetime

# Read MASTER_TMP file
MASTER_TMP = pd.read_pickle("_000020_FRAME_MASTER_TMP.pickle").copy()

# Create "final" MASTER frame with exkluded patients that are:
# 1. Over 18 at START obs period (not necessary -
#    all are that from the beginning (SCB, SOS))
# 2. Alive at GB.EXPLANATORY_DATA_END_DATE
# 3. No mig event in obs period, and lst before Inv and first after Utv
# 4. No invalid MIGR, DOD or CAN dates (DATE_OUTCOME_INACCURATE)
# 5. No contradiction between 190 and C43 (already fixed in 000010_CREATE_MASTER_FRAME
# 6. Does not seem necessary to exclude >=95 years for example
#    for matching later. Do that later if necessary.

MASTER = MASTER_TMP[(~MASTER_TMP["DEAD_BEFORE_INDEX"]) &
                    (~MASTER_TMP["PID_MIGR_2005_2019"]) &
                    ((MASTER_TMP["SISTA_POSTTYP_FORE_2005"].isna()) |
                     (MASTER_TMP["SISTA_POSTTYP_FORE_2005"]=="Inv")) &
                    ((MASTER_TMP["FORSTA_POSTTYP_EFTER_2019"].isna()) |
                     (MASTER_TMP["FORSTA_POSTTYP_EFTER_2019"]=="Utv")) &
                     (MASTER_TMP["DATE_OUTCOME_INACCURATE"].isna()) &
                    ].copy()

del MASTER_TMP

# Några konsistens tester:
# GB.table_nan(MASTER_TMP["SISTA_POSTTYP_FORE_2005"], MASTER_TMP["SISTA_DATUM_FORE_2015"].isna())
# GB.table_nan(MASTER_TMP["FORSTA_POSTTYP_EFTER_2019"], MASTER_TMP["FORSTA_DATUM_EFTER_2019"].isna())

MASTER.to_pickle("_000030_FRAME_MASTER.pickle")

print("Finished")




