import _Globalconstants as GB
import time
import pandas as pd
import datetime

# Read files
MASTER_TMP = pd.read_pickle("_000010_FRAME_MASTER_TMP.pickle").copy()
PID_MIGR = pd.read_pickle("_000010_FRAME_PID_MIGR.pickle").copy()

# Kolla första MIG event efter OBS period och sista MIG event före OBS period
lopnr_EJ_MIG_OBS_MEN_FORE_ELLER_EFTER = MASTER_TMP.loc[MASTER_TMP["EJ_MIG_OBS_MEN_FORE_ELLER_EFTER"], "LopNr"].copy()

MASTER_TMP["SISTA_POSTTYP_FORE_2005"]   = pd.NA
MASTER_TMP["SISTA_DATUM_FORE_2015"]     = pd.NA
MASTER_TMP["FORSTA_POSTTYP_EFTER_2019"] = pd.NA
MASTER_TMP["FORSTA_DATUM_EFTER_2019"]   = pd.NA


LEN1 = len(lopnr_EJ_MIG_OBS_MEN_FORE_ELLER_EFTER)

start_time_forloop = time.time()

# Det tog 36 minuter för 10000 iterations
# Hela körningen tog 62.4 timmar
iter = 0
for lopnr in lopnr_EJ_MIG_OBS_MEN_FORE_ELLER_EFTER:
    iter += 1

    # --------------------------------------------------
    # Get all dates before OBS period that is not NA
    # --------------------------------------------------
    tmpdates = PID_MIGR.loc[(PID_MIGR["LopNr"]==lopnr) &
                            (PID_MIGR["Datum_date"]<GB.EXPLANATORY_DATA_START_DATE), "Datum_date"].copy()
    # Remove all NA before taking max/min
    dates_before_OBS_nona = tmpdates[~tmpdates.isna()].copy()
    if not(dates_before_OBS_nona.empty):
        # This is maxdate for this individual. Can only use on non-NAs
        maxdate_before = max(dates_before_OBS_nona)
        MASTER_TMP.loc[MASTER_TMP["LopNr"]==lopnr, "SISTA_DATUM_FORE_2015"] = maxdate_before
        # Note! .copy() not needed when taking first element of series!
        posttyp_before = PID_MIGR.loc[(PID_MIGR["LopNr"]==lopnr) &
                            (PID_MIGR["Datum_date"]==maxdate_before), "Posttyp"].iloc[0]
        MASTER_TMP.loc[MASTER_TMP["LopNr"]==lopnr, "SISTA_POSTTYP_FORE_2005"] = posttyp_before

        if (iter % 100) == 0:
            print("Progress: ", 100 * (iter / LEN1), "%")
            print(maxdate_before)
            print(posttyp_before)

    #--------------------------------------------------
    # Get all dates after OBS period that is not NA
    # --------------------------------------------------
    tmpdates = PID_MIGR.loc[(PID_MIGR["LopNr"]==lopnr) &
                            (PID_MIGR["Datum_date"]>GB.END_DATE), "Datum_date"].copy()
    # Remove all NA before taking max/min
    dates_after_OBS_nona = tmpdates[~tmpdates.isna()].copy()
    if not(dates_after_OBS_nona.empty):
        # This is mindate for this individual. Can only use on non-NAs
        # Note that min(series) is not a series, but an element of a series, just
        # like series.iloc[0] is not a series but the first element of the series
        mindate_after = min(dates_after_OBS_nona)
        MASTER_TMP.loc[MASTER_TMP["LopNr"]==lopnr, "FORSTA_DATUM_EFTER_2019"] = mindate_after
        # Note! .copy() not needed when taking first element of series!
        posttyp_after = PID_MIGR.loc[(PID_MIGR["LopNr"]==lopnr) &
                            (PID_MIGR["Datum_date"]==mindate_after), "Posttyp"].iloc[0]
        MASTER_TMP.loc[MASTER_TMP["LopNr"]==lopnr, "FORSTA_POSTTYP_EFTER_2019"] = posttyp_after

        if (iter % 100) == 0:
            print("Progress: ", 100 * (iter / LEN1), "%")
            print(mindate_after)
            print(posttyp_after)



end_time_forloop = time.time()

print("Tid att köra for loop:", (end_time_forloop-start_time_forloop)/60, " minutes")

MASTER_TMP.to_pickle("_000020_FRAME_MASTER_TMP.pickle")

