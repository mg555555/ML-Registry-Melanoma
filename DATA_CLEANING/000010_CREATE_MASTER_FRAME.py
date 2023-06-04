import _Globalconstants as GB
import pandas as pd
import numpy as np
import time
import datetime
#import copy

print("All raw data registers: \n", GB.RAWDATA_DICT.keys(), sep="")
print()

# Read PID and birth file
# 1=Man, 2=Woman
PID_FOD = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["PID_FOD"], GB.DTYPES_DICT["PID_FOD"])
# Death dates
PID_DOD = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["PID_DOD"], GB.DTYPES_DICT["PID_DOD"])
# Migrations
PID_MIGR = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["PID_MIGR"], GB.DTYPES_DICT["PID_MIGR"])
# Read cancer registry (not BCC)
CAN = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["CAN"], GB.DTYPES_DICT["CAN"])

print(PID_FOD.info(), "\n")
print(PID_DOD.info(), "\n")
print(PID_MIGR.info(), "\n")
print(CAN.info(), "\n")

# Create temporary master file (one row per person)
# All individuals are guaranteed to be
# at least 18 at 20050701 (see R:\STAT\001_ML_REG\Styrdokument\Etikansökan\Maskininlärningsbaserade
# _prediktionsmodeller_för_hudcancer_tränade_på_registerdata_/
# 01-ansokan-om-etikprovning-beskrivning-av-forskningsprojektet)
MASTER_TMP = PID_FOD[["LopNr", "Fodelsear", "Kon", "FodLand_EU28", "UtlSvBakg"]].copy()
# Age (rounded down) at index date. If born in 2013, age=(2015-1)-2013 = 1 year
MASTER_TMP["AGE_INDEX"] = (GB.INDEX_DATE.isocalendar().year-1) - MASTER_TMP["Fodelsear"]


######################################################################################
# Add columns that indicate if row starts with ICD7 "190" and ICD10 "C43"
# Note! NaN is interpretated as boolean True!!
# Remove all NaN first and replace with "" for example
######################################################################################
CAN["ICDO10_nona"] = CAN["ICDO10"].fillna("").copy()
CAN["ICD7_nona"] = CAN["ICD7"].fillna("").copy()

# Either startswith xxx or not
CAN["ICD7_STARTSWITH_190"]  = CAN["ICD7_nona"].str.startswith(GB.MM_ICD7_STARTSWITH).astype(dtype=bool).copy()
CAN["ICD10_STARTSWITH_C43"] = CAN["ICDO10_nona"].str.startswith(GB.MM_ICD10_STARTSWITH).astype(dtype=bool).copy()

# Three options: empty, startswith xxx or nonempty that does not start with xxx
CAN["ICD7_STARTSWITH_190_3st"] = pd.NA
CAN.loc[CAN["ICD7_nona"].str.startswith(GB.MM_ICD7_STARTSWITH), "ICD7_STARTSWITH_190_3st"] = "Starts with 190"
CAN.loc[~CAN["ICD7_nona"].str.startswith(GB.MM_ICD7_STARTSWITH), "ICD7_STARTSWITH_190_3st"] = "Does not start with 190"
CAN.loc[CAN["ICD7_nona"]=="", "ICD7_STARTSWITH_190_3st"] = "Empty"

CAN["ICD10_STARTSWITH_C43_3st"] = pd.NA
CAN.loc[CAN["ICDO10_nona"].str.startswith(GB.MM_ICD10_STARTSWITH), "ICD10_STARTSWITH_C43_3st"] = "Starts with C43"
CAN.loc[~CAN["ICDO10_nona"].str.startswith(GB.MM_ICD10_STARTSWITH), "ICD10_STARTSWITH_C43_3st"] = "Does not start with C43"
CAN.loc[CAN["ICDO10_nona"]=="", "ICD10_STARTSWITH_C43_3st"] = "Empty"

# Not sure if we should exclude those _individuals_ with contradictory ICD7=MM and ICD10=MM?
# Exclude all indivviduals with contradictory / incomplete OUTCOME, but not for
# the input data, X, before index date

# Create column for the lengths of DIADAT string dates:
CAN["DIADAT_len"] = CAN["DIADAT"].str.len().copy()

# Check the format of those with 6:
print("Dates with length 6: \n")
print(CAN["DIADAT"][CAN["DIADAT_len"]==6])
# They have YYYYMM


######################################################
#                    Create dates
######################################################
# Create dates in CAN:
CAN["DIADAT_date"] = CAN["DIADAT"].copy()
CAN["DIADAT_date"] = CAN["DIADAT_date"].apply(lambda x: GB.convert_date_YYYYMMDD_YYYYMM15(x))

# PID_DOD datum
PID_DOD["DodDatum_date"] = PID_DOD["DodDatum"].copy()
PID_DOD["DodDatum_date"] = PID_DOD["DodDatum_date"].apply(lambda x: GB.convert_date_YYYYMMDD_YYYYMM15(x))

# PID_MIGR
PID_MIGR["Datum_date"] = PID_MIGR["Datum"].copy()
PID_MIGR["Datum_date"] = PID_MIGR["Datum_date"].apply(lambda x: GB.convert_date_YYYYMMDD_YYYYMM15(x))


################################################################################
# Mark all patients in MASTER that have NA-dates in CAN, PID_DOD or PID_MIGR
# All individuals who are not NA in MASTER_TMP["DATE_OUTCOME_INACCURATE"]
# should be excluded
################################################################################
MASTER_TMP["DATE_OUTCOME_INACCURATE"] = pd.NA

canNA_tmp = CAN.loc[ CAN["DIADAT_date"].isna(), "LopNr" ].copy()
MASTER_TMP.loc[ MASTER_TMP["LopNr"].isin(canNA_tmp), "DATE_OUTCOME_INACCURATE" ] = "Invalid CAN date"

dodNA_tmp = PID_DOD.loc[ PID_DOD["DodDatum_date"].isna(), "Lopnr" ].copy()
MASTER_TMP.loc[ MASTER_TMP["LopNr"].isin(dodNA_tmp), "DATE_OUTCOME_INACCURATE" ] = "Invalid PID_DOD date"

migNA_tmp = PID_MIGR.loc[ PID_MIGR["Datum_date"].isna(), "LopNr" ].copy()
MASTER_TMP.loc[ MASTER_TMP["LopNr"].isin(migNA_tmp), "DATE_OUTCOME_INACCURATE" ] = "Invalid PID_MIGR date"


#################################################################################
# Included individuals must be alive at EXPLANATORY_DATA_END_DATE (2014-12-31)
# Mark those who are dead at EXPLANATORY_DATA_END_DATE,
# i.e. death date<=EXPLANATORY_DATA_END_DATE
# OBS! Här räknas pd.NA<=GB.EXPLANATORY_DATA_END_DATE som falsk!
#################################################################################
MASTER_TMP["DEAD_BEFORE_INDEX"] = False

dead_before_index = PID_DOD.loc[ PID_DOD["DodDatum_date"]<=GB.EXPLANATORY_DATA_END_DATE, "Lopnr" ].copy()
MASTER_TMP.loc[ MASTER_TMP["LopNr"].isin(dead_before_index), "DEAD_BEFORE_INDEX" ] = True


#################################################################################
# Kolla om datum i MIGR är
# 1) före EXPLANATORY_DATA_START_DATE
# 2) i [EXPLANATORY_DATA_START_DATE, END_DATE]
# 3) Efter END_DATE
# De som är NA är här False!
#################################################################################
PID_MIGR["Datum_FORE_2005"] = False
PID_MIGR.loc[ PID_MIGR["Datum_date"]<GB.EXPLANATORY_DATA_START_DATE, "Datum_FORE_2005" ] = True

PID_MIGR["Datum_i_2005_2019"] = False
PID_MIGR.loc[ (PID_MIGR["Datum_date"]>=GB.EXPLANATORY_DATA_START_DATE) &
              (PID_MIGR["Datum_date"]<=GB.END_DATE), "Datum_i_2005_2019" ] = True

PID_MIGR["Datum_EFTER_2019"] = False
PID_MIGR.loc[ PID_MIGR["Datum_date"]>GB.END_DATE, "Datum_EFTER_2019" ] = True

# Gör 3 subsets:
PID_MIGR_FORE_2005  = PID_MIGR[PID_MIGR["Datum_FORE_2005"]].copy()
PID_MIGR_EFTER_2019 = PID_MIGR[PID_MIGR["Datum_EFTER_2019"]].copy()
PID_MIGR_2005_2019  = PID_MIGR[PID_MIGR["Datum_i_2005_2019"]].copy()

MASTER_TMP["PID_MIGR_FORE_2005"] = False
MASTER_TMP.loc[MASTER_TMP["LopNr"].isin(PID_MIGR_FORE_2005["LopNr"]), "PID_MIGR_FORE_2005"] = True

MASTER_TMP["PID_MIGR_EFTER_2019"] = False
MASTER_TMP.loc[MASTER_TMP["LopNr"].isin(PID_MIGR_EFTER_2019["LopNr"]), "PID_MIGR_EFTER_2019"] = True

MASTER_TMP["PID_MIGR_2005_2019"] = False
MASTER_TMP.loc[MASTER_TMP["LopNr"].isin(PID_MIGR_2005_2019["LopNr"]), "PID_MIGR_2005_2019"] = True

# Dela in pat i 3 grupper: 1) inga mig events, 2) mig event i obs period, 3) ej mig event i obs period men före eller efter obs period
# Alla i 1) ska inkl, alla i 2) ska exkl, men 3) beror på
MASTER_TMP["INGEN_MIGR"] = False
MASTER_TMP.loc[~MASTER_TMP["PID_MIGR_FORE_2005"] & ~MASTER_TMP["PID_MIGR_EFTER_2019"] & ~MASTER_TMP["PID_MIGR_2005_2019"], "INGEN_MIGR"] = True

MASTER_TMP["EJ_MIG_OBS_MEN_FORE_ELLER_EFTER"] = False
MASTER_TMP.loc[~MASTER_TMP["PID_MIGR_2005_2019"] & (MASTER_TMP["PID_MIGR_FORE_2005"] | MASTER_TMP["PID_MIGR_EFTER_2019"]), "EJ_MIG_OBS_MEN_FORE_ELLER_EFTER"] = True


# Check dates:
print("Dates with length 6 (CAN): \n")
print(CAN.loc[CAN["DIADAT_len"]==6, "DIADAT_date"], "\n")
print("Dates with length 8 (CAN): \n")
print(CAN.loc[CAN["DIADAT_len"]==8, "DIADAT_date"], "\n")
print("Number if NA in dates CAN: \n")
print(pd.value_counts(CAN["DIADAT_date"].isna()))
print("Those dates that are NA in CAN:\n")
print(CAN.loc[CAN["DIADAT_date"].isna(), "DIADAT"], "\n")

print("Number if NA in dates PID_DOD: \n")
print(pd.value_counts(PID_DOD["DodDatum_date"].isna()))
print("Those dates that are NA PID_DOD:\n")
print(PID_DOD.loc[PID_DOD["DodDatum_date"].isna(), "DodDatum"], "\n")

print("Number if NA in dates PID_MIGR: \n")
print(pd.value_counts(PID_MIGR["Datum_date"].isna()))
print("Those dates that are NA PID_MIGR:\n")
print(PID_MIGR.loc[PID_MIGR["Datum_date"].isna(), "Datum"], "\n")

# If date in outcome/label period
# Note: NA>=some date is False!
CAN["BETWEEN_INDEX_AND_ENDDATE"] = ( (GB.END_DATE >= CAN["DIADAT_date"]) &
                                     (CAN["DIADAT_date"] >= GB.INDEX_DATE) )
CAN.loc[CAN["DIADAT_date"].isna(), "BETWEEN_INDEX_AND_ENDDATE"] = pd.NA

# Check consistency between ICD10 and ICD7:
print(GB.table_nan(CAN["ICD7_STARTSWITH_190"], CAN["ICD10_STARTSWITH_C43"]))
print(GB.table_nan(CAN["ICD7_STARTSWITH_190_3st"], CAN["ICD10_STARTSWITH_C43_3st"]))

# Only include those as MM cases that do not have contradictory ICD7 and ICD10 codes:

# Create subset of CAN with only MM codes:
# Note! Cases are only whose that have either
# (Empty, C43*), (190*, Empty) or (190*, C43*)
CAN_MM = CAN[ ((CAN["ICD7_STARTSWITH_190_3st"]=="Starts with 190") &
              (CAN["ICD10_STARTSWITH_C43_3st"]=="Starts with C43")) |
              ((CAN["ICD7_STARTSWITH_190_3st"]=="Starts with 190") &
              (CAN["ICD10_STARTSWITH_C43_3st"]=="Empty")) |
              ((CAN["ICD7_STARTSWITH_190_3st"]=="Empty") &
              (CAN["ICD10_STARTSWITH_C43_3st"]=="Starts with C43")) ].copy()

# Crosstab of ICD7 and ICD10 for MM dataframe:
res=GB.table_nan(CAN_MM["ICD7"], CAN_MM["ICDO10"])
print(res)
# res.to_clipboard()

# The subset of
CAN_MM_AFTER_INDEX = CAN_MM[CAN_MM["BETWEEN_INDEX_AND_ENDDATE"]].copy()
print("Number of unique individuals that are MM cases (not taking migration into account):", len(set(CAN_MM_AFTER_INDEX["LopNr"])))


####################################
# Create (preliminary) MM column
####################################
MASTER_TMP["MM_BETWEEN_INDEX_AND_ENDDATE"] = False
MASTER_TMP.loc[ MASTER_TMP["LopNr"].isin(CAN_MM_AFTER_INDEX["LopNr"]), "MM_BETWEEN_INDEX_AND_ENDDATE"] = True


MASTER_TMP_CASES = MASTER_TMP[MASTER_TMP["MM_BETWEEN_INDEX_AND_ENDDATE"]].copy()
MASTER_TMP_CONTROLS = MASTER_TMP[~MASTER_TMP["MM_BETWEEN_INDEX_AND_ENDDATE"]].copy()

# A small subsample
MASTER_TMP_CONTROLS_SMALLSAMPLE = MASTER_TMP_CONTROLS.sample(n=38879, random_state=GB.RANDOM_TEST_SEED)

del MASTER_TMP_CONTROLS

MASTER_TMP_SMALLSAMPLE = pd.concat([MASTER_TMP_CASES, MASTER_TMP_CONTROLS_SMALLSAMPLE])

# MASTER_TMP_SMALLSAMPLE.to_csv("MASTER_TMP_SMALLSAMPLE.txt", sep='\t', index=False)

#################################
# Spara ner filerna
#################################
MASTER_TMP.to_pickle("_000010_FRAME_MASTER_TMP.pickle")
PID_FOD.to_pickle("_000010_FRAME_PID_FOD.pickle")
PID_DOD.to_pickle("_000010_FRAME_PID_DOD.pickle")
PID_MIGR.to_pickle("_000010_FRAME_PID_MIGR.pickle")
CAN.to_pickle("_000010_FRAME_CAN.pickle")

#TEST = pd.read_pickle("MASTER_TMP.pickle").copy()

print("Finished")




