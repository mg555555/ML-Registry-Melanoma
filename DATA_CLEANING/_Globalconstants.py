import time
import pandas as pd
#import copy
import datetime

# Define global constants and functions etc

DATADIR_SCB = "R:/STAT/001_ML_REG/DATA_SCB_220308/DATA/"
DATADIR_SOC = "R:/STAT/001_ML_REG/DATA_SOC_220425/DATA/"

# Melanoma definition
# Mostly for old data we use ICD7
MM_ICD7_STARTSWITH = "190"
# For data newer than at least 2012 (Previous study) we can use ICD10
MM_ICD10_STARTSWITH = "C43"
# INDEX DATE
INDEX_DATE = datetime.date(2015, 1, 1)
# End of observation period after index
END_DATE = datetime.date(2019, 12, 31)
# Start date for all data before index date (<INDEX_DATE)
# Take 4 July 2005 because it is the beginning of a week
EXPLANATORY_DATA_START_DATE = datetime.date(2005, 7, 4)
EXPLANATORY_DATA_END_DATE = datetime.date(2014, 12, 31)
EXPLANATORY_DATA_CENSOR_DATE = datetime.date(2014, 6, 30)

# Some seeds
RANDOM_TEST_SEED = 499567590

SEED_TRAIN_VAL_TESTSET_CASES = 374783151
SEED_TRAIN_VAL_TESTSET_KONTS = 570471590

SEED_TRAIN_KONTS_10 = 504754161
SEED_VAL_KONTS_10   = 201694820
SEED_TEST_KONTS_10  = 825192557

SEED_TRAIN_KONTS_100 = 757892732
SEED_VAL_KONTS_100   = 307288460
SEED_TEST_KONTS_100  = 336588532


# Size of train val och testset för fallen
VAL_SIZE_CASES = 3000
TEST_SIZE_CASES = 3000
# TRAIN is the rest, do not need to specify

# Create dictionary for all raw data files
RAWDATA_DICT = {"PID_FOD":    DATADIR_SCB+"SCB_Leverans_RTB/SP_Lev_Fodelseuppg.txt",
                "PID_DOD":    DATADIR_SCB+"SCB_Leverans_RTB/SP_Lev_DodDatum.txt",
                "PID_MIGR":   DATADIR_SCB+"SCB_Leverans_RTB/SP_Lev_Migrationer.txt",
                "CAN":        DATADIR_SOC+"Ut_best_30720_2021/R_CAN_30720_2021.txt",
                "CAN_BCC":    DATADIR_SOC+"Ut_best_30720_2021/R_CAN_BC_30720_2021.txt",
                "DORS":       DATADIR_SOC+"Ut_best_30720_2021/R_DORS_30720_2021.txt",
                "OV":         DATADIR_SOC+"Ut_best_30720_2021/R_PAR_OV_30720_2021.txt",
                "SV":         DATADIR_SOC+"Ut_best_30720_2021/R_PAR_SV_30720_2021.txt",
                "HUDMELANOM": DATADIR_SOC+"Ut_best_30720_2021/SOS_HUDMELANOM_30720_2021.txt",
                "LMED_05_06": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_05_06_30720_2021.txt",
                "LMED_07_08": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_07_08_30720_2021.txt",
                "LMED_09_10": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_09_10_30720_2021.txt",
                "LMED_11_12": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_11_12_30720_2021.txt",
                "LMED_13_14": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_13_14_30720_2021.txt",
                "LMED_15_16": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_15_16_30720_2021.txt",
                "LMED_17_18": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_17_18_30720_2021.txt",
                "LMED_19_20": DATADIR_SOC+"Ut_best_30720_2021_lmed_05_14/R_LMED_19_20_30720_2021.txt"
                }

DTYPES_DICT = {"PID_FOD": {"UtlSvBakg": str},
               "PID_DOD": {"DodDatum": str},
               "PID_MIGR": {"Datum": str},
               "CAN": {"REGION": str,
                       "DIADAT": str,
                       "SJUKHUS": str,
                       "DIGR":str,
                       "OBD1": str,
                       "ICDO3": str,
                       "ICDO10": str,
                       "ICD9": str,
                       "ICD7": str,
                       "SIDA": str,
                       "SNOMED3": str,
                       "SNOMEDO10": str,
                       "BEN": str,
                       "PAD": str,
                       "dodsdat": str,
                       "dbgrund1": str},
               "OV": {"MVO": str,
                      "INDATUMA": str,
                      "HDIA": str,
                      "DIAGNOS": str},
               "SV": {"MVO": str,
                      "INDATUMA": str,
                      "HDIA": str,
                      "DIAGNOS": str},
               "LMED": {"ATC": str,
                        "EDATUM": str,
                        "VERKS": str}
               }

def READ_RAWFILE_TABSEP(filename):
    start_time = time.time()
    DF = pd.read_table(filename, delimiter='\t', encoding='ANSI')
    end_time = time.time()
    print(filename, " read in: ", "{:.1f}".format(end_time - start_time), " seconds")
    print()
    return DF

def READ_RAWFILE_DTYPES_TABSEP(filename, dtype):
    start_time = time.time()
    DF = pd.read_table(filename, delimiter='\t', encoding='ANSI', dtype=dtype)
    end_time = time.time()
    print(filename, " read in: ", "{:.1f}".format(end_time - start_time), " seconds")
    print()
    return DF



# Returns a crosstable of X and Y (panda series) with column/rows for NaN
# with label "label"
def table_nan(X, Y=None, label="NaN"):
    if Y is None:
        A = X.copy()
        A[pd.isna(A)] = label
        return pd.value_counts(A)
    else:
        A = X.copy()
        B = Y.copy()
        A[pd.isna(A)] = label
        B[pd.isna(B)] = label
        return pd.crosstab(A, B)


# Convert date strings
def convert_date_YYYYMMDD_YYYYMM15(x):
    if len(x)==8:
        try:
            return datetime.datetime.strptime(x, "%Y%m%d").date()
        except:
            try:
                # Try substring of first 6 chars (seems to work)
                return datetime.datetime.strptime(x[:6] + "15", "%Y%m%d").date()
            except:
                return pd.NA
    elif len(x)==6:
        try:
            return datetime.datetime.strptime(x+"15", "%Y%m%d").date()
        except:
            return pd.NA
    else:
        return pd.NA





