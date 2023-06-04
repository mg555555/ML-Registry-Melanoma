import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math
import scipy.stats as stat


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Read MASTER_TMP file
#MASTER = pd.read_pickle("_000050_FRAME_MASTER_True_False.pickle").copy()
# LMED_05_06 = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["LMED_05_06"], GB.DTYPES_DICT["LMED_05_06"]).copy()
print("Reading LMED_05_06...")
LMED_05_06 = GB.READ_RAWFILE_TABSEP(GB.RAWDATA_DICT["LMED_05_06"]).copy()
#OV_ORIG = GB.READ_RAWFILE_TABSEP(GB.RAWDATA_DICT["OV"]).copy()
print("LMED_05_06 Finished")


# Check all lengths of the dates before transforming them
print("Creating datelengths... 1")

dates = LMED_05_06["EDATUM"].astype(str)
datelengths = [pd.NA]*len(dates)
for i in range(0, len(dates)):
    datelengths[i] = len(dates[i])

print("Finished1")

# GB.table_nan(pd.Series(datelengths))
# 10    133235103
# dtype: int64

# GB.table_nan(LMED_05_06["EDATUM"].isna())
# GB.table_nan(OV["INDATUMA"]=="")

# Check out those who are NaN in the length??

#print("Finished3")

# Create dates:
print("Creating dates...")

# datetime.datetime.strptime(LMED_05_06["EDATUM"][0], "%Y-%m-%d")

OV["INDATUMA_date"] = OV["INDATUMA"].astype(str).apply(lambda x: GB.convert_date_YYYYMMDD_YYYYMM15(x)).copy()

# Only take those with 8 length date
# OV.loc[OV["INDATUMA_len"]!=8, "INDATUMA_date"] = pd.NA

print("Finished4")

# Check those that are 8 length but still NA
# x=OV.loc[(OV["INDATUMA_date"].isna()) & (OV["INDATUMA_len"]==8), "INDATUMA"]

# Spara OV
print("Saving...")
OV.to_pickle("_000060_OV.pickle")
print("Finished saving")





