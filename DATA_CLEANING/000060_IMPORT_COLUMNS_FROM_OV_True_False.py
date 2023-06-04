import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)

# Read MASTER_TMP file
#MASTER = pd.read_pickle("_000050_FRAME_MASTER_True_False.pickle").copy()
OV = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["OV"], GB.DTYPES_DICT["OV"]).copy()
#OV_ORIG = GB.READ_RAWFILE_TABSEP(GB.RAWDATA_DICT["OV"]).copy()

print("OV read")

# Check all lengths of the dates before transforming them
print("Creating datelengths... 1")

dates = OV["INDATUMA"].astype(str)
datelengths = [pd.NA]*len(dates)
for i in range(0, len(dates)):
    datelengths[i] = len(dates[i])

print("Finished1")

# GB.table_nan(pd.Series(datelengths))
# 8    166940416
# 3        34857
# 4        29814
# 1           55
# 7           15
# 6            3
# 5            2
# dtype: int64

# GB.table_nan(OV["INDATUMA"].isna())
# GB.table_nan(OV["INDATUMA"]=="")

# Check out those who are NaN in the length??
print("Creating datelengths... 2")

OV["INDATUMA_len"] = pd.Series(datelengths)
date3s = OV.loc[OV["INDATUMA_len"]==3, "INDATUMA"]
date4s = OV.loc[OV["INDATUMA_len"]==4, "INDATUMA"]
# Some of these 6a are wrong(060201)
date6s = OV.loc[OV["INDATUMA_len"]==6, "INDATUMA"]
date7s = OV.loc[OV["INDATUMA_len"]==7, "INDATUMA"]

lopnrtmp = OV.loc[OV["INDATUMA"].isna(), "LopNr"]

print("Finished2")

# Check the "originals":
#OV_ORIG["INDATUMA_len"] = pd.Series(datelengths)
#date3s_ORIG = OV_ORIG.loc[OV_ORIG["INDATUMA_len"]==3, "INDATUMA"]
#date4s_ORIG = OV_ORIG.loc[OV_ORIG["INDATUMA_len"]==4, "INDATUMA"]
# Some of these 6a are wrong(060201)
#date6s_ORIG = OV_ORIG.loc[OV_ORIG["INDATUMA_len"]==6, "INDATUMA"]
#date7s_ORIG = OV_ORIG.loc[OV_ORIG["INDATUMA_len"]==7, "INDATUMA"]

#print("Finished3")

# Create dates:
print("Creating dates...")

OV["INDATUMA_date"] = OV["INDATUMA"].astype(str).apply(lambda x: GB.convert_date_YYYYMMDD_YYYYMM15(x)).copy()

# Only take those with 8 length date
OV.loc[OV["INDATUMA_len"]!=8, "INDATUMA_date"] = pd.NA

print("Finished4")

# Check those that are 8 length but still NA
# x=OV.loc[(OV["INDATUMA_date"].isna()) & (OV["INDATUMA_len"]==8), "INDATUMA"]

# Spara OV
print("Saving...")
OV.to_pickle("_000060_OV.pickle")
print("Finished saving")





