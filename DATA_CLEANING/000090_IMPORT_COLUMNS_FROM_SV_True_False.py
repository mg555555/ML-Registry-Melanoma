import _Globalconstants as GB
import time
import pandas as pd
import datetime
#import math

# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', None)
# pd.option_context('display.max_rows', None, 'display.max_columns', None)

print("Reading SV...")
SV = GB.READ_RAWFILE_DTYPES_TABSEP(GB.RAWDATA_DICT["SV"], GB.DTYPES_DICT["SV"]).copy()
print("SV read")

# Check all lengths of the dates before transforming them
print("Creating datelengths... 1")

dates = SV["INDATUMA"].astype(str)
datelengths = [pd.NA]*len(dates)
for i in range(0, len(dates)):
    datelengths[i] = len(dates[i])

print("Finished1")

# GB.table_nan(pd.Series(datelengths))
# 8    42717146
# 6        1554
# 4        1456
# 3         947
# 2         197
# 7          62
# 5           1
# 1           1
# dtype: int64

# GB.table_nan(SV["INDATUMA"].isna())
# GB.table_nan(SV["INDATUMA"]=="")

# Check out those who are NaN in the length??
print("Creating datelengths... 2")

SV["INDATUMA_len"] = pd.Series(datelengths)
date3s = SV.loc[SV["INDATUMA_len"]==3, "INDATUMA"]
date4s = SV.loc[SV["INDATUMA_len"]==4, "INDATUMA"]
# Some of these 6a are wrong(060201)
date6s = SV.loc[SV["INDATUMA_len"]==6, "INDATUMA"]
date7s = SV.loc[SV["INDATUMA_len"]==7, "INDATUMA"]

lopnrtmp = SV.loc[SV["INDATUMA"].isna(), "LopNr"]

print("Finished2")

# Check the "originals":
#SV_ORIG["INDATUMA_len"] = pd.Series(datelengths)
#date3s_ORIG = SV_ORIG.loc[SV_ORIG["INDATUMA_len"]==3, "INDATUMA"]
#date4s_ORIG = SV_ORIG.loc[SV_ORIG["INDATUMA_len"]==4, "INDATUMA"]
# Some of these 6a are wrong(060201)
#date6s_ORIG = SV_ORIG.loc[SV_ORIG["INDATUMA_len"]==6, "INDATUMA"]
#date7s_ORIG = SV_ORIG.loc[SV_ORIG["INDATUMA_len"]==7, "INDATUMA"]

#print("Finished3")

# Create dates:
print("Creating dates...")

SV["INDATUMA_date"] = SV["INDATUMA"].astype(str).apply(lambda x: GB.convert_date_YYYYMMDD_YYYYMM15(x)).copy()

# Only take those with 8 length (all 6s are before 2005 for example)
SV.loc[SV["INDATUMA_len"]!=8, "INDATUMA_date"] = pd.NA

print("Finished4")

# GB.table_nan(SV["INDATUMA_date"].isna(), SV["INDATUMA_len"])
# INDATUMA_len   1    2    3     4  5     6   7         8
# INDATUMA_date
# False          0    0    0     0  0     0   0  42717110
# True           1  197  947  1456  1  1554  62        36

# Check those that are 8 length but still NA
# x=SV.loc[(SV["INDATUMA_date"].isna()) & (SV["INDATUMA_len"]==8), "INDATUMA"]

# Spara SV
print("Saving...")
SV.to_pickle("_000090_SV.pickle")
print("Finished saving")





