import _Globalconstants as GB
import time
import pandas as pd
import datetime
import sys
# import random
import numpy as np


# np.random.get_state()
# np.random.RandomState(1234)


MASTER = pd.read_pickle("_000030_FRAME_MASTER.pickle").copy()

#--------------------------------------------------------------------------
# Create train, val and testset
NUMBER_CASES = sum(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"])
NUMBER_KONTS = sum(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"])

NUMBER_TRAIN_CASES = NUMBER_CASES - GB.VAL_SIZE_CASES - GB.TEST_SIZE_CASES

TRAINVALTEST_CASES = pd.Series(["train"]*NUMBER_TRAIN_CASES + \
                     ["val"]*GB.VAL_SIZE_CASES + \
                     ["test"]*GB.TEST_SIZE_CASES)

# Enter seed GB.SEED_TRAIN_VAL_TESTSET
# using np.random.RandomState(GB.SEED_TRAIN_VAL_TESTSET)
# does not generate the same result
TRAINVALTEST_CASES_SAMPLED = TRAINVALTEST_CASES.sample(n=NUMBER_CASES, random_state=GB.SEED_TRAIN_VAL_TESTSET_CASES).copy()
#--------------------------------------------------------------------------
# Do the corresponding for the controls
NUMBER_TRAIN_KONTS = round(NUMBER_KONTS*NUMBER_TRAIN_CASES/NUMBER_CASES)
NUMBER_VAL_KONTS   = round(NUMBER_KONTS*GB.VAL_SIZE_CASES/NUMBER_CASES)
NUMBER_TEST_KONTS  = round(NUMBER_KONTS*GB.TEST_SIZE_CASES/NUMBER_CASES)


if NUMBER_TRAIN_KONTS+NUMBER_VAL_KONTS+NUMBER_TEST_KONTS!=NUMBER_KONTS:
    print("The sum of control groups does not add up")
    sys.exit()

TRAINVALTEST_KONTS = pd.Series(["train"]*NUMBER_TRAIN_KONTS + \
                     ["val"]*NUMBER_VAL_KONTS + \
                     ["test"]*NUMBER_TEST_KONTS)

TRAINVALTEST_KONTS_SAMPLED = TRAINVALTEST_KONTS.sample(n=NUMBER_KONTS, random_state=GB.SEED_TRAIN_VAL_TESTSET_KONTS).copy()
#--------------------------------------------------------------------------
# Create group variable in MASTER:
MASTER["GROUP_TRAINVALTEST"] = pd.NA

# If not casting to list (removing indexes) it does not work??
# .reset_index() may also work??
MASTER.loc[MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"], "GROUP_TRAINVALTEST"] = list(TRAINVALTEST_CASES_SAMPLED).copy()

MASTER.loc[~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"], "GROUP_TRAINVALTEST"] = list(TRAINVALTEST_KONTS_SAMPLED).copy()
#--------------------------------------------------------------------------
# Make a smaller version of the control-groups (1/10)
# NOTE! We must also randomize "empty" (NA's) in this series
# We should make it nested, so that 1/10 of train controls gets
# randomized to the smaller set and likewise for val and test
NUMBER_TRAIN_KONTS_10 = round(NUMBER_KONTS*NUMBER_TRAIN_CASES/NUMBER_CASES/10)
NUMBER_VAL_KONTS_10   = round(NUMBER_KONTS*GB.VAL_SIZE_CASES/NUMBER_CASES/10)
NUMBER_TEST_KONTS_10  = round(NUMBER_KONTS*GB.TEST_SIZE_CASES/NUMBER_CASES/10)
NUMBER_NA_TRAIN_KONTS_10 = NUMBER_TRAIN_KONTS - NUMBER_TRAIN_KONTS_10
NUMBER_NA_VAL_KONTS_10   = NUMBER_VAL_KONTS   - NUMBER_VAL_KONTS_10
NUMBER_NA_TEST_KONTS_10  = NUMBER_TEST_KONTS  - NUMBER_TEST_KONTS_10

TRAIN_KONTS_10 = pd.Series(["train"]*NUMBER_TRAIN_KONTS_10 + \
                        [pd.NA]*NUMBER_NA_TRAIN_KONTS_10)

VAL_KONTS_10 = pd.Series(["val"]*NUMBER_VAL_KONTS_10 + \
                        [pd.NA]*NUMBER_NA_VAL_KONTS_10)

TEST_KONTS_10 = pd.Series(["test"]*NUMBER_TEST_KONTS_10 +
                        [pd.NA]*NUMBER_NA_TEST_KONTS_10)

TRAIN_KONTS_10_SAMPLED = TRAIN_KONTS_10.sample(n=NUMBER_TRAIN_KONTS, random_state=GB.SEED_TRAIN_KONTS_10).copy()

VAL_KONTS_10_SAMPLED = VAL_KONTS_10.sample(n=NUMBER_VAL_KONTS, random_state=GB.SEED_VAL_KONTS_10).copy()

TEST_KONTS_10_SAMPLED = TEST_KONTS_10.sample(n=NUMBER_TEST_KONTS, random_state=GB.SEED_TEST_KONTS_10).copy()

# Assign new nested smaller (1/10) labels
MASTER["GROUP_TRAINVALTEST_10"] = pd.NA

MASTER.loc[(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="train"), "GROUP_TRAINVALTEST_10"] = list(TRAIN_KONTS_10_SAMPLED).copy()

MASTER.loc[(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="val"), "GROUP_TRAINVALTEST_10"] = list(VAL_KONTS_10_SAMPLED).copy()

MASTER.loc[(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="test"), "GROUP_TRAINVALTEST_10"] = list(TEST_KONTS_10_SAMPLED).copy()

# Add the cases also in to the same column:
MASTER.loc[(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="train"), "GROUP_TRAINVALTEST_10"] = "train"

MASTER.loc[(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="val"), "GROUP_TRAINVALTEST_10"] = "val"

MASTER.loc[(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="test"), "GROUP_TRAINVALTEST_10"] = "test"

#--------------------------------------------------------------------------

# Make a smaller version of the control-groups (1/100)
NUMBER_TRAIN_KONTS_100 = round(NUMBER_KONTS*NUMBER_TRAIN_CASES/NUMBER_CASES/100)
NUMBER_VAL_KONTS_100   = round(NUMBER_KONTS*GB.VAL_SIZE_CASES/NUMBER_CASES/100)
NUMBER_TEST_KONTS_100  = round(NUMBER_KONTS*GB.TEST_SIZE_CASES/NUMBER_CASES/100)
NUMBER_NA_TRAIN_KONTS_100 = NUMBER_TRAIN_KONTS_10 - NUMBER_TRAIN_KONTS_100
NUMBER_NA_VAL_KONTS_100   = NUMBER_VAL_KONTS_10   - NUMBER_VAL_KONTS_100
NUMBER_NA_TEST_KONTS_100  = NUMBER_TEST_KONTS_10  - NUMBER_TEST_KONTS_100

TRAIN_KONTS_100 = pd.Series(["train"]*NUMBER_TRAIN_KONTS_100 + \
                        [pd.NA]*NUMBER_NA_TRAIN_KONTS_100)

VAL_KONTS_100 = pd.Series(["val"]*NUMBER_VAL_KONTS_100 + \
                        [pd.NA]*NUMBER_NA_VAL_KONTS_100)

TEST_KONTS_100 = pd.Series(["test"]*NUMBER_TEST_KONTS_100 +
                        [pd.NA]*NUMBER_NA_TEST_KONTS_100)

TRAIN_KONTS_100_SAMPLED = TRAIN_KONTS_100.sample(n=NUMBER_TRAIN_KONTS_10, random_state=GB.SEED_TRAIN_KONTS_100).copy()

VAL_KONTS_100_SAMPLED = VAL_KONTS_100.sample(n=NUMBER_VAL_KONTS_10, random_state=GB.SEED_VAL_KONTS_100).copy()

TEST_KONTS_100_SAMPLED = TEST_KONTS_100.sample(n=NUMBER_TEST_KONTS_10, random_state=GB.SEED_TEST_KONTS_100).copy()

# Assign new nested smaller (1/10) labels
MASTER["GROUP_TRAINVALTEST_100"] = pd.NA

MASTER.loc[(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST_10"]=="train"), "GROUP_TRAINVALTEST_100"] = list(TRAIN_KONTS_100_SAMPLED).copy()

MASTER.loc[(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST_10"]=="val"), "GROUP_TRAINVALTEST_100"] = list(VAL_KONTS_100_SAMPLED).copy()

MASTER.loc[(~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST_10"]=="test"), "GROUP_TRAINVALTEST_100"] = list(TEST_KONTS_100_SAMPLED).copy()


# Add the cases also into the same column:
MASTER.loc[(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="train"), "GROUP_TRAINVALTEST_100"] = "train"

MASTER.loc[(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="val"), "GROUP_TRAINVALTEST_100"] = "val"

MASTER.loc[(MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]) & (MASTER["GROUP_TRAINVALTEST"]=="test"), "GROUP_TRAINVALTEST_100"] = "test"
#--------------------------------------------------------------------------

TMP_CASES = MASTER[MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]]
TMP_KONTS = MASTER[~MASTER["MM_BETWEEN_INDEX_AND_ENDDATE"]]

del TMP_CASES
del TMP_KONTS

# GB.table_nan(TMP_CASES["GROUP_TRAINVALTEST"], TMP_CASES["GROUP_TRAINVALTEST_10"])
# GB.table_nan(TMP_KONTS["GROUP_TRAINVALTEST"], TMP_KONTS["GROUP_TRAINVALTEST_10"])
# GB.table_nan(TMP_CASES["GROUP_TRAINVALTEST_10"], TMP_CASES["GROUP_TRAINVALTEST_100"])
# GB.table_nan(TMP_KONTS["GROUP_TRAINVALTEST_10"], TMP_KONTS["GROUP_TRAINVALTEST_100"])

MASTER.to_pickle("_000040_FRAME_MASTER.pickle")

print("Finished")












