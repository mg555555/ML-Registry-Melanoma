import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import data.data_generation as data_generation

class get_df:
    #get data from data-generation.py
    data = data_generation.NonTimeSeriesDataGenerator
    
    
    #define X, y
    X = data.X_train
    y = data.y_train
    
    #split data into train and test
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=.2)
    #get valudation set 
    X_val = data.X_val
    
    