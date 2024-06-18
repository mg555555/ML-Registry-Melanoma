from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion, make_pipeline, Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer

import numpy as np
import pickle
import pandas as pd

class NonTimeSeriesDataGenerator:
    def __init__(self, pickle_path):
        with open(pickle_path, 'rb') as f:
            self.MASTER = pickle.load(f)

        self.MASTER_train = self.MASTER[self.MASTER["GROUP_TRAINVALTEST"] == "train"]
        self.MASTER_val = self.MASTER[self.MASTER["GROUP_TRAINVALTEST"] == "val"]

        # Remove outcome column and column for train/validation grouping
        self.X_train = self.MASTER_train.loc[:, (self.MASTER_train.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE') & (self.MASTER_train.columns != 'GROUP_TRAINVALTEST')]
        self.y_train = self.MASTER_train.loc[:, self.MASTER_train.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']

        self.X_val = self.MASTER_val.loc[:, (self.MASTER_val.columns != 'MM_BETWEEN_INDEX_AND_ENDDATE') & (self.MASTER_val.columns != 'GROUP_TRAINVALTEST')]
        self.y_val = self.MASTER_val.loc[:, self.MASTER_val.columns == 'MM_BETWEEN_INDEX_AND_ENDDATE']
        
        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        
        return None

    def create_synthetic_data(self, num_row_train=2000, num_row_val=500):
        X_train_data = pd.DataFrame({col: np.random.choice([True, False], size=num_row_train) for col in self.X_train.columns})
        self.X_train = pd.DataFrame(X_train_data, columns=self.X_train.columns)

        X_val_data = pd.DataFrame({col: np.random.choice([True, False], size=num_row_val) for col in self.X_val.columns})
        self.X_val = pd.DataFrame(X_val_data, columns=self.X_val.columns)

        y_train_data = pd.DataFrame({col: np.random.choice([False, True], size=num_row_train, p=[0.8, 0.2]) for col in self.y_train.columns})
        self.y_train = pd.DataFrame(y_train_data, columns=self.y_train.columns)
        
        y_val_data = pd.DataFrame({col: np.random.choice([False, True], size=num_row_val, p=[0.8, 0.2]) for col in self.y_val.columns})
        self.y_val = pd.DataFrame(y_val_data, columns=self.y_val.columns)

        self.y_train = self.y_train.to_numpy().ravel()
        self.y_val = self.y_val.to_numpy().ravel()

        # Transform into integer values
        self.y_train = self.y_train.astype(int)
        self.y_val = self.y_val.astype(int)

        self.X_train = self.X_train.astype(int)
        self.X_val = self.X_val.astype(int)

        return None

    def get_train_data(self):
        """Return the training data (features and target)."""
        return self.X_train, self.y_train

    def get_val_data(self):
        """Return the validation data (features and target)."""
        return self.X_val, self.y_val


