from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle

import numpy as np
import pickle
import pandas as pd

class TimeSeriesDataGenerator:
    def __init__(self, pickle_path, time_steps):
        with open(pickle_path, 'rb') as f:
            self.MASTER = pickle.load(f)

        self.time_steps = time_steps

        self.MASTER_train = self.MASTER[self.MASTER["GROUP_TRAINVALTEST"] == "train"]
        self.MASTER_val = self.MASTER[self.MASTER["GROUP_TRAINVALTEST"] == "val"]

        self.X_train = self.MASTER_train.drop(columns=['MM_BETWEEN_INDEX_AND_ENDDATE', 'GROUP_TRAINVALTEST'])
        self.y_train = self.MASTER_train['MM_BETWEEN_INDEX_AND_ENDDATE']

        self.X_val = self.MASTER_val.drop(columns=['MM_BETWEEN_INDEX_AND_ENDDATE', 'GROUP_TRAINVALTEST'])
        self.y_val = self.MASTER_val['MM_BETWEEN_INDEX_AND_ENDDATE']
        
        self.X_train = self.X_train.reset_index(drop=True)
        self.y_train = self.y_train.reset_index(drop=True)
        
        return None

    def create_synthetic_data(self, num_samples_train=80, num_samples_val=30):
        X_train_data = np.random.choice([0, 1], size=(num_samples_train, self.time_steps, self.X_train.shape[1]))
        self.X_train = pd.DataFrame(np.reshape(X_train_data, (num_samples_train * self.time_steps, self.X_train.shape[1])),
                                    columns=self.X_train.columns)

        X_val_data = np.random.choice([0, 1], size=(num_samples_val, self.time_steps, self.X_val.shape[1]))
        self.X_val = pd.DataFrame(np.reshape(X_val_data, (num_samples_val * self.time_steps, self.X_val.shape[1])),
                                  columns=self.X_val.columns)

        y_train_data = np.random.choice([0, 1], size=(num_samples_train, self.time_steps), p=[0.8, 0.2])
        self.y_train = np.ravel(y_train_data)

        y_val_data = np.random.choice([0, 1], size=(num_samples_val, self.time_steps), p=[0.8, 0.2])
        self.y_val = np.ravel(y_val_data)

        self.X_train = self.X_train.astype(int)
        self.X_val = self.X_val.astype(int)

        return None

    def get_train_data(self):
        """Return the training data (features and target) in tensor format."""
        num_samples_train = len(self.y_train) // self.time_steps
        X_train_tensor = np.reshape(self.X_train.to_numpy(), (num_samples_train, self.time_steps, -1))
        y_train_tensor = np.reshape(self.y_train, (num_samples_train, self.time_steps))
        return X_train_tensor, y_train_tensor

    def get_val_data(self):
        """Return the validation data (features and target) in tensor format."""
        num_samples_val = len(self.y_val) // self.time_steps
        X_val_tensor = np.reshape(self.X_val.to_numpy(), (num_samples_val, self.time_steps, -1))
        y_val_tensor = np.reshape(self.y_val, (num_samples_val, self.time_steps))
        return X_val_tensor, y_val_tensor
