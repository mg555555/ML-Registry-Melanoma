import pandas as pd
import numpy as np
from data.data_generation import NonTimeSeriesDataGenerator



class Data_loader:
    """
    This class loads the data
    """
    
    def __init__(self, data_path=None, pickle_path=None):
        """
        Initialize the data loader with the data path or generator
        """
        self.data_path = data_path
        self.pickle_path = pickle_path
    
    def load_data(self):
        """
        Load the data from a CSV file or generate synthetic data
        """
        if self.data_path:
            # Load binary time series data from csv file
            self.data = pd.read_csv(self.data_path)
            # Assuming the CSV has the same structure as the generated data
            self.X_train, self.X_val = self.data.iloc[:80, :-1], self.data.iloc[80:, :-1]
            self.y_train, self.y_val = self.data.iloc[:80, -1], self.data.iloc[80:, -1]
        elif self.pickle_path:
            # Generate synthetic data using the provided pickle file
            generator = NonTimeSeriesDataGenerator(self.pickle_path)
            generator.create_synthetic_data()
            self.X_train, self.y_train = generator.get_train_data()
            self.X_val, self.y_val = generator.get_val_data()
        return self.X_train, self.y_train, self.X_val, self.y_val
