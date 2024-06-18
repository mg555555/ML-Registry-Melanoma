import os
import json
import pickle
import logging
import yaml
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from data.data_generation import NonTimeSeriesDataGenerator
from training.trainer import Trainer
from data.data_loading import Data_loader

def load_config(config_path):
    """
    Load the configuration from a YAML file.
    
    Parameters:
    config_path (str): The path to the YAML configuration file.

    Returns:
    dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def calibrate_classifier(model, X_train, y_train):
    """
    Calibrate the classifier using CalibratedClassifierCV.
    
    Parameters:
    model: The model to be calibrated.
    X_train: Training features.
    y_train: Training labels.

    Returns:
    CalibratedClassifierCV: The calibrated model.
    """
    calibrated_model = CalibratedClassifierCV(estimator=model, cv='prefit')
    calibrated_model.fit(X_train, y_train)
    return calibrated_model

def get_logger(log_file, log_level=logging.INFO):
    """
    Set up a logger that writes to a file with a specific format.

    Parameters:
    log_file (str): The path to the log file.
    log_level (logging.LEVEL): The logging level.

    Returns:
    logger: Configured logger instance.
    """
    # Ensure the log directory exists
    log_dir = os.path.dirname(log_file)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger('MLProject')
    logger.setLevel(log_level)
    handler = logging.FileHandler(log_file)
    handler.setLevel(log_level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def initialize_model(config):
    """
    Initialize a model based on the configuration.

    Parameters:
    config (dict): Configuration dictionary.

    Returns:
    model: Initialized model instance.
    """
    model_type = config['model']['type']
    hyperparameters = config['model']['hyperparameters'][model_type]
    
    if model_type == 'decision_tree':
        model = DecisionTreeClassifier(**hyperparameters)
    elif model_type == 'logistic_regression':
        model = LogisticRegression(**hyperparameters)
    elif model_type == 'xgboost':
        model = XGBClassifier(**hyperparameters)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model

def main():
    # Load configuration
    config_path = 'configs/config.yaml'
    config = load_config(config_path)

    # Set up logger
    logger = get_logger(config['logging']['log_file'], log_level=logging.INFO)
    logger.info("Starting the training process")

    # Load and preprocess data
    data_loader = Data_loader(pickle_path=config['data']['pickle_path'])
    X_train, y_train, X_val, y_val = data_loader.load_data()
    logger.info("Data loaded successfully")

    # Initialize the model
    model = initialize_model(config)
    logger.info(f"Model {config['model']['type']} initialized")

    # Train the model
    trainer = Trainer(config, logger)
    best_model, validation_metrics = trainer.train_model(model, X_train, y_train, X_val, y_val)
    logger.info("Model training completed")

    # Calibrate the model
    calibrated_model = calibrate_classifier(best_model, X_train, y_train)
    logger.info("Model calibration completed")

    # Evaluate the model
    y_val_pred = calibrated_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred)

    logger.info(f"Accuracy: {accuracy}")
    logger.info(f"F1 Score: {f1}")
    logger.info(f"ROC AUC: {roc_auc}")

    # Ensure the output directory exists
    output_dir = config['data']['output_path']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the trained model and metrics if needed
    if config['evaluation']['save_model']:
        model_output_path = os.path.join(output_dir, 'model.pkl')
        with open(model_output_path, 'wb') as f:
            pickle.dump(calibrated_model, f)
        logger.info(f"Model saved to {model_output_path}")

    if config['evaluation']['save_metrics']:
        metrics_output_path = os.path.join(output_dir, 'metrics.json')
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        with open(metrics_output_path, 'w') as f:
            json.dump(metrics, f)
        logger.info(f"Metrics saved to {metrics_output_path}")
        print("Model training completed")

if __name__ == "__main__":
    main()
