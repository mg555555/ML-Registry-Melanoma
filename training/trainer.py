# training/trainer.py

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

class Trainer:
    def __init__(self, config, logger):
        """
        Initializes the Trainer with the given configuration and logger.
        
        Parameters:
        - config: dict, configuration settings loaded from config.yaml
        - logger: Logger, logger instance for logging information
        """
        self.config = config
        self.logger = logger

    def train_model(self, model, X_train, y_train, X_val, y_val):
        """
        Trains the given model using GridSearchCV for hyperparameter tuning.
        
        Parameters:
        - model: object, the model to be trained
        - X_train: array-like, training features
        - y_train: array-like, training labels
        - X_val: array-like, validation features
        - y_val: array-like, validation labels
        """
        model_name = type(model).__name__
        self.logger.info(f"Training {model_name}...")

        # Map model class names to the keys in the config hyperparameters
        model_name_mapping = {
            'DecisionTreeClassifier': 'decision_tree',
            'LogisticRegression': 'logistic_regression',
            'XGBClassifier': 'xgboost',
            'SharingLogisticSubModel': 'SPSM_lr',  
        }

        if model_name not in model_name_mapping:
            raise ValueError(f"Unknown model type: {model_name}")

        # Fetch hyperparameters from the config
        config_model_name = model_name_mapping[model_name]
        param_grid = self.config['model']['hyperparameters'][config_model_name]
        
        # Initialize GridSearchCV
        grid_search = GridSearchCV(estimator=model,
                                   param_grid=param_grid,
                                   cv=self.config['training']['cross_validation']['params']['n_splits'],
                                   scoring=self.config['training']['scoring'],
                                   n_jobs=-1,
                                   verbose=1)
        
        # Fit the model
        grid_search.fit(X_train, y_train)
        
        # Log best parameters and best score
        self.logger.info(f"Best parameters for {model_name}: {grid_search.best_params_}")
        self.logger.info(f"Best cross-validation score for {model_name}: {grid_search.best_score_}")

        # Set the best estimator as the model
        best_model = grid_search.best_estimator_

        # Validate the model on validation set
        y_val_pred = best_model.predict(X_val)
        accuracy = accuracy_score(y_val, y_val_pred)
        f1 = f1_score(y_val, y_val_pred)
        roc_auc = roc_auc_score(y_val, y_val_pred)
        
        self.logger.info(f"Validation Accuracy for {model_name}: {accuracy}")
        self.logger.info(f"Validation F1 Score for {model_name}: {f1}")
        self.logger.info(f"Validation ROC AUC for {model_name}: {roc_auc}")

        # Return the trained model and validation metrics
        return best_model, {'accuracy': accuracy, 'f1': f1, 'roc_auc': roc_auc}
