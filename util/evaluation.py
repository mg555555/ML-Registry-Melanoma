import yaml
import joblib
import pickle
import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.inspection import permutation_importance
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
# Add additional necessary imports based on your classifiers


class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        self.results = []
        self.feature_importances = []
    
    def evaluate_models(self, X, D_t, y_col):
        for classifier_name, params in self.config['model']['hyperparameters'].items():
            for param in ParameterGrid(params):
                model = self._get_model_instance(classifier_name)
                model.set_params(**param)
                for i in range(5):  # Iterating over different splits
                    X_train, X_test, y_train, y_test = train_test_split(X, D_t[y_col], stratify=D_t[y_col], test_size=0.2)
                    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, stratify=y_train, test_size=0.2)
                    
                    if classifier_name in ['xgboost', 'PSM_lr', 'SPSM_lr']:
                        y_train = y_train.replace({'no': 0, 'yes': 1})
                        y_test = y_test.replace({'no': 0, 'yes': 1})
                        y_val = y_val.replace({'no': 0, 'yes': 1})

                    model.fit(X_train, y_train)
                    
                    # Save the fitted model to pkl file
                    joblib.dump(model, f'model_{classifier_name}_{i}.pkl')
                    print(f"Model saved to model_{classifier_name}_{i}.pkl")
                    
                    self._evaluate_and_save_results(model, classifier_name, param, i, X_train, y_train, X_test, y_test, X_val, y_val)
                    self._extract_feature_importances(model, classifier_name, X_train, X_test, y_test, i)
    
    def _get_model_instance(self, classifier_name):
        classifier_dict = {
            'decision_tree': DecisionTreeClassifier(),
            'logistic_regression': LogisticRegression(),
            'xgboost': HistGradientBoostingClassifier(),
            'PSM_lr': self.config['model']['classifiers']['PSM_lr'],
            'SPSM_lr': self.config['model']['classifiers']['SPSM_lr'],
            'redflag': self.redflag_inst
        }
        return classifier_dict[classifier_name]

    def _evaluate_and_save_results(self, model, classifier_name, param, split, X_train, y_train, X_test, y_test, X_val, y_val):
        train_auc = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        test_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        val_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
        train_f1 = f1_score(y_train, model.predict(X_train), average='weighted')
        test_f1 = f1_score(y_test, model.predict(X_test), average='weighted')
        val_f1 = f1_score(y_val, model.predict(X_val), average='weighted')
        train_cm = confusion_matrix(y_train, model.predict(X_train))
        test_cm = confusion_matrix(y_test, model.predict(X_test))
        val_cm = confusion_matrix(y_val, model.predict(X_val))
        
        self.results.append({
            'model': classifier_name,
            'params': param,
            'split': split,
            'train_auc': train_auc,
            'test_auc': test_auc,
            'val_auc': val_auc,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'val_f1': val_f1,
            'train_cm': train_cm,
            'test_cm': test_cm,
            'val_cm': val_cm
        })

    def _extract_feature_importances(self, model, classifier_name, X_train, X_test, y_test, seed):
        if classifier_name == 'decision_tree':
            feature_names = X_train.columns
            importances = model.feature_importances_
            important_features = {feature_names[idx]: importances[idx] for idx in np.where(importances > 0)[0]}
            self.feature_importances.append(important_features)
            
        elif classifier_name == 'xgboost':
            result = permutation_importance(model, X_test, y_test, n_repeats=10, random_state=42)
            importances = result.importances_mean
            std_devs = result.importances_std
            feature_names = X_train.columns
            
            data = {
                'Feature': feature_names,
                'Importance': importances,
                'Standard Deviation': std_devs,
                'Seed': [seed] * len(feature_names)
            }
            importance_df = pd.DataFrame(data)
            importance_df = importance_df[importance_df['Importance'] > 0]
            self.feature_importances.append(importance_df)
                    
        elif classifier_name == 'logistic_regression':
            feature_names = X_train.columns
            coefficients = model.coef_[0]
            sorted_features = sorted(zip(feature_names, coefficients), key=lambda x: abs(x[1]), reverse=True)
            important_features = {feat: coef for feat, coef in sorted_features[:10]}
            self.feature_importances.append(important_features)

        elif classifier_name in ['PSM_lr', 'SPSM_lr']:
            self.feature_importances.append('not implemented')

        elif classifier_name == 'redflag':
            self.feature_importances.append('not implemented')

    def get_results(self):
        df_results = pd.DataFrame(self.results)
        df_results['feature_importances'] = self.feature_importances
        return df_results

