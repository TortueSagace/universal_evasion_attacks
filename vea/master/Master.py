"""
The Master class allows handling of all attacks, neighborhoods, model training, data, testing, and optimization.
The attributes of Master can be changed by the chatbot depending on the user's queries.
Parameters are built as params[category][name][parameter].

- Categories are: model, data, attack, neighborhood, environment
- Model (default values, can be completed by the user): XGBClassifier, LGBClassifier, ExtraTreesClassifier, HistogramGradientBoostingClassifier.
- Data: should be added by the user but contains default datasets: Iris and MNIST.
- Attack (fixed list): HillClimbing, SimulatedAnnealing, TabuSearch.
- Neighborhood (fixed list): Radar.
- Environment (fixed list): time, user, ...
"""

import sys
import os
import json
import shutil
import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder
from scipy.stats import pearsonr
import optuna
import time
import importlib
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import pkg_resources

file_path = pkg_resources.resource_filename(
            'vea', 'master/master_params.json'
        )

class Master:
    def __init__(self, params_file=file_path, modifications_log='modifications.log', verbosity=1):
        self.params_file = params_file
        self.modifications_log = modifications_log
        self.verbosity = verbosity  # Control verbosity
        self.random_state = 42  # Global random state, can be changed by the user

        # Load default parameters
        with open(params_file, 'r') as file:
            self.params = json.load(file)

        # Keep a copy of default parameters for rollback
        self.default_params = json.loads(json.dumps(self.params))  # Deep copy

        # Initialize parameters
        self.model_params = self.params.get('model', {})
        self.data_params = self.params.get('data', {})
        self.attack_params = self.params.get('attack', {})
        self.neighborhood_params = self.params.get('neighborhood', {})
        # Environment parameters (not implemented yet)
        # self.environment_params = self.params.get('environment', {})

        # Initialize modifications log
        if not os.path.exists(self.modifications_log):
            with open(self.modifications_log, 'w') as log_file:
                log_file.write('Modifications Log\n')
                log_file.write('=================\n\n')

        # Dictionary to store model performances per dataset
        self.model_performances = {}
        # Dictionary to store attack performances per dataset
        self.attack_performances = {}
        # Store mu_k per dataset for SLARC cost function
        self.mu_k = {}
        # Store attack histories per dataset and attack
        self.attack_histories = {}

    # ----------------------- Data Handling Methods ----------------------- #

    def load_dataset(self, dataset_name, dataset_path):
        """
        Load a dataset from the given path and update the data section in the JSON file.

        Parameters:
            dataset_name: str, the name of the dataset (e.g., 'CICIoT23')
            dataset_path: str, the path to the dataset file (e.g., 'data/CICIoT23/merged_data.csv')
        """
        if dataset_name in self.data_params:
            if self.verbosity > 0:
                print(f"Dataset {dataset_name} already exists in data parameters.")
            return

        # Update data parameters
        new_dataset_params = {
            "test_split": 0.2,
            "apply_standardization": False,
            "apply_normalization": False,
            "apply_yeojohnson": False,
            "shuffle": True,
            "random_state": self.random_state,
            "features": "all",
            "target": "label",
            "download": False,
            "data_path": dataset_path,
            "max_rows_number": None  # No limit by default
        }

        self.data_params[dataset_name] = new_dataset_params
        self.params['data'] = self.data_params

        # Record modification
        self.record_modification('data', dataset_name, new_dataset_params)

        # Save updated parameters to JSON file
        self.save_params()

        if self.verbosity > 0:
            print(f"Dataset {dataset_name} has been added to data parameters.")

    def merge_csv_files(self, dataset_name, directory_path, label_mappings, max_rows_number=None):
        """
        Merge all CSV files in a directory into a single dataset with label assignment.

        Parameters:
            dataset_name: str, the name of the dataset to create or update.
            directory_path: str, the path to the directory containing CSV files.
            label_mappings: dict, mapping of filename patterns to labels (e.g., {'Mirai': 'Mirai'}).
            max_rows_number: int, optional, maximum number of rows to read.

        Example:
            label_mappings = {
                'Mirai': 'Mirai',
                'Gafgyt': 'Gafgyt'
            }
        """
        if self.verbosity > 0:
            print(f"Merging CSV files from {directory_path} into dataset {dataset_name}...")
        merged_data = []
        total_rows = 0

        for filename in os.listdir(directory_path):
            if filename.endswith('.csv'):
                file_path = os.path.join(directory_path, filename)
                label_assigned = None
                for pattern, label in label_mappings.items():
                    if pattern in filename:
                        label_assigned = label
                        break
                if label_assigned is None:
                    continue  # Skip files that don't match any pattern

                # Read the CSV file
                if self.verbosity > 1:
                    print(f"Reading {filename} with label '{label_assigned}'...")
                for chunk in pd.read_csv(file_path, chunksize=100000):
                    chunk['label'] = label_assigned
                    merged_data.append(chunk)
                    total_rows += len(chunk)
                    if max_rows_number is not None and total_rows >= max_rows_number:
                        if self.verbosity > 0:
                            print(f"Reached maximum row limit ({max_rows_number}). Stopping data merge.")
                        break
                if max_rows_number is not None and total_rows >= max_rows_number:
                    break

        if not merged_data:
            if self.verbosity > 0:
                print("No data was merged. Please check the label mappings and directory path.")
            return

        # Concatenate all chunks
        merged_df = pd.concat(merged_data, ignore_index=True)
        if self.verbosity > 0:
            print(f"Total rows merged: {len(merged_df)}")

        # Update data parameters
        dataset_path = f"data/{dataset_name}/merged_data.csv"
        os.makedirs(os.path.dirname(dataset_path), exist_ok=True)
        merged_df.to_csv(dataset_path, index=False)

        new_dataset_params = {
            "test_split": 0.2,
            "apply_standardization": False,
            "apply_normalization": False,
            "apply_yeojohnson": False,
            "shuffle": True,
            "random_state": self.random_state,
            "features": "all",
            "target": "label",
            "download": False,
            "data_path": dataset_path,
            "max_rows_number": max_rows_number
        }

        self.data_params[dataset_name] = new_dataset_params
        self.params['data'] = self.data_params

        # Record modification
        self.record_modification('data', dataset_name, new_dataset_params)

        # Save updated parameters to JSON file
        self.save_params()

        if self.verbosity > 0:
            print(f"Dataset {dataset_name} has been created and added to data parameters.")

    def prepare_dataset(self, dataset_name):
        """
        Prepare the dataset according to the parameters in the JSON file.
        This includes loading the data, applying preprocessing, and splitting into train/test sets.

        Parameters:
            dataset_name: str, the name of the dataset to prepare.
        """
        if dataset_name not in self.data_params:
            if self.verbosity > 0:
                print(f"Dataset {dataset_name} is not in data parameters.")
            return None, None, None, None

        params = self.data_params[dataset_name]
        data_path = params['data_path']
        max_rows_number = params.get('max_rows_number', None)

        if self.verbosity > 0:
            print(f"Loading dataset {dataset_name} from {data_path}...")

        if params['download']:
            # Handle downloadable datasets
            if dataset_name.lower() == 'iris':
                from sklearn.datasets import load_iris
                data = load_iris(as_frame=True)
                X = data['data']
                y = data['target']
                X['label'] = y
                df = X
            elif dataset_name.lower() == 'mnist':
                from sklearn.datasets import fetch_openml
                data = fetch_openml('mnist_784', version=1, as_frame=True)
                df = data.frame
                df.rename(columns={'class': 'label'}, inplace=True)
            else:
                if self.verbosity > 0:
                    print(f"Downloadable dataset {dataset_name} is not supported.")
                return None, None, None, None

            if max_rows_number is not None:
                df = df.iloc[:max_rows_number]
                if self.verbosity > 0:
                    print(f"Applied max_rows_number: {max_rows_number}")

        else:
            # Handle non-downloadable datasets
            if not os.path.exists(data_path):
                if self.verbosity > 0:
                    print(f"Data path {data_path} does not exist.")
                return None, None, None, None
            # Read CSV file with chunks to handle large files
            df_iterator = pd.read_csv(data_path, chunksize=100000)
            df_chunks = []
            total_rows = 0
            for chunk in df_iterator:
                df_chunks.append(chunk)
                total_rows += len(chunk)
                if max_rows_number is not None and total_rows >= max_rows_number:
                    if self.verbosity > 0:
                        print(f"Reached maximum row limit ({max_rows_number}). Stopping data load.")
                    break
            df = pd.concat(df_chunks, ignore_index=True)
            if max_rows_number is not None and total_rows > max_rows_number:
                df = df.iloc[:max_rows_number]
            if self.verbosity > 0:
                print(f"Total rows loaded: {len(df)}")

        # Remove unwanted features if specified
        features = params.get('features', 'all')
        if features != 'all':
            df = df[features + [params['target']]]

        # Remove specific features if requested
        features_to_remove = params.get('features_to_remove', [])
        if features_to_remove:
            df.drop(columns=features_to_remove, inplace=True, errors='ignore')

        # Separate features and target
        X = df.drop(columns=[params['target']])
        y = df[params['target']]

        if X.isnull().any().any():
            if self.verbosity > 0:
                print("Missing values found in X. Filling missing values with column means.")
            X.fillna(X.mean(), inplace=True)

        # Encode string labels to numeric labels
        if y.dtype == 'object' or isinstance(y.iloc[0], str):
            le = LabelEncoder()
            y = le.fit_transform(y)
            if self.verbosity > 0:
                print("Encoded string labels to numeric labels.")

        # Store the number of classes
        self.n_classes = len(np.unique(y))
        if self.verbosity > 1:
            print(f"Number of classes: {self.n_classes}")

        # Apply preprocessing
        if params.get('apply_standardization', False):
            scaler = StandardScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            if self.verbosity > 0:
                print("Applied standardization.")
        if params.get('apply_normalization', False):
            scaler = MinMaxScaler()
            X = pd.DataFrame(scaler.fit_transform(X), columns=X.columns)
            if self.verbosity > 0:
                print("Applied normalization.")
        if params.get('apply_yeojohnson', False):
            pt = PowerTransformer(method='yeo-johnson')
            X = pd.DataFrame(pt.fit_transform(X), columns=X.columns)
            if self.verbosity > 0:
                print("Applied Yeo-Johnson transformation.")

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=params.get('test_split', 0.2),
            shuffle=params.get('shuffle', True),
            random_state=params.get('random_state', self.random_state)
        )
        if self.verbosity > 0:
            print("Completed train-test split.")

        return X_train, X_test, y_train, y_test

    def delete_dataset(self, dataset_name):
        """
        Delete the dataset files and remove it from data parameters.

        Parameters:
            dataset_name: str, the name of the dataset to delete.
        """
        if dataset_name not in self.data_params:
            if self.verbosity > 0:
                print(f"Dataset {dataset_name} does not exist in data parameters.")
            return

        data_path = self.data_params[dataset_name]['data_path']
        if os.path.exists(data_path):
            os.remove(data_path)
            if self.verbosity > 0:
                print(f"Deleted data file {data_path}.")
        else:
            if self.verbosity > 0:
                print(f"Data file {data_path} does not exist.")

        # Remove dataset from data parameters
        del self.data_params[dataset_name]
        self.params['data'] = self.data_params

        # Record modification
        self.record_modification('data', dataset_name, {'action': 'deleted'})

        # Save updated parameters to JSON file
        self.save_params()

        if self.verbosity > 0:
            print(f"Dataset {dataset_name} has been removed from data parameters.")

    def remove_feature(self, dataset_name, feature_name):
        """
        Remove a specific feature from the dataset.

        Parameters:
            dataset_name: str, the name of the dataset.
            feature_name: str, the name of the feature to remove.
        """
        if dataset_name not in self.data_params:
            if self.verbosity > 0:
                print(f"Dataset {dataset_name} is not in data parameters.")
            return

        # Update features_to_remove list
        features_to_remove = self.data_params[dataset_name].get('features_to_remove', [])
        if feature_name not in features_to_remove:
            features_to_remove.append(feature_name)
            self.data_params[dataset_name]['features_to_remove'] = features_to_remove

            # Record modification
            self.record_modification('data', dataset_name, {'features_to_remove': features_to_remove})

            # Save updated parameters to JSON file
            self.save_params()

            if self.verbosity > 0:
                print(f"Feature '{feature_name}' has been marked for removal in dataset {dataset_name}.")
        else:
            if self.verbosity > 0:
                print(f"Feature '{feature_name}' is already marked for removal in dataset {dataset_name}.")

    # ----------------------- Model Training Methods ----------------------- #

    def train_model(self, model_name, dataset_name, metric='f1', optimization=False, n_trials=50):
        """
        Train a specific model and evaluate it using specified metrics.

        Parameters:
            model_name: str, the name of the model to train.
            dataset_name: str, the name of the dataset to use.
            metric: str, the evaluation metric to optimize or report (default 'f1').
            optimization: bool, whether to perform Optuna optimization (default False).
            n_trials: int, number of trials for Optuna optimization (default 50).
        """
        if model_name not in self.model_params:
            if self.verbosity > 0:
                print(f"Model {model_name} is not implemented yet.")
            return None, None

        # Prepare the dataset
        X_train, X_test, y_train, y_test = self.prepare_dataset(dataset_name)
        if X_train is None:
            if self.verbosity > 0:
                print(f"Failed to prepare dataset {dataset_name}.")
            return None, None


        # Import the model class
        model_class = self.get_model_class(model_name)
        if model_class is None:
            if self.verbosity > 0:
                print(f"Model {model_name} is not available.")
            return None, None

        # Perform optimization if requested
        if optimization:
            best_params = self.optimize_model(model_name, X_train, y_train, metric, n_trials)
            # Update model parameters with the best found
            self.model_params[model_name].update(best_params)
            self.record_modification('model', model_name, best_params)
            self.save_params()
            if self.verbosity > 0:
                print(f"Optimization completed. Best parameters saved for {model_name}.")

        # Initialize the model with parameters
        model_params = self.model_params[model_name].copy()
        model_params['random_state'] = self.random_state  # Ensure random_state is set
        if model_name == 'XGBClassifier' and self.n_classes > 2:
            model_params['num_class'] = self.n_classes
            model_params['objective'] = 'multi:softprob'

        model_class = self.get_model_class(model_name)
        model = model_class(**model_params)

        # Train the model
        model.fit(X_train, y_train)

        # Make predictions
        y_pred = model.predict(X_test.values)  # Convert X_test to NumPy array

        # Evaluate the model
        evaluation = self.evaluate_model(y_test, y_pred)

        if self.verbosity > 0:
            print(f"Results for {model_name}:")
            for key, value in evaluation.items():
                print(f"{key}: {value}")

        # After training, store the model performance
        if model is not None:
            # Store feature importances if available
            feature_importances = self.get_feature_importances(model, X_train.columns)
            self.model_performances[dataset_name] = {
                'model_name': model_name,
                'evaluation': evaluation,
                'feature_importances': feature_importances
            }

        return model, evaluation

    def get_feature_importances(self, model, feature_names):
            """
            Get feature importances from the trained model.

            Parameters:
                model: trained model object.
                feature_names: list of feature names.

            Returns:
                A pandas Series of feature importances indexed by feature names.
            """
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                return pd.Series(importances, index=feature_names).sort_values(ascending=False)
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
                if importances.ndim > 1:
                    importances = importances[0]
                return pd.Series(importances, index=feature_names).sort_values(ascending=False)
            else:
                if self.verbosity > 0:
                    print("Model does not have feature_importances_ or coef_.")
                return pd.Series([0]*len(feature_names), index=feature_names)

    def train_all_models(self, dataset_name, metric='f1', optimization=False, n_trials=50, return_best=True):
        """
        Train all models and report results.

        Parameters:
            dataset_name: str, the name of the dataset to use.
            metric: str, the evaluation metric to optimize or report (default 'f1').
            optimization: bool, whether to perform Optuna optimization (default False).
            n_trials: int, number of trials for Optuna optimization (default 50).
            return_best: bool, whether to return only the best model based on the metric.

        Returns:
            If return_best is True, returns the best model and its evaluation.
            Otherwise, returns a dictionary of models and their evaluations.
        """
        models = {}
        evaluations = {}

        for model_name in self.model_params.keys():
            if self.verbosity > 0:
                print(f"\nTraining model {model_name}...")

            model, evaluation = self.train_model(model_name, dataset_name, metric, optimization, n_trials)
            if model is not None:
                models[model_name] = model
                evaluations[model_name] = evaluation

        if not models:
            if self.verbosity > 0:
                print("No models were successfully trained.")
            return None

        if return_best:
            # Find the best model based on the specified metric
            best_model_name = max(evaluations, key=lambda k: evaluations[k].get(metric, 0))
            if self.verbosity > 0:
                print(f"\nBest model based on {metric}: {best_model_name}")
            return models[best_model_name], evaluations[best_model_name]
        else:
            return models, evaluations

    def get_model_class(self, model_name):
        """
        Get the model class based on the model name.

        Parameters:
            model_name: str, the name of the model.

        Returns:
            The model class if available, otherwise None.
        """
        try:
            if model_name == 'XGBClassifier':
                from xgboost import XGBClassifier
                return XGBClassifier
            elif model_name == 'LGBMClassifier':
                from lightgbm import LGBMClassifier
                return LGBMClassifier
            elif model_name == 'ExtraTreesClassifier':
                from sklearn.ensemble import ExtraTreesClassifier
                return ExtraTreesClassifier
            elif model_name == 'HistogramGradientBoostingClassifier':
                from sklearn.experimental import enable_hist_gradient_boosting
                from sklearn.ensemble import HistGradientBoostingClassifier
                return HistGradientBoostingClassifier
            else:
                return None
        except ImportError as e:
            if self.verbosity > 0:
                print(f"Import error for {model_name}: {e}")
            return None

    def evaluate_model(self, y_true, y_pred):
        """
        Evaluate the model using various metrics.

        Parameters:
            y_true: array-like, true labels.
            y_pred: array-like, predicted labels.

        Returns:
            A dictionary of evaluation metrics.
        """
        if self.n_classes == 2:
            average_method = 'binary'
        else:
            average_method = 'macro'
    
        evaluation = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average=average_method, zero_division=0),
            'recall': recall_score(y_true, y_pred, average=average_method, zero_division=0),
            'f1': f1_score(y_true, y_pred, average=average_method, zero_division=0),
            'confusion_matrix': confusion_matrix(y_true, y_pred).tolist()
        }
        return evaluation

    def optimize_model(self, model_name, X_train, y_train, metric='f1', n_trials=50):
        """
        Optimize the model using Optuna.

        Parameters:
            model_name: str, the name of the model.
            X_train: DataFrame, training features.
            y_train: Series, training labels.
            metric: str, the evaluation metric to optimize (default 'f1').
            n_trials: int, number of trials for optimization.

        Returns:
            A dictionary of the best parameters found.
        """

        if self.verbosity > 1:
            print(f"Number of classes in training data: {self.n_classes}")
            print("Label distribution in training data:")
            print(pd.Series(y_train).value_counts())
        
        def objective(trial):
            # Define the parameter search space based on the model
            if model_name == 'XGBClassifier':
                if self.n_classes == 2:
                    objective_param = 'binary:logistic'
                    eval_metric_param = 'logloss'
                else:
                    objective_param = 'multi:softprob'
                    eval_metric_param = 'mlogloss'

                params = {
                    'objective': objective_param,
                    'eval_metric': eval_metric_param,
                    'booster': 'gbtree',
                    'n_estimators': 1000,
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'subsample': trial.suggest_uniform('subsample', 0.1, 1),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
                    'gamma': trial.suggest_loguniform('gamma', 1e-8, 1.0),
                    'min_child_weight': trial.suggest_int('min_child_weight', 1, 300),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                    'random_state': self.random_state
                }
                if self.n_classes > 2:
                    params['num_class'] = self.n_classes

                from xgboost import XGBClassifier
                model = XGBClassifier(**params)
            elif model_name == 'LGBMClassifier':
                params = {
                    'objective': 'binary',
                    'metric': 'binary_logloss',
                    'boosting_type': 'gbdt',
                    'n_estimators': 1000,
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 1, 10),
                    'subsample': trial.suggest_uniform('subsample', 0.1, 1),
                    'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.1, 1),
                    'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-8, 1.0),
                    'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-8, 1.0),
                    'random_state': self.random_state
                }
                from lightgbm import LGBMClassifier
                model = LGBMClassifier(**params)
            elif model_name == 'HistogramGradientBoostingClassifier':
                params = {
                    'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.3),
                    'max_depth': trial.suggest_int('max_depth', 1, 100),
                    'l2_regularization': trial.suggest_loguniform('l2_regularization', 1e-8, 1.0),
                    'max_bins': trial.suggest_int('max_bins', 2, 255),
                    'random_state': self.random_state
                }
                from sklearn.experimental import enable_hist_gradient_boosting
                from sklearn.ensemble import HistGradientBoostingClassifier
                model = HistGradientBoostingClassifier(**params)
            elif model_name == 'ExtraTreesClassifier':
                params = {
                    'n_estimators': 100,
                    'max_depth': trial.suggest_int('max_depth', 1, 100),
                    'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                    'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                    'max_features': trial.suggest_uniform('max_features', 0.1, 1.0),
                    'random_state': self.random_state
                }
                from sklearn.ensemble import ExtraTreesClassifier
                model = ExtraTreesClassifier(**params)
            else:
                return None

            # Cross-validation (you can adjust or add CV if needed)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_train)
            if self.n_classes == 2:
                score = f1_score(y_train, y_pred, average='binary', zero_division=0)
            else:
                score = f1_score(y_train, y_pred, average='macro', zero_division=0)

            return score

        study = optuna.create_study(direction='maximize')
        study.optimize(objective, n_trials=n_trials)

        return study.best_params

    # ----------------------- Evasion Attack Methods ----------------------- #

    def get_attack_class(self, attack_name):
        """
        Dynamically import the attack class based on the attack name.

        Parameters:
            attack_name: str, the name of the attack.

        Returns:
            The attack class if found, else None.
        """
        try:
            module = importlib.import_module(f"attacks.{attack_name}")
            attack_class = getattr(module, attack_name)
            return attack_class
        except (ImportError, AttributeError) as e:
            if self.verbosity > 0:
                print(f"Could not import attack {attack_name}: {e}")
            return None

    def get_neighborhood_class(self, neighborhood_name):
        """
        Dynamically import the neighborhood class based on the neighborhood name.

        Parameters:
            neighborhood_name: str, the name of the neighborhood.

        Returns:
            The neighborhood class if found, else None.
        """
        try:
            module = importlib.import_module(f"neighborhoods.{neighborhood_name}")
            neighborhood_class = getattr(module, neighborhood_name)
            return neighborhood_class
        except (ImportError, AttributeError) as e:
            if self.verbosity > 0:
                print(f"Could not import neighborhood {neighborhood_name}: {e}")
            return None

    def get_constraints(self, dataset_name, input_sample):
        """
        Generate constraints for the attack based on the dataset.

        Parameters:
            dataset_name: str, the name of the dataset.
            input_sample: array-like, the input sample (used to get feature length).

        Returns:
            A dictionary of constraints.
        """
        # For simplicity, we'll clip values between 1st and 99th percentiles
        X_train, _, y_train, _ = self.prepare_dataset(dataset_name)
        clip_min = X_train.quantile(0.01)
        clip_max = X_train.quantile(0.99)
        num_features = input_sample.shape[0]

        # Placeholder for equality, inequality, and categorical constraints
        constraints = {
            "equality": [],  # Example: ["x[4] = 0.0", "x[5] = 0.0"]
            "inequality": [],  # Example: ["x[2] <= x[3]"]
            "clip_min": clip_min.tolist(),
            "clip_max": clip_max.tolist(),
            "categorical": [None] * num_features  # Update if there are categorical features
        }

        return constraints

    def SLARC_cost_function(self, dataset_name):
        """
        Generate the SLARC cost function for the given dataset.

        Parameters:
            dataset_name: str, the name of the dataset.

        Returns:
            A callable cost function.
        """
        # Get the dataset
        X_train, X_test, y_train, y_test = self.prepare_dataset(dataset_name)
        all_features = X_train.columns.tolist()

        # Compute μ_k, σ_k, x_k_min, x_k_max
        mu_k = X_train[y_train == 1].mean()
        sigma_k = X_train[y_train == 1].std().replace(0, 1e-6)  # Avoid division by zero
        x_k_min = X_train[y_train == 1].quantile(0.01)
        x_k_max = X_train[y_train == 1].quantile(0.99)

        # Get omega_k_dict based on feature importances or correlations
        if dataset_name in self.model_performances:
            # Use feature importances from the best model
            feature_importances = self.model_performances[dataset_name]['feature_importances']
            omega_k_series = feature_importances / feature_importances.sum()
        else:
            # Use correlation coefficients with the label
            omega_k_series = self.compute_feature_correlations(X_train, y_train)
            omega_k_series = omega_k_series.abs() / omega_k_series.abs().sum()

        # Ensure omega_k_series has all features
        missing_features = set(all_features) - set(omega_k_series.index)
        for feature in missing_features:
            omega_k_series[feature] = 0
        omega_k_series = omega_k_series[all_features]

        def cost_function(x1, x0):
            x1_series = pd.Series(x1, index=all_features)
            x0_series = pd.Series(x0, index=all_features)

            delta = np.abs(x1_series - x0_series)

            sgn_term = np.sign(np.abs(x1_series - mu_k) - np.abs(x0_series - mu_k))

            # Compute rho_k_mu
            denom_mu = mu_k - x_k_min
            denom_mu = denom_mu.replace(0, 1e-6)  # Avoid division by zero
            rho_k_mu = (x_k_max - mu_k) / denom_mu
            rho_k_mu = rho_k_mu.replace([np.inf, -np.inf], 0).fillna(0)

            # Compute rho_k_x1
            denom_x1 = x1_series - x_k_min
            denom_x1 = denom_x1.replace(0, 1e-6)  # Avoid division by zero
            rho_k_x1 = (x_k_max - x1_series) / denom_x1
            rho_k_x1 = rho_k_x1.replace([np.inf, -np.inf], 0).fillna(0)

            # Compute epsilon_k
            condition = (rho_k_mu - 1) * (rho_k_x1 - 1)
            epsilon_k_values = np.where(condition >= 0, 1, np.maximum(rho_k_mu, 1 / rho_k_mu))
            epsilon_k = pd.Series(epsilon_k_values, index=all_features)
            epsilon_k = epsilon_k.replace([np.inf, -np.inf], 0).fillna(0)

            # Compute SLARC_k
            slarc_k = (omega_k_series / sigma_k) * epsilon_k * delta * sgn_term
            slarc_k = slarc_k.replace([np.inf, -np.inf], 0).fillna(0)

            total_SLARC = slarc_k.sum()
            return total_SLARC

        return cost_function

    def compute_feature_correlations(self, X, y):
        """
        Compute the Pearson correlation coefficients between features and the label.

        Parameters:
            X: DataFrame, features.
            y: Series, labels.

        Returns:
            A Series of correlation coefficients.
        """
        correlations = {}
        for feature in X.columns:
            corr, _ = pearsonr(X[feature], y)
            correlations[feature] = corr
        return pd.Series(correlations)

    
    def run_attack(self, attack_name, model_wrapper, input_sample, dataset_name, cost_function='SLARC_cost_function', **kwargs):
        """
        Run a specific attack on an input sample and record cost history.

        Parameters:
            attack_name: str, the name of the attack to run.
            model_wrapper: object, a wrapper of the trained model with predict methods.
            input_sample: array-like, the input sample to attack.
            dataset_name: str, the name of the dataset (used for parameter retrieval).
            cost_function: str or callable, the cost function to use.
            **kwargs: additional parameters to override default attack parameters.

        Returns:
            A tuple containing the attack result and the cost history.
        """
        if attack_name not in self.attack_params:
            if self.verbosity > 0:
                print(f"Attack {attack_name} is not implemented yet.")
            return None

        # Get attack parameters
        attack_params = self.attack_params[attack_name].copy()
        attack_params.update(kwargs)

        # Get neighborhood parameters
        neighborhood_name = attack_params.get('neighborhood', 'Radar')
        if neighborhood_name not in self.neighborhood_params:
            if self.verbosity > 0:
                print(f"Neighborhood {neighborhood_name} is not implemented yet.")
            return None
        neighborhood_params = self.neighborhood_params[neighborhood_name]

        # Prepare the neighborhood
        neighborhood_class = self.get_neighborhood_class(neighborhood_name)
        if neighborhood_class is None:
            if self.verbosity > 0:
                print(f"Neighborhood {neighborhood_name} class could not be found.")
            return None

        # Get constraints for the attack
        constraints = self.get_constraints(dataset_name, input_sample)

        # Initialize the neighborhood
        neighborhood = neighborhood_class(constraints=constraints, **neighborhood_params)

        # Prepare the cost function
        if cost_function == 'SLARC_cost_function':
            cost_function_callable = self.SLARC_cost_function(dataset_name)
        elif cost_function == 'L2_norm':
            cost_function_callable = self.L2_norm
        elif callable(cost_function):
            cost_function_callable = cost_function
        else:
            if self.verbosity > 0:
                print("Invalid cost function.")
            return None

        # Import the attack class dynamically
        attack_class = self.get_attack_class(attack_name)
        if attack_class is None:
            if self.verbosity > 0:
                print(f"Attack class {attack_name} could not be found.")
            return None

        # Initialize the attack
        attack = attack_class(estimator=model_wrapper, verbose=self.verbosity)
        attack.run_params.update(attack_params)  # Update attack parameters

        # Run the attack
        try:
            t0 = time.time()
            result = attack.run(
                input=input_sample,
                cost_function=cost_function_callable,
                neighborhood=neighborhood,
                **attack_params
            )
            time_taken = time.time() - t0
            if self.verbosity > 0:
                print(f"Attack {attack_name} completed in {time_taken:.2f} seconds.")

            # Extract cost history from the attack
            cost_history = attack.heuristic_history.copy()  # Assuming this attribute exists

            return result, cost_history
        except Exception as e:
            if self.verbosity > 0:
                print(f"Error during attack {attack_name}: {e}")
            return None

    def compare_attacks(self, attack_names, model_wrapper, X_attack_samples, dataset_name, cost_function='SLARC_cost_function', parallel_processing=None, n_jobs=-1, **kwargs):
        """
        Compare multiple attacks on a set of samples, with optional parallel processing.

        Parameters:
            attack_names: list of str, names of attacks to compare.
            model_wrapper: object, a wrapper of the trained model with predict methods.
            X_attack_samples: array-like, samples to attack.
            dataset_name: str, the name of the dataset.
            cost_function: str or callable, the cost function to use.
            parallel_processing: bool, whether to use parallel processing (default True if len(X_attack_samples) > 1).
            n_jobs: int, number of jobs to run in parallel (default -1, uses all processors).
            **kwargs: additional parameters to override default attack parameters.

        Returns:
            A dictionary with attack results, performance metrics, and cost histories.
        """
        if parallel_processing is None:
            parallel_processing = len(X_attack_samples) > 1  # Default to True if more than one sample

        results = {}
        attack_times = {name: 0 for name in attack_names}
        best_counts = {name: 0 for name in attack_names}
        best_absolute_scores = {name: [] for name in attack_names}
        cost_histories = {name: {} for name in attack_names}  # Store cost histories

        if parallel_processing:
            if self.verbosity > 0:
                print("Running attacks in parallel...")

            # Define a helper function to run attacks on a single sample
            def run_attacks_on_sample(idx, x):
                sample_results = []
                sample_histories = {name: None for name in attack_names}
                best_cost = np.inf
                best_attack = None

                # If there are conflicts between **kwargs and attack parameters, **kwargs will override
                # Or else, we will get an error "multiple values for ..."
                #if 'cost_function' in kwargs:
                #    cost_function = kwargs.pop('cost_function')

                print('kwargs are ', kwargs)

                for attack_name in attack_names:
                    result = self.run_attack(
                        attack_name=attack_name,
                        model_wrapper=model_wrapper,
                        input_sample=x,
                        dataset_name=dataset_name,
                        #cost_function=cost_function,
                        **kwargs
                    )
                    if result is None:
                        continue

                    attack_result, cost_history = result  # Unpack result and cost history
                    x_adv = attack_result[0]
                    rel_cost = self.SLARC_cost_function(dataset_name)(x_adv, x)
                    abs_cost = self.SLARC_cost_function(dataset_name)(x_adv, self.mu_k[dataset_name])
                    y_pred = model_wrapper.predict(x_adv)

                    # Record results
                    res = {
                        "sample": idx + 1,
                        "attack": attack_name,
                        "success": int(y_pred[0] != model_wrapper.predict(x)[0]),  # Misclassification
                        "relative_cost": rel_cost,
                        "absolute_cost": abs_cost,
                        "x_adv": x_adv
                    }

                    # Update best cost and attack
                    if abs_cost < best_cost:
                        best_cost = abs_cost
                        best_attack = attack_name

                    # Store individual attack results and cost histories
                    sample_results.append(res)
                    sample_histories[attack_name] = cost_history

                return sample_results, sample_histories, best_attack

            # Run attacks in parallel
            parallel_results = Parallel(n_jobs=n_jobs)(
                delayed(run_attacks_on_sample)(idx, x) for idx, x in enumerate(X_attack_samples)
            )

            # Aggregate results
            for idx, (sample_results, sample_histories, best_attack) in enumerate(parallel_results):
                if self.verbosity > 0:
                    print(f"Best attack for sample {idx + 1}: {best_attack}")

                # Update results and histories
                for res in sample_results:
                    attack_name = res['attack']
                    if attack_name not in results:
                        results[attack_name] = []
                    results[attack_name].append(res)
                    best_absolute_scores[attack_name].append(res['absolute_cost'])

                    # Store cost history
                    cost_histories[attack_name][idx] = sample_histories[attack_name]

                # Increment best attack count
                if best_attack:
                    best_counts[best_attack] += 1

        else:
            if self.verbosity > 0:
                print("Running attacks sequentially...")

            for idx, x in enumerate(X_attack_samples):
                if self.verbosity > 0:
                    print(f"Running attacks for sample {idx + 1}...")

                best_cost = np.inf
                best_attack = None

                for attack_name in attack_names:
                    result = self.run_attack(
                        attack_name=attack_name,
                        model_wrapper=model_wrapper,
                        input_sample=x,
                        dataset_name=dataset_name,
                        cost_function=cost_function,
                        **kwargs
                    )
                    if result is None:
                        continue

                    attack_result, cost_history = result  # Unpack result and cost history
                    x_adv = attack_result[0]
                    rel_cost = self.SLARC_cost_function(dataset_name)(x_adv, x)
                    abs_cost = self.SLARC_cost_function(dataset_name)(x_adv, self.mu_k[dataset_name])
                    y_pred = model_wrapper.predict(x_adv)

                    # Record results
                    res = {
                        "sample": idx + 1,
                        "attack": attack_name,
                        "success": int(y_pred[0] != model_wrapper.predict(x)[0]),  # Misclassification
                        "relative_cost": rel_cost,
                        "absolute_cost": abs_cost,
                        "x_adv": x_adv
                    }

                    # Update best cost and attack
                    if abs_cost < best_cost:
                        best_cost = abs_cost
                        best_attack = attack_name

                    # Update attack times and scores
                    # Assuming attack_result[2] is time taken
                    attack_times[attack_name] += attack_result[2] if len(attack_result) > 2 else 0
                    best_absolute_scores[attack_name].append(abs_cost)

                    # Store individual attack results
                    if attack_name not in results:
                        results[attack_name] = []
                    results[attack_name].append(res)

                    # Store cost history
                    if attack_name not in cost_histories:
                        cost_histories[attack_name] = {}
                    cost_histories[attack_name][idx] = cost_history

                if self.verbosity > 0:
                    print(f"Best attack for sample {idx + 1}: {best_attack}")

                # Increment best attack count
                if best_attack:
                    best_counts[best_attack] += 1

        # Store attack performances and histories
        self.attack_performances[dataset_name] = {
            'results': results,
            'attack_times': attack_times,
            'best_counts': best_counts,
            'best_absolute_scores': best_absolute_scores,
            'cost_histories': cost_histories
        }

        return self.attack_performances[dataset_name]

    def plot_cost_evolution(self, dataset_name, attack_name, samples_indices=None):
        """
        Plot the cost evolution for all specified samples for a given attack.

        Parameters:
            dataset_name: str, the name of the dataset.
            attack_name: str, the name of the attack.
            samples_indices: list of int, indices of samples to plot (default is all samples).

        Returns:
            None. Displays the plot.
        """
        if dataset_name not in self.attack_performances:
            if self.verbosity > 0:
                print(f"No attack performances recorded for dataset {dataset_name}.")
            return

        attack_data = self.attack_performances[dataset_name]
        cost_histories = attack_data.get('cost_histories', {})
        if attack_name not in cost_histories:
            if self.verbosity > 0:
                print(f"No cost histories found for attack {attack_name} in dataset {dataset_name}.")
            return

        attack_histories = cost_histories[attack_name]

        if not attack_histories:
            if self.verbosity > 0:
                print(f"No cost histories recorded for attack {attack_name}.")
            return

        # Determine samples to plot
        if samples_indices is None:
            samples_indices = list(attack_histories.keys())

        # Plot cost evolutions
        plt.figure(figsize=(10, 6))
        for idx in samples_indices:
            history = attack_histories.get(idx)
            if history is not None:
                costs = [sample[1] for sample in history]
                plt.plot(costs, label=f'Sample {idx + 1}')
            else:
                if self.verbosity > 0:
                    print(f"No cost history for sample {idx + 1} in attack {attack_name}.")

        plt.title(f'Cost Evolution for {attack_name} on Dataset {dataset_name}')
        plt.xlabel('Iteration')
        plt.ylabel('Cost')
        plt.legend()
        plt.grid(True)
        plt.show()

    # ----------------------- Parameter Handling Methods ----------------------- #

    def update_params(self, category, name, new_params):
        """
        Update parameters in the JSON file and record the modifications.

        Parameters:
            category: str, the category of the parameters (e.g., 'attack')
            name: str, the name of the parameter set (e.g., 'SimulatedAnnealing')
            new_params: dict, the new parameters to update
        """
        if category in self.params and name in self.params[category]:
            # Record the original parameters for rollback
            original_params = self.params[category][name].copy()
            self.record_modification(category, name, original_params)

            # Update parameters
            self.params[category][name].update(new_params)

            # Save updated parameters to JSON file
            self.save_params()

            if self.verbosity > 0:
                print(f"Parameters for {name} in category {category} have been updated.")
        else:
            if self.verbosity > 0:
                print(f"Category {category} or name {name} does not exist.")

    def record_modification(self, category, name, params):
        """
        Record modifications to the parameters in the modifications log file.

        Parameters:
            category: str, the category of the parameters
            name: str, the name of the parameter set
            params: dict, the parameters that were modified
        """
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(self.modifications_log, 'a') as log_file:
            log_file.write(f"[{timestamp}] Modified {category} -> {name}:\n")
            log_file.write(json.dumps(params, indent=4))
            log_file.write('\n\n')

    def save_params(self):
        """
        Save the current parameters to the JSON file.
        """
        with open(self.params_file, 'w') as file:
            json.dump(self.params, file, indent=4)

    def rollback_json(self, category=None, name=None):
        """
        Rollback the JSON file to its default values or revert specific sections.

        Parameters:
            category: str, optional, the category to rollback (e.g., 'attack')
            name: str, optional, the name within the category to rollback (e.g., 'SimulatedAnnealing')
        """
        if category is None:
            # Rollback the entire JSON file
            self.params = json.loads(json.dumps(self.default_params))  # Deep copy
            self.save_params()
            if self.verbosity > 0:
                print("JSON file has been rolled back to default values.")
        elif name is None:
            # Rollback a specific category
            if category in self.default_params:
                self.params[category] = json.loads(json.dumps(self.default_params[category]))
                self.save_params()
                if self.verbosity > 0:
                    print(f"Category {category} has been rolled back to default values.")
            else:
                if self.verbosity > 0:
                    print(f"Category {category} does not exist in default parameters.")
        else:
            # Rollback a specific name within a category
            if category in self.default_params and name in self.default_params[category]:
                self.params[category][name] = json.loads(json.dumps(self.default_params[category][name]))
                self.save_params()
                if self.verbosity > 0:
                    print(f"{name} in category {category} has been rolled back to default values.")
            else:
                if self.verbosity > 0:
                    print(f"{name} in category {category} does not exist in default parameters.")

    # ----------------------- User Access to Random State ----------------------- #

    def set_random_state(self, random_state):
        """
        Set the global random state.

        Parameters:
            random_state: int, the random state to set.
        """
        self.random_state = random_state
        if self.verbosity > 0:
            print(f"Random state set to {random_state}.")

    def get_random_state(self):
        """
        Get the current global random state.

        Returns:
            int, the current random state.
        """
        return self.random_state

# Example usage:
def master_example():
    master = Master(verbosity=2)

    # Example: Change random state
    master.set_random_state(123)

    # Example: Set verbosity
    master.verbosity = 1  # Control output verbosity

    # Example: Merge CSV files and create a dataset
    label_mappings = {
        'Mirai-UDPPlain': 'Mirai',
        'Mirai-GREIP_flood': 'Mirai',
        'DDoS-SlowLoris': 'SlowLoris'
    }
    master.merge_csv_files(
        dataset_name='CICIoT23',
        directory_path='data/CICIoT23',
        label_mappings=label_mappings,
        max_rows_number=500000  # Optional: limit to first 500,000 rows
    )

    # Example: Remove a feature from the dataset
    master.remove_feature('CICIoT23', 'Number')

    # Example: Prepare the dataset
    X_train, X_test, y_train, y_test = master.prepare_dataset('CICIoT23')

    # Example: Train a specific model without optimization
    model, evaluation = master.train_model('XGBClassifier', 'CICIoT23')

    # Example: Train all models with optimization and get the best model
    best_model, best_evaluation = master.train_all_models('CICIoT23', optimization=True, n_trials=10)