"""
Model Training Module
Train and tune machine learning models for click prediction
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import warnings
warnings.filterwarnings('ignore')


class ModelTrainer:
    """Class for training and tuning ML models"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        
    def create_baseline_model(self, y_train):
        """
        Create a simple baseline model (majority class predictor)
        
        Parameters:
        -----------
        y_train : array-like
            Training target values
            
        Returns:
        --------
        float
            Baseline accuracy (proportion of majority class)
        """
        majority_class = pd.Series(y_train).mode()[0]
        baseline_accuracy = (y_train == majority_class).mean()
        
        print(f"\n📊 Baseline Model (Majority Class Predictor):")
        print(f"   Predicts class: {majority_class}")
        print(f"   Baseline accuracy: {baseline_accuracy:.4f}")
        
        return baseline_accuracy
    
    def train_logistic_regression(self, X_train, y_train, tune=True):
        """
        Train Logistic Regression model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        model
            Trained model
        """
        print("\n" + "="*80)
        print("LOGISTIC REGRESSION")
        print("="*80)
        
        if tune:
            print("🔍 Performing hyperparameter tuning...")
            
            param_grid = {
                'C': [0.001, 0.01, 0.1, 1, 10, 100],
                'penalty': ['l1', 'l2'],
                'solver': ['liblinear', 'saga'],
                'max_iter': [1000]
            }
            
            lr = LogisticRegression(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                lr,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
        else:
            print("🚀 Training with default parameters...")
            model = LogisticRegression(
                random_state=self.random_state,
                max_iter=1000,
                solver='lbfgs'
            )
            model.fit(X_train, y_train)
            print("✓ Model trained successfully")
        
        self.models['Logistic Regression'] = model
        return model
    
    def train_decision_tree(self, X_train, y_train, tune=True):
        """
        Train Decision Tree model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        model
            Trained model
        """
        print("\n" + "="*80)
        print("DECISION TREE")
        print("="*80)
        
        if tune:
            print("🔍 Performing hyperparameter tuning...")
            
            param_grid = {
                'max_depth': [5, 10, 15, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'criterion': ['gini', 'entropy']
            }
            
            dt = DecisionTreeClassifier(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                dt,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
        else:
            print("🚀 Training with default parameters...")
            model = DecisionTreeClassifier(random_state=self.random_state)
            model.fit(X_train, y_train)
            print("✓ Model trained successfully")
        
        self.models['Decision Tree'] = model
        return model
    
    def train_random_forest(self, X_train, y_train, tune=True):
        """
        Train Random Forest model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        model
            Trained model
        """
        print("\n" + "="*80)
        print("RANDOM FOREST")
        print("="*80)
        
        if tune:
            print("🔍 Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5],
                'min_samples_leaf': [1, 2],
                'max_features': ['sqrt', 'log2']
            }
            
            rf = RandomForestClassifier(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                rf,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
        else:
            print("🚀 Training with default parameters...")
            model = RandomForestClassifier(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
            model.fit(X_train, y_train)
            print("✓ Model trained successfully")
        
        self.models['Random Forest'] = model
        return model
    
    def train_gradient_boosting(self, X_train, y_train, tune=True):
        """
        Train Gradient Boosting model
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        model
            Trained model
        """
        print("\n" + "="*80)
        print("GRADIENT BOOSTING")
        print("="*80)
        
        if tune:
            print("🔍 Performing hyperparameter tuning...")
            
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'min_samples_split': [2, 5],
                'subsample': [0.8, 1.0]
            }
            
            gb = GradientBoostingClassifier(random_state=self.random_state)
            
            grid_search = GridSearchCV(
                gb,
                param_grid,
                cv=5,
                scoring='roc_auc',
                n_jobs=-1,
                verbose=1
            )
            
            grid_search.fit(X_train, y_train)
            
            model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV ROC-AUC score: {grid_search.best_score_:.4f}")
        else:
            print("🚀 Training with default parameters...")
            model = GradientBoostingClassifier(random_state=self.random_state)
            model.fit(X_train, y_train)
            print("✓ Model trained successfully")
        
        self.models['Gradient Boosting'] = model
        return model
    
    def handle_imbalance_smote(self, X_train, y_train):
        """
        Handle class imbalance using SMOTE
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
            
        Returns:
        --------
        tuple
            (X_resampled, y_resampled)
        """
        print("\n🔄 Applying SMOTE for class imbalance...")
        
        smote = SMOTE(random_state=self.random_state)
        X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
        
        print(f"   Original samples: {len(y_train)}")
        print(f"   Resampled samples: {len(y_resampled)}")
        print(f"   Class distribution after SMOTE:")
        print(f"   Class 0: {(y_resampled == 0).sum()}")
        print(f"   Class 1: {(y_resampled == 1).sum()}")
        
        return X_resampled, y_resampled
    
    def train_all_models(self, X_train, y_train, tune=False):
        """
        Train all available models
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target
        tune : bool
            Whether to perform hyperparameter tuning
            
        Returns:
        --------
        dict
            Dictionary of trained models
        """
        print("\n" + "="*80)
        print("TRAINING ALL MODELS")
        print("="*80)
        
        # Baseline
        self.create_baseline_model(y_train)
        
        # Train all models
        self.train_logistic_regression(X_train, y_train, tune=tune)
        self.train_decision_tree(X_train, y_train, tune=tune)
        self.train_random_forest(X_train, y_train, tune=tune)
        self.train_gradient_boosting(X_train, y_train, tune=tune)
        
        print("\n" + "="*80)
        print(f"✓ All {len(self.models)} models trained successfully")
        print("="*80)
        
        return self.models
    
    def save_model(self, model, model_name, filepath=None):
        """
        Save a trained model
        
        Parameters:
        -----------
        model : trained model
            Model to save
        model_name : str
            Name of the model
        filepath : str
            Path to save the model (optional)
        """
        if filepath is None:
            filepath = f"models/{model_name.lower().replace(' ', '_')}.pkl"
        
        joblib.dump(model, filepath)
        print(f"✓ Model saved to {filepath}")
    
    @staticmethod
    def load_model(filepath):
        """
        Load a saved model
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        model
            Loaded model
        """
        model = joblib.load(filepath)
        print(f"✓ Model loaded from {filepath}")
        return model


if __name__ == "__main__":
    print("Model Training Module")
    print("Use ModelTrainer() to train machine learning models")
