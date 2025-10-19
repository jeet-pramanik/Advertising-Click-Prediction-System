"""
Preprocessing Pipeline Module
Handles data cleaning, encoding, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
import joblib
import warnings
warnings.filterwarnings('ignore')


class PreprocessingPipeline:
    """Complete preprocessing pipeline for advertising data"""
    
    def __init__(self):
        self.numerical_imputer = SimpleImputer(strategy='median')
        self.categorical_imputer = SimpleImputer(strategy='most_frequent')
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.numerical_features = None
        self.categorical_features = None
        
    def identify_feature_types(self, df, target_column='clicked'):
        """
        Identify numerical and categorical features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str
            Name of target column to exclude
            
        Returns:
        --------
        tuple
            (numerical_features, categorical_features)
        """
        # Exclude target and identifier columns
        exclude_columns = [target_column, 'user_id', 'date']
        feature_columns = [col for col in df.columns if col not in exclude_columns]
        
        # Identify numerical features
        numerical_features = df[feature_columns].select_dtypes(
            include=[np.number]
        ).columns.tolist()
        
        # Identify categorical features
        categorical_features = df[feature_columns].select_dtypes(
            include=['object', 'category']
        ).columns.tolist()
        
        self.numerical_features = numerical_features
        self.categorical_features = categorical_features
        
        return numerical_features, categorical_features
    
    def handle_missing_values(self, df, fit=True):
        """
        Handle missing values in numerical and categorical features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the imputers (True for training data)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with imputed values
        """
        df = df.copy()
        
        # Handle numerical missing values
        if self.numerical_features:
            if fit:
                df[self.numerical_features] = self.numerical_imputer.fit_transform(
                    df[self.numerical_features]
                )
                print(f"✓ Fitted numerical imputer (strategy: median)")
            else:
                df[self.numerical_features] = self.numerical_imputer.transform(
                    df[self.numerical_features]
                )
        
        # Handle categorical missing values
        if self.categorical_features:
            if fit:
                df[self.categorical_features] = self.categorical_imputer.fit_transform(
                    df[self.categorical_features]
                )
                print(f"✓ Fitted categorical imputer (strategy: most_frequent)")
            else:
                df[self.categorical_features] = self.categorical_imputer.transform(
                    df[self.categorical_features]
                )
        
        return df
    
    def encode_categorical_features(self, df, fit=True, encoding_type='label'):
        """
        Encode categorical features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the encoders (True for training data)
        encoding_type : str
            'label' for label encoding, 'onehot' for one-hot encoding
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with encoded features
        """
        df = df.copy()
        
        if encoding_type == 'label':
            for col in self.categorical_features:
                if col in df.columns:
                    if fit:
                        le = LabelEncoder()
                        df[col] = le.fit_transform(df[col].astype(str))
                        self.label_encoders[col] = le
                        print(f"✓ Label encoded: {col}")
                    else:
                        if col in self.label_encoders:
                            # Handle unseen categories
                            le = self.label_encoders[col]
                            df[col] = df[col].astype(str).apply(
                                lambda x: le.transform([x])[0] if x in le.classes_ else -1
                            )
        
        elif encoding_type == 'onehot':
            # For one-hot encoding
            df = pd.get_dummies(df, columns=self.categorical_features, drop_first=True)
            print(f"✓ One-hot encoded categorical features")
        
        return df
    
    def scale_features(self, df, fit=True):
        """
        Scale numerical features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        fit : bool
            Whether to fit the scaler (True for training data)
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with scaled features
        """
        df = df.copy()
        
        if self.numerical_features:
            numerical_cols = [col for col in self.numerical_features if col in df.columns]
            
            if fit:
                df[numerical_cols] = self.scaler.fit_transform(df[numerical_cols])
                print(f"✓ Fitted and scaled numerical features")
            else:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
                print(f"✓ Scaled numerical features")
        
        return df
    
    def prepare_data(self, df, target_column='clicked', fit=True, encoding_type='label'):
        """
        Complete preprocessing pipeline
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        target_column : str
            Name of target column
        fit : bool
            Whether to fit transformers (True for training data)
        encoding_type : str
            Type of encoding for categorical features
            
        Returns:
        --------
        tuple
            (X, y) features and target
        """
        print("\n" + "="*80)
        print("PREPROCESSING PIPELINE")
        print("="*80)
        
        df = df.copy()
        
        # Identify feature types (only on first fit)
        if fit:
            self.identify_feature_types(df, target_column)
            print(f"\n📊 Feature Types Identified:")
            print(f"   Numerical features: {len(self.numerical_features)}")
            print(f"   Categorical features: {len(self.categorical_features)}")
        
        # Handle missing values
        print(f"\n🔧 Handling Missing Values...")
        df = self.handle_missing_values(df, fit=fit)
        
        # Encode categorical features
        print(f"\n🔤 Encoding Categorical Features...")
        df = self.encode_categorical_features(df, fit=fit, encoding_type=encoding_type)
        
        # Separate features and target
        if target_column in df.columns:
            y = df[target_column]
            X = df.drop(columns=[target_column])
            
            # Remove non-feature columns
            cols_to_drop = ['user_id', 'date']
            X = X.drop(columns=[col for col in cols_to_drop if col in X.columns])
        else:
            y = None
            X = df.drop(columns=['user_id', 'date'], errors='ignore')
        
        # Update numerical features list after encoding
        if fit:
            self.numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
        
        # Scale features
        print(f"\n⚖️  Scaling Features...")
        X = self.scale_features(X, fit=fit)
        
        # Store feature names
        if fit:
            self.feature_names = X.columns.tolist()
        
        print(f"\n✓ Preprocessing completed")
        print(f"   Feature shape: {X.shape}")
        if y is not None:
            print(f"   Target shape: {y.shape}")
        print("="*80)
        
        return X, y
    
    def save_pipeline(self, filepath='models/preprocessor.pkl'):
        """
        Save the preprocessing pipeline
        
        Parameters:
        -----------
        filepath : str
            Path to save the pipeline
        """
        pipeline_data = {
            'numerical_imputer': self.numerical_imputer,
            'categorical_imputer': self.categorical_imputer,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'numerical_features': self.numerical_features,
            'categorical_features': self.categorical_features
        }
        
        joblib.dump(pipeline_data, filepath)
        print(f"✓ Preprocessing pipeline saved to {filepath}")
    
    @classmethod
    def load_pipeline(cls, filepath='models/preprocessor.pkl'):
        """
        Load a saved preprocessing pipeline
        
        Parameters:
        -----------
        filepath : str
            Path to the saved pipeline
            
        Returns:
        --------
        PreprocessingPipeline
            Loaded pipeline object
        """
        pipeline_data = joblib.load(filepath)
        
        pipeline = cls()
        pipeline.numerical_imputer = pipeline_data['numerical_imputer']
        pipeline.categorical_imputer = pipeline_data['categorical_imputer']
        pipeline.scaler = pipeline_data['scaler']
        pipeline.label_encoders = pipeline_data['label_encoders']
        pipeline.feature_names = pipeline_data['feature_names']
        pipeline.numerical_features = pipeline_data['numerical_features']
        pipeline.categorical_features = pipeline_data['categorical_features']
        
        print(f"✓ Preprocessing pipeline loaded from {filepath}")
        return pipeline


def split_data(X, y, test_size=0.2, random_state=42, stratify=True):
    """
    Split data into training and testing sets
    
    Parameters:
    -----------
    X : pandas.DataFrame or numpy.ndarray
        Features
    y : pandas.Series or numpy.ndarray
        Target
    test_size : float
        Proportion of test set
    random_state : int
        Random seed for reproducibility
    stratify : bool
        Whether to stratify split by target
        
    Returns:
    --------
    tuple
        X_train, X_test, y_train, y_test
    """
    stratify_param = y if stratify else None
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify_param
    )
    
    print(f"\n✓ Data split completed:")
    print(f"   Training set: {X_train.shape[0]:,} samples ({(1-test_size)*100:.0f}%)")
    print(f"   Test set:     {X_test.shape[0]:,} samples ({test_size*100:.0f}%)")
    
    return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    print("Preprocessing Pipeline Module")
    print("Use PreprocessingPipeline() to create a preprocessing pipeline")
