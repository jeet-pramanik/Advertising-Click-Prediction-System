"""
Data Loading Utilities
Functions for loading and initial data handling
"""

import pandas as pd
import numpy as np
from pathlib import Path


class DataLoader:
    """Class for loading and managing advertising data"""
    
    def __init__(self, data_path=None):
        """
        Initialize DataLoader
        
        Parameters:
        -----------
        data_path : str or Path
            Path to the data file
        """
        self.data_path = data_path
        self.data = None
        
    def load_csv(self, filepath):
        """
        Load data from CSV file
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        pandas.DataFrame
            Loaded data
        """
        try:
            self.data = pd.read_csv(filepath)
            print(f"✓ Data loaded successfully from {filepath}")
            print(f"  Shape: {self.data.shape}")
            return self.data
        except FileNotFoundError:
            print(f"✗ Error: File not found at {filepath}")
            return None
        except Exception as e:
            print(f"✗ Error loading data: {str(e)}")
            return None
    
    def get_basic_info(self):
        """
        Get basic information about the loaded dataset
        
        Returns:
        --------
        dict
            Dictionary containing basic dataset information
        """
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return None
        
        info = {
            'shape': self.data.shape,
            'n_rows': self.data.shape[0],
            'n_columns': self.data.shape[1],
            'columns': list(self.data.columns),
            'dtypes': self.data.dtypes.to_dict(),
            'missing_values': self.data.isnull().sum().to_dict(),
            'missing_percentage': (self.data.isnull().sum() / len(self.data) * 100).to_dict(),
            'memory_usage': self.data.memory_usage(deep=True).sum() / 1024**2  # MB
        }
        
        return info
    
    def display_summary(self):
        """Display comprehensive summary of the dataset"""
        if self.data is None:
            print("✗ No data loaded. Please load data first.")
            return
        
        print("\n" + "="*80)
        print("DATASET SUMMARY")
        print("="*80)
        
        info = self.get_basic_info()
        
        print(f"\n📊 Dataset Dimensions:")
        print(f"   Rows: {info['n_rows']:,}")
        print(f"   Columns: {info['n_columns']}")
        print(f"   Memory Usage: {info['memory_usage']:.2f} MB")
        
        print(f"\n📋 Column Information:")
        for col, dtype in info['dtypes'].items():
            missing = info['missing_values'][col]
            missing_pct = info['missing_percentage'][col]
            if missing > 0:
                print(f"   {col:30s} | {str(dtype):15s} | Missing: {missing:4d} ({missing_pct:.1f}%)")
            else:
                print(f"   {col:30s} | {str(dtype):15s} | Complete ✓")
        
        # Target variable analysis
        if 'clicked' in self.data.columns:
            print(f"\n🎯 Target Variable (clicked):")
            click_counts = self.data['clicked'].value_counts()
            click_rate = self.data['clicked'].mean()
            print(f"   Not Clicked (0): {click_counts.get(0, 0):,} ({(1-click_rate)*100:.2f}%)")
            print(f"   Clicked (1):     {click_counts.get(1, 0):,} ({click_rate*100:.2f}%)")
            print(f"   Click-Through Rate: {click_rate*100:.2f}%")
            
            # Check for class imbalance
            imbalance_ratio = click_counts.max() / click_counts.min() if len(click_counts) > 1 else 1
            print(f"   Imbalance Ratio: {imbalance_ratio:.2f}:1")
            if imbalance_ratio > 3:
                print("   ⚠️  Significant class imbalance detected!")
        
        print("\n" + "="*80)
    
    def get_numerical_features(self):
        """Get list of numerical feature columns"""
        if self.data is None:
            return []
        return self.data.select_dtypes(include=[np.number]).columns.tolist()
    
    def get_categorical_features(self):
        """Get list of categorical feature columns"""
        if self.data is None:
            return []
        return self.data.select_dtypes(include=['object', 'category']).columns.tolist()
    
    def get_feature_types(self):
        """
        Categorize features into numerical and categorical
        
        Returns:
        --------
        dict
            Dictionary with 'numerical' and 'categorical' feature lists
        """
        return {
            'numerical': self.get_numerical_features(),
            'categorical': self.get_categorical_features()
        }


def load_advertising_data(filepath='data/raw/advertising_data.csv'):
    """
    Convenience function to load advertising data
    
    Parameters:
    -----------
    filepath : str
        Path to the data file
        
    Returns:
    --------
    pandas.DataFrame
        Loaded advertising data
    """
    loader = DataLoader()
    data = loader.load_csv(filepath)
    
    if data is not None:
        loader.display_summary()
    
    return data


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    data = loader.load_csv('data/raw/advertising_data.csv')
    
    if data is not None:
        loader.display_summary()
        
        print("\nFeature Types:")
        feature_types = loader.get_feature_types()
        print(f"Numerical features: {feature_types['numerical']}")
        print(f"Categorical features: {feature_types['categorical']}")
