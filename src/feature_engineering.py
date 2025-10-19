"""
Feature Engineering Module
Creates new features and transforms existing ones for better model performance
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class FeatureEngineer:
    """Class for creating and transforming features"""
    
    def __init__(self):
        self.label_encoders = {}
        
    def create_age_groups(self, df, age_column='age'):
        """
        Create age group categories
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        age_column : str
            Name of age column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with age_group column added
        """
        df = df.copy()
        
        bins = [0, 25, 35, 45, 55, 100]
        labels = ['18-25', '26-35', '36-45', '46-55', '55+']
        
        df['age_group'] = pd.cut(df[age_column], bins=bins, labels=labels, include_lowest=True)
        
        print(f"✓ Created age_group feature")
        return df
    
    def create_income_groups(self, df, income_column='income'):
        """
        Create income group categories
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        income_column : str
            Name of income column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with income_group column added
        """
        df = df.copy()
        
        bins = [0, 40000, 60000, 80000, 120000, float('inf')]
        labels = ['Low', 'Lower-Middle', 'Middle', 'Upper-Middle', 'High']
        
        df['income_group'] = pd.cut(df[income_column], bins=bins, labels=labels)
        
        print(f"✓ Created income_group feature")
        return df
    
    def create_time_of_day(self, df, hour_column='hour_of_day'):
        """
        Create time of day categories
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        hour_column : str
            Name of hour column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with time_of_day column added
        """
        df = df.copy()
        
        conditions = [
            (df[hour_column] >= 0) & (df[hour_column] < 6),
            (df[hour_column] >= 6) & (df[hour_column] < 12),
            (df[hour_column] >= 12) & (df[hour_column] < 18),
            (df[hour_column] >= 18) & (df[hour_column] < 24)
        ]
        
        choices = ['Night', 'Morning', 'Afternoon', 'Evening']
        
        df['time_of_day'] = np.select(conditions, choices, default='Unknown')
        
        print(f"✓ Created time_of_day feature")
        return df
    
    def create_engagement_score(self, df, time_column='time_spent_on_site', pages_column='pages_viewed'):
        """
        Create engagement score combining time spent and pages viewed
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        time_column : str
            Name of time spent column
        pages_column : str
            Name of pages viewed column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with engagement_score column added
        """
        df = df.copy()
        
        # Normalize both features to 0-1 scale
        time_normalized = (df[time_column] - df[time_column].min()) / (df[time_column].max() - df[time_column].min())
        pages_normalized = (df[pages_column] - df[pages_column].min()) / (df[pages_column].max() - df[pages_column].min())
        
        # Weighted combination (60% time, 40% pages)
        df['engagement_score'] = 0.6 * time_normalized + 0.4 * pages_normalized
        
        print(f"✓ Created engagement_score feature")
        return df
    
    def create_is_weekend(self, df, day_column='day_of_week'):
        """
        Create binary weekend indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        day_column : str
            Name of day of week column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with is_weekend column added
        """
        df = df.copy()
        
        weekend_days = ['Saturday', 'Sunday']
        df['is_weekend'] = df[day_column].isin(weekend_days).astype(int)
        
        print(f"✓ Created is_weekend feature")
        return df
    
    def create_is_mobile(self, df, device_column='device'):
        """
        Create binary mobile indicator
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
        device_column : str
            Name of device column
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with is_mobile column added
        """
        df = df.copy()
        
        df['is_mobile'] = (df[device_column] == 'Mobile').astype(int)
        
        print(f"✓ Created is_mobile feature")
        return df
    
    def create_interaction_features(self, df):
        """
        Create interaction features between important variables
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with interaction features added
        """
        df = df.copy()
        
        # Age-Income interaction
        if 'age' in df.columns and 'income' in df.columns:
            df['age_income_interaction'] = df['age'] * df['income'] / 1000000
            print(f"✓ Created age_income_interaction feature")
        
        # Pages-Time interaction (engagement)
        if 'pages_viewed' in df.columns and 'time_spent_on_site' in df.columns:
            df['pages_time_interaction'] = df['pages_viewed'] * df['time_spent_on_site'] / 100
            print(f"✓ Created pages_time_interaction feature")
        
        # Previous clicks-Current engagement
        if 'previous_clicks' in df.columns and 'engagement_score' in df.columns:
            df['history_engagement'] = df['previous_clicks'] * df['engagement_score']
            print(f"✓ Created history_engagement feature")
        
        return df
    
    def create_all_features(self, df):
        """
        Apply all feature engineering steps
        
        Parameters:
        -----------
        df : pandas.DataFrame
            Input dataframe
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with all engineered features
        """
        print("\n" + "="*80)
        print("FEATURE ENGINEERING")
        print("="*80)
        
        df = self.create_age_groups(df)
        df = self.create_income_groups(df)
        df = self.create_time_of_day(df)
        df = self.create_engagement_score(df)
        df = self.create_is_weekend(df)
        df = self.create_is_mobile(df)
        df = self.create_interaction_features(df)
        
        print(f"\n✓ Feature engineering completed")
        print(f"  Total features: {df.shape[1]}")
        print("="*80)
        
        return df


def engineer_features(df):
    """
    Convenience function to apply feature engineering
    
    Parameters:
    -----------
    df : pandas.DataFrame
        Input dataframe
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with engineered features
    """
    engineer = FeatureEngineer()
    return engineer.create_all_features(df)


if __name__ == "__main__":
    # Example usage
    print("Feature Engineering Module")
    print("Use engineer_features(df) to apply all feature engineering steps")
