"""
Data Generator for Advertising Click Prediction System
Generates synthetic realistic advertising dataset with all required features
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)


def generate_advertising_dataset(n_samples=10000):
    """
    Generate synthetic advertising dataset with realistic patterns
    
    Parameters:
    -----------
    n_samples : int
        Number of samples to generate (default: 10000)
    
    Returns:
    --------
    pandas.DataFrame
        Generated dataset with all features
    """
    
    print(f"Generating {n_samples} samples of advertising data...")
    
    # User Demographics
    ages = np.random.normal(35, 12, n_samples).clip(18, 70).astype(int)
    genders = np.random.choice(['Male', 'Female', 'Other'], n_samples, p=[0.48, 0.50, 0.02])
    
    # Income levels (correlated with age)
    base_income = 30000 + (ages - 18) * 1500
    income = (base_income + np.random.normal(0, 15000, n_samples)).clip(20000, 200000).astype(int)
    
    # Education levels
    education = np.random.choice(
        ['High School', 'Bachelor', 'Master', 'PhD', 'Some College'],
        n_samples,
        p=[0.25, 0.35, 0.25, 0.10, 0.05]
    )
    
    # Ad Characteristics
    ad_topics = np.random.choice(
        ['Technology', 'Fashion', 'Food', 'Travel', 'Sports', 'Entertainment', 'Finance', 'Health'],
        n_samples,
        p=[0.15, 0.15, 0.12, 0.10, 0.10, 0.13, 0.15, 0.10]
    )
    
    ad_positions = np.random.choice(
        ['Top', 'Sidebar', 'Bottom', 'Middle', 'Pop-up'],
        n_samples,
        p=[0.30, 0.25, 0.15, 0.20, 0.10]
    )
    
    ad_sizes = np.random.choice(
        ['Small', 'Medium', 'Large', 'Banner'],
        n_samples,
        p=[0.20, 0.35, 0.25, 0.20]
    )
    
    # User Behavior
    time_spent_on_site = np.random.gamma(2, 50, n_samples).clip(10, 600).astype(int)  # seconds
    pages_viewed = np.random.poisson(3.5, n_samples).clip(1, 20)
    
    # Previous clicks (creates correlation with future clicks)
    previous_clicks = np.random.poisson(1.5, n_samples).clip(0, 10)
    
    # Temporal Features
    start_date = datetime(2024, 1, 1)
    dates = [start_date + timedelta(days=random.randint(0, 365)) for _ in range(n_samples)]
    
    day_of_week = [date.strftime('%A') for date in dates]
    hour_of_day = np.random.choice(range(24), n_samples)
    
    # Season
    months = [date.month for date in dates]
    seasons = []
    for month in months:
        if month in [12, 1, 2]:
            seasons.append('Winter')
        elif month in [3, 4, 5]:
            seasons.append('Spring')
        elif month in [6, 7, 8]:
            seasons.append('Summer')
        else:
            seasons.append('Fall')
    
    # Device Information
    devices = np.random.choice(
        ['Mobile', 'Desktop', 'Tablet'],
        n_samples,
        p=[0.55, 0.35, 0.10]
    )
    
    # Operating System
    os_types = []
    for device in devices:
        if device == 'Mobile':
            os_types.append(np.random.choice(['iOS', 'Android'], p=[0.45, 0.55]))
        elif device == 'Desktop':
            os_types.append(np.random.choice(['Windows', 'MacOS', 'Linux'], p=[0.70, 0.25, 0.05]))
        else:  # Tablet
            os_types.append(np.random.choice(['iOS', 'Android'], p=[0.60, 0.40]))
    
    # Browser
    browsers = np.random.choice(
        ['Chrome', 'Safari', 'Firefox', 'Edge', 'Other'],
        n_samples,
        p=[0.50, 0.25, 0.15, 0.08, 0.02]
    )
    
    # Create Click Target (with realistic patterns)
    click_probability = np.zeros(n_samples)
    
    # Base probability
    click_probability += 0.15
    
    # Age effect (younger users click more)
    click_probability += (50 - ages) / 500
    
    # Income effect (middle-income clicks more)
    income_normalized = (income - income.min()) / (income.max() - income.min())
    click_probability += 0.1 * (1 - abs(income_normalized - 0.5) * 2)
    
    # Ad position effect
    position_boost = {'Top': 0.15, 'Middle': 0.10, 'Sidebar': 0.05, 'Bottom': 0.02, 'Pop-up': -0.05}
    click_probability += np.array([position_boost[pos] for pos in ad_positions])
    
    # Device effect (Mobile users click more)
    device_boost = {'Mobile': 0.08, 'Desktop': 0.05, 'Tablet': 0.03}
    click_probability += np.array([device_boost[dev] for dev in devices])
    
    # Time effect (evening hours have higher clicks)
    time_boost = np.where((hour_of_day >= 18) & (hour_of_day <= 22), 0.10, 0)
    click_probability += time_boost
    
    # Engagement effect (more engagement = more clicks)
    click_probability += (time_spent_on_site / 600) * 0.15
    click_probability += (pages_viewed / 20) * 0.10
    
    # Previous clicks effect (history matters)
    click_probability += (previous_clicks / 10) * 0.20
    
    # Ad topic interest (some topics perform better)
    topic_boost = {
        'Technology': 0.10, 'Fashion': 0.12, 'Food': 0.08,
        'Travel': 0.11, 'Sports': 0.07, 'Entertainment': 0.09,
        'Finance': 0.05, 'Health': 0.08
    }
    click_probability += np.array([topic_boost[topic] for topic in ad_topics])
    
    # Clip probabilities to valid range
    click_probability = click_probability.clip(0, 1)
    
    # Generate actual clicks based on probability
    clicked = np.random.binomial(1, click_probability)
    
    # Create DataFrame
    df = pd.DataFrame({
        'user_id': range(1, n_samples + 1),
        'age': ages,
        'gender': genders,
        'income': income,
        'education': education,
        'ad_topic': ad_topics,
        'ad_position': ad_positions,
        'ad_size': ad_sizes,
        'time_spent_on_site': time_spent_on_site,
        'pages_viewed': pages_viewed,
        'previous_clicks': previous_clicks,
        'date': dates,
        'day_of_week': day_of_week,
        'hour_of_day': hour_of_day,
        'season': seasons,
        'device': devices,
        'os': os_types,
        'browser': browsers,
        'clicked': clicked
    })
    
    # Add some missing values (realistic scenario)
    missing_indices = np.random.choice(n_samples, size=int(n_samples * 0.03), replace=False)
    df.loc[missing_indices[:len(missing_indices)//3], 'income'] = np.nan
    df.loc[missing_indices[len(missing_indices)//3:2*len(missing_indices)//3], 'education'] = np.nan
    df.loc[missing_indices[2*len(missing_indices)//3:], 'time_spent_on_site'] = np.nan
    
    print(f"Dataset generated successfully!")
    print(f"Shape: {df.shape}")
    print(f"Click rate: {df['clicked'].mean():.2%}")
    
    return df


def save_dataset(df, filename='advertising_data.csv'):
    """Save the generated dataset to CSV file"""
    filepath = f'data/raw/{filename}'
    df.to_csv(filepath, index=False)
    print(f"Dataset saved to {filepath}")
    return filepath


if __name__ == "__main__":
    # Generate dataset
    df = generate_advertising_dataset(n_samples=10000)
    
    # Display basic info
    print("\n" + "="*50)
    print("Dataset Preview:")
    print("="*50)
    print(df.head())
    
    print("\n" + "="*50)
    print("Dataset Info:")
    print("="*50)
    print(df.info())
    
    print("\n" + "="*50)
    print("Statistical Summary:")
    print("="*50)
    print(df.describe())
    
    # Save to CSV
    save_dataset(df)
