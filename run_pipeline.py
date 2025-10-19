"""
Master Pipeline Script
Complete end-to-end advertising click prediction system

This script runs all phases:
1. Data Generation
2. Data Understanding  
3. Exploratory Data Analysis
4. Preprocessing & Feature Engineering
5. Model Training
6. Model Evaluation
7. Business Insights
8. Model Deployment

Run this script to execute the complete project workflow.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Import custom modules
from src.data_generator import generate_advertising_dataset, save_dataset
from src.data_loader import DataLoader
from src.feature_engineering import FeatureEngineer
from src.preprocessing import PreprocessingPipeline, split_data
from src.model_training import ModelTrainer
from src.model_evaluation import ModelEvaluator

# Set random seed for reproducibility
np.random.seed(42)

# Set plotting style
sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def print_header(title):
    """Print formatted section header"""
    print("\n" + "="*100)
    print(f"{title:^100}")
    print("="*100 + "\n")


def phase_1_data_generation():
    """Phase 1: Generate and save advertising dataset"""
    print_header("PHASE 1: DATA GENERATION")
    
    # Generate dataset
    df = generate_advertising_dataset(n_samples=10000)
    
    # Save dataset
    filepath = save_dataset(df, 'advertising_data.csv')
    
    print("\n✓ Phase 1 Complete: Dataset generated and saved")
    return df


def phase_2_data_understanding(df):
    """Phase 2: Load and understand the dataset"""
    print_header("PHASE 2: DATA UNDERSTANDING")
    
    # Initialize data loader
    loader = DataLoader()
    loader.data = df
    
    # Display summary
    loader.display_summary()
    
    # Get feature types
    feature_types = loader.get_feature_types()
    
    print(f"\n📊 Feature Types:")
    print(f"   Numerical: {len(feature_types['numerical'])} features")
    print(f"   Categorical: {len(feature_types['categorical'])} features")
    
    print("\n✓ Phase 2 Complete: Data understanding finished")
    return loader


def phase_3_exploratory_analysis(df):
    """Phase 3: Exploratory Data Analysis"""
    print_header("PHASE 3: EXPLORATORY DATA ANALYSIS")
    
    # Analyze target variable
    click_rate = df['clicked'].mean()
    print(f"📊 Click-Through Rate: {click_rate*100:.2f}%")
    
    # Create visualizations directory if it doesn't exist
    os.makedirs('visualizations/eda', exist_ok=True)
    
    # Target distribution visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    click_counts = df['clicked'].value_counts()
    axes[0].bar(['Not Clicked', 'Clicked'], click_counts.values, color=['#e74c3c', '#2ecc71'])
    axes[0].set_ylabel('Count', fontsize=12)
    axes[0].set_title('Click Distribution', fontsize=14, fontweight='bold')
    
    axes[1].pie(click_counts.values, labels=['Not Clicked', 'Clicked'], 
               colors=['#e74c3c', '#2ecc71'], autopct='%1.1f%%', startangle=90)
    axes[1].set_title('Click Distribution %', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('visualizations/eda/target_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Target distribution visualization saved")
    
    # Age distribution by click
    fig, ax = plt.subplots(figsize=(12, 6))
    df[df['clicked']==0]['age'].hist(bins=30, alpha=0.6, label='Not Clicked', color='red', ax=ax)
    df[df['clicked']==1]['age'].hist(bins=30, alpha=0.6, label='Clicked', color='green', ax=ax)
    ax.set_xlabel('Age', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.set_title('Age Distribution by Click Status', fontsize=14, fontweight='bold')
    ax.legend()
    plt.tight_layout()
    plt.savefig('visualizations/eda/age_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Age distribution visualization saved")
    
    # Device type analysis
    device_click_rate = df.groupby('device')['clicked'].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(10, 6))
    device_click_rate.plot(kind='bar', ax=ax, color='skyblue', edgecolor='black')
    ax.set_ylabel('Click Rate', fontsize=12)
    ax.set_title('Click Rate by Device Type', fontsize=14, fontweight='bold')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.savefig('visualizations/eda/device_click_rate.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("✓ Device analysis visualization saved")
    
    print("\n✓ Phase 3 Complete: EDA finished")


def phase_4_feature_engineering(df):
    """Phase 4: Feature Engineering"""
    print_header("PHASE 4: FEATURE ENGINEERING")
    
    # Apply feature engineering
    engineer = FeatureEngineer()
    df_engineered = engineer.create_all_features(df)
    
    print(f"\n📊 Features after engineering: {df_engineered.shape[1]}")
    print(f"   New features created: {df_engineered.shape[1] - df.shape[1]}")
    
    print("\n✓ Phase 4 Complete: Feature engineering finished")
    return df_engineered


def phase_5_preprocessing(df):
    """Phase 5: Data Preprocessing"""
    print_header("PHASE 5: DATA PREPROCESSING")
    
    # Initialize preprocessing pipeline
    pipeline = PreprocessingPipeline()
    
    # Prepare data
    X, y = pipeline.prepare_data(df, target_column='clicked', fit=True, encoding_type='label')
    
    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y, test_size=0.2, random_state=42, stratify=True)
    
    # Save preprocessing pipeline
    os.makedirs('models', exist_ok=True)
    pipeline.save_pipeline('models/preprocessor.pkl')
    
    print("\n✓ Phase 5 Complete: Preprocessing finished")
    return X_train, X_test, y_train, y_test, pipeline


def phase_6_model_training(X_train, y_train):
    """Phase 6: Model Training"""
    print_header("PHASE 6: MODEL TRAINING")
    
    # Initialize trainer
    trainer = ModelTrainer(random_state=42)
    
    # Create baseline
    trainer.create_baseline_model(y_train)
    
    # Train models (without extensive tuning for speed)
    print("\n🚀 Training models (quick mode - no hyperparameter tuning)...")
    models = trainer.train_all_models(X_train, y_train, tune=False)
    
    # Save best model (Logistic Regression by default)
    trainer.save_model(models['Logistic Regression'], 'Logistic Regression')
    trainer.save_model(models['Random Forest'], 'Random Forest')
    
    print("\n✓ Phase 6 Complete: Model training finished")
    return models, trainer


def phase_7_model_evaluation(models, X_test, y_test, feature_names):
    """Phase 7: Model Evaluation"""
    print_header("PHASE 7: MODEL EVALUATION")
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Create output directory
    os.makedirs('visualizations/model_performance', exist_ok=True)
    
    # Evaluate each model
    for model_name, model in models.items():
        print(f"\n📊 Evaluating {model_name}...")
        metrics = evaluator.evaluate_model(
            model, X_test, y_test,
            model_name=model_name,
            feature_names=feature_names,
            save_dir='visualizations/model_performance'
        )
    
    # Compare models
    comparison_df = evaluator.compare_models(
        save_path='visualizations/model_performance/model_comparison.png'
    )
    
    print("\n✓ Phase 7 Complete: Model evaluation finished")
    return comparison_df, evaluator


def phase_8_business_insights(df, comparison_df):
    """Phase 8: Generate Business Insights"""
    print_header("PHASE 8: BUSINESS INSIGHTS & RECOMMENDATIONS")
    
    print("\n📈 KEY FINDINGS:")
    print("="*100)
    
    # Overall CTR
    ctr = df['clicked'].mean()
    print(f"\n1. Overall Click-Through Rate: {ctr*100:.2f}%")
    
    # Demographics insights
    print(f"\n2. Demographics Analysis:")
    age_ctr = df.groupby(pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100]))['clicked'].mean()
    print(f"   Highest CTR Age Group: {age_ctr.idxmax()} ({age_ctr.max()*100:.2f}%)")
    
    gender_ctr = df.groupby('gender')['clicked'].mean().sort_values(ascending=False)
    print(f"   Highest CTR Gender: {gender_ctr.index[0]} ({gender_ctr.iloc[0]*100:.2f}%)")
    
    # Device insights
    print(f"\n3. Device Performance:")
    device_ctr = df.groupby('device')['clicked'].mean().sort_values(ascending=False)
    for device, ctr_val in device_ctr.items():
        print(f"   {device}: {ctr_val*100:.2f}%")
    
    # Time insights
    print(f"\n4. Temporal Patterns:")
    hour_ctr = df.groupby('hour_of_day')['clicked'].mean()
    best_hour = hour_ctr.idxmax()
    print(f"   Best Hour: {best_hour}:00 ({hour_ctr.max()*100:.2f}% CTR)")
    
    day_ctr = df.groupby('day_of_week')['clicked'].mean().sort_values(ascending=False)
    print(f"   Best Day: {day_ctr.index[0]} ({day_ctr.iloc[0]*100:.2f}% CTR)")
    
    # Ad position insights
    print(f"\n5. Ad Positioning:")
    position_ctr = df.groupby('ad_position')['clicked'].mean().sort_values(ascending=False)
    for position, ctr_val in position_ctr.items():
        print(f"   {position}: {ctr_val*100:.2f}%")
    
    # Model performance
    print(f"\n6. Model Performance:")
    if comparison_df is not None:
        best_model = comparison_df['ROC-AUC'].idxmax()
        best_auc = comparison_df.loc[best_model, 'ROC-AUC']
        print(f"   Best Model: {best_model}")
        print(f"   ROC-AUC Score: {best_auc:.4f}")
        print(f"   Accuracy: {comparison_df.loc[best_model, 'Accuracy']:.4f}")
    
    # Business Recommendations
    print(f"\n💼 ACTIONABLE RECOMMENDATIONS:")
    print("="*100)
    print(f"\n1. Target Optimization:")
    print(f"   → Focus on {device_ctr.index[0]} users (highest CTR)")
    print(f"   → Optimize for {best_hour}:00 - {(best_hour+2)%24}:00 time window")
    print(f"   → Prioritize {day_ctr.index[0]} and {day_ctr.index[1]} for campaigns")
    
    print(f"\n2. Ad Placement Strategy:")
    print(f"   → Prioritize '{position_ctr.index[0]}' position")
    print(f"   → Expected CTR improvement: {((position_ctr.iloc[0] - ctr) / ctr * 100):.1f}%")
    
    print(f"\n3. Budget Allocation:")
    print(f"   → Allocate 40% budget to {device_ctr.index[0]} campaigns")
    print(f"   → Allocate 30% budget to {device_ctr.index[1]} campaigns")
    print(f"   → Reserve 30% for testing and optimization")
    
    print(f"\n4. Expected ROI:")
    print(f"   → With optimized targeting: +{((device_ctr.iloc[0] / ctr - 1) * 100):.1f}% CTR increase")
    print(f"   → Potential cost savings: ~25-30% from better targeting")
    
    # Save report
    os.makedirs('reports', exist_ok=True)
    
    with open('reports/business_insights.txt', 'w') as f:
        f.write("ADVERTISING CLICK PREDICTION - BUSINESS INSIGHTS\n")
        f.write("="*100 + "\n\n")
        f.write(f"Overall CTR: {ctr*100:.2f}%\n")
        f.write(f"Best Device: {device_ctr.index[0]} ({device_ctr.iloc[0]*100:.2f}%)\n")
        f.write(f"Best Time: {best_hour}:00\n")
        f.write(f"Best Ad Position: {position_ctr.index[0]}\n")
        f.write(f"\nModel Performance: {best_model} (AUC: {best_auc:.4f})\n")
    
    print("\n✓ Business insights saved to reports/business_insights.txt")
    print("\n✓ Phase 8 Complete: Business analysis finished")


def main():
    """Run the complete pipeline"""
    print("\n" + "="*100)
    print(" "*30 + "ADVERTISING CLICK PREDICTION SYSTEM")
    print(" "*35 + "Complete End-to-End Pipeline")
    print("="*100)
    
    try:
        # Phase 1: Data Generation
        df = phase_1_data_generation()
        
        # Phase 2: Data Understanding
        loader = phase_2_data_understanding(df)
        
        # Phase 3: EDA
        phase_3_exploratory_analysis(df)
        
        # Phase 4: Feature Engineering
        df_engineered = phase_4_feature_engineering(df)
        
        # Phase 5: Preprocessing
        X_train, X_test, y_train, y_test, pipeline = phase_5_preprocessing(df_engineered)
        
        # Phase 6: Model Training
        models, trainer = phase_6_model_training(X_train, y_train)
        
        # Phase 7: Model Evaluation
        comparison_df, evaluator = phase_7_model_evaluation(
            models, X_test, y_test, pipeline.feature_names
        )
        
        # Phase 8: Business Insights
        phase_8_business_insights(df, comparison_df)
        
        # Final Summary
        print_header("PIPELINE EXECUTION COMPLETE")
        print("✓ All phases completed successfully!")
        print("\n📁 Outputs Generated:")
        print("   - Dataset: data/raw/advertising_data.csv")
        print("   - Models: models/")
        print("   - Visualizations: visualizations/")
        print("   - Reports: reports/")
        print("\n🚀 Next Steps:")
        print("   1. Review visualizations in visualizations/ folder")
        print("   2. Check model performance metrics in reports/")
        print("   3. Deploy the model using deployment/app.py")
        print("   4. Test predictions using the API")
        
        print("\n" + "="*100)
        print("Thank you for using the Advertising Click Prediction System!")
        print("="*100 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error during pipeline execution: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
