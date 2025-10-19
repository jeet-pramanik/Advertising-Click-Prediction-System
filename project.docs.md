# Comprehensive Development Prompt: Advertising Click Prediction System

## Project Overview
You are tasked with developing a complete Advertising Click Prediction system that helps companies optimize their ad campaigns by predicting whether users will click on advertisements. This is a binary classification problem using machine learning techniques.

## Project Objectives
1. Build a predictive model to determine the likelihood of ad clicks
2. Identify key features that influence click-through rates
3. Provide actionable insights for ad targeting optimization
4. Create visualizations to communicate findings effectively
5. Deliver a production-ready solution with proper evaluation metrics

## Technical Stack Requirements
- **Language**: Python 3.8+
- **Core Libraries**: 
  - scikit-learn (model building)
  - pandas (data manipulation)
  - numpy (numerical operations)
  - matplotlib & seaborn (visualization)
- **Primary Algorithm**: Logistic Regression (with potential for model comparison)
- **Development Environment**: Jupyter Notebook or Python scripts

## Detailed Development Requirements

### Phase 1: Data Acquisition and Understanding

**Dataset Requirements:**
- Source a realistic advertising dataset with the following features (or similar):
  - User demographics (age, gender, income level, education)
  - Ad characteristics (topic, position, size, format)
  - User behavior (time spent on site, pages viewed, previous clicks)
  - Temporal features (day of week, hour of day, season)
  - Device information (mobile, desktop, tablet)
  - Target variable: Clicked (0 = No, 1 = Yes)

**You can use:**
- Publicly available datasets (Kaggle, UCI repository)
- Synthetic data generation if needed
- Recommended: "Advertising Dataset" or similar click-through rate datasets

**Data Understanding Tasks:**
- Load and display the first few rows of the dataset
- Provide dataset dimensions (rows, columns)
- List all features with their data types
- Check for missing values and document the percentage
- Display basic statistical summaries (mean, median, std, min, max)
- Analyze the class distribution (clicked vs not clicked)
- Check for class imbalance and calculate the imbalance ratio

### Phase 2: Exploratory Data Analysis (EDA)

**Visualization Requirements:**

1. **Target Variable Analysis:**
   - Create a count plot showing click vs no-click distribution
   - Calculate and display the click-through rate (CTR) percentage

2. **Univariate Analysis:**
   - Distribution plots for continuous features (age, income, time spent)
   - Bar charts for categorical features (gender, device type, ad position)

3. **Bivariate Analysis:**
   - Click rate by age groups (create age bins if needed)
   - Click rate by gender and income level
   - Click rate by time of day and day of week
   - Click rate by device type
   - Heatmap showing correlations between numerical features

4. **Feature Insights:**
   - Identify which demographics have highest click rates
   - Determine optimal ad positioning
   - Find best times for ad display
   - Analyze device-specific behavior patterns

**Statistical Analysis:**
- Perform correlation analysis between features
- Identify multicollinearity issues
- Conduct statistical tests (chi-square for categorical, t-tests for continuous)

### Phase 3: Data Preprocessing

**Data Cleaning:**
- Handle missing values using appropriate strategies:
  - Imputation (mean/median for numerical, mode for categorical)
  - Deletion if missing data is minimal (<5%)
  - Document all decisions made
  
**Feature Engineering:**
- Create new features if beneficial:
  - Age groups/bins (e.g., 18-25, 26-35, 36-45, 46+)
  - Time-based features (morning, afternoon, evening, night)
  - Interaction features (e.g., age × income)
  - Engagement score (combination of time spent and pages viewed)

**Encoding Categorical Variables:**
- Apply one-hot encoding for nominal variables (device type, ad topic)
- Apply label encoding or ordinal encoding where appropriate
- Handle the target variable (ensure it's 0 and 1)

**Feature Scaling:**
- Standardize or normalize numerical features
- Use StandardScaler or MinMaxScaler from scikit-learn
- Important: Fit scaler only on training data to prevent data leakage

**Data Splitting:**
- Split data into training and testing sets (80-20 or 70-30)
- Use stratified split to maintain class distribution
- Set random state for reproducibility
- Consider creating a validation set if hyperparameter tuning is extensive

### Phase 4: Model Development

**Baseline Model:**
- Create a simple baseline (predict majority class)
- Calculate baseline accuracy for comparison

**Logistic Regression Implementation:**

```python
# Example structure (not complete code)
from sklearn.linear_model import LogisticRegression

# Initialize model
model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    solver='lbfgs'
)

# Train model
# Make predictions
```

**Model Training:**
- Fit the logistic regression model on training data
- Document training time
- Explain the coefficients and their interpretations

**Hyperparameter Tuning:**
- Experiment with different regularization parameters (C values)
- Try different solvers (lbfgs, liblinear, saga)
- Test L1, L2, and elastic net penalties
- Use GridSearchCV or RandomizedSearchCV for systematic tuning
- Implement cross-validation (5-fold or 10-fold)

**Advanced Considerations:**
- Handle class imbalance if present:
  - Adjust class weights
  - Use SMOTE or other resampling techniques
  - Try threshold adjustment
- Feature selection to improve model performance
- Polynomial features for non-linear relationships

### Phase 5: Model Evaluation

**Performance Metrics (Critical):**

Calculate and display all of the following:
- **Accuracy**: Overall correctness
- **Precision**: Of predicted clicks, how many were correct?
- **Recall/Sensitivity**: Of actual clicks, how many did we catch?
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under ROC curve
- **Confusion Matrix**: Detailed breakdown of predictions

**Visualization Requirements:**

1. **Confusion Matrix Heatmap:**
   - Show true positives, true negatives, false positives, false negatives
   - Use annotations for clear reading

2. **ROC Curve:**
   - Plot True Positive Rate vs False Positive Rate
   - Display AUC score on the plot
   - Add diagonal reference line

3. **Precision-Recall Curve:**
   - Especially important for imbalanced datasets
   - Show the trade-off between precision and recall

4. **Feature Importance:**
   - Extract and visualize logistic regression coefficients
   - Show top 10 most influential features (both positive and negative)
   - Create a horizontal bar chart

**Model Interpretation:**
- Explain what each important feature's coefficient means
- Discuss odds ratios for significant predictors
- Provide business insights from feature importance

### Phase 6: Business Insights and Recommendations

**Analysis Deliverables:**

1. **Click-Through Rate Insights:**
   - Which user segments have highest CTR?
   - What are optimal ad characteristics?
   - Best times and platforms for ad display

2. **Targeting Recommendations:**
   - High-value user profiles to target
   - Features to avoid or de-emphasize
   - Budget allocation suggestions

3. **ROI Projections:**
   - Expected improvement from model deployment
   - Cost-benefit analysis framework
   - A/B testing recommendations

4. **Actionable Strategies:**
   - Specific recommendations for ad creative
   - Audience segmentation strategy
   - Bidding strategy optimization

### Phase 7: Model Deployment Preparation

**Create Production-Ready Code:**

```python
# Example structure for prediction pipeline
def predict_ad_click(user_features):
    """
    Predict whether a user will click on an ad
    
    Parameters:
    user_features: dict or DataFrame with user characteristics
    
    Returns:
    prediction: 0 or 1
    probability: click probability
    """
    # Preprocessing steps
    # Feature engineering
    # Scaling
    # Prediction
    # Return results
```

**Deliverables:**
- Model serialization (save using pickle or joblib)
- Preprocessing pipeline saved separately
- Input validation function
- Prediction API structure (Flask/FastAPI outline)
- Documentation for model usage

### Phase 8: Documentation and Reporting

**Create Comprehensive Documentation:**

1. **Executive Summary (Non-Technical):**
   - Problem statement
   - Solution approach
   - Key findings in business terms
   - Expected impact

2. **Technical Documentation:**
   - Data description and preprocessing steps
   - Model selection rationale
   - Performance metrics with interpretations
   - Limitations and assumptions

3. **Visualizations Dashboard:**
   - Compile all key visualizations
   - Create an executive-friendly visual summary
   - Include comparative charts (before/after scenarios)

4. **Code Documentation:**
   - Inline comments explaining logic
   - Docstrings for all functions
   - README file with setup instructions

**Presentation Structure:**
- Problem and business context (2 slides)
- Data exploration findings (3-4 slides)
- Model approach and methodology (2 slides)
- Results and performance (2-3 slides)
- Business recommendations (2-3 slides)
- Next steps and future work (1 slide)

## Additional Model Comparisons (Optional but Recommended)

To validate that logistic regression is the best choice, compare with:
- Decision Trees
- Random Forest
- Gradient Boosting (XGBoost, LightGBM)
- Support Vector Machines

Create a comparison table showing accuracy, precision, recall, F1, and AUC for each model.

## Success Criteria

Your solution should achieve:
- **Minimum AUC Score**: 0.75+ (0.80+ is excellent)
- **Balanced Performance**: F1-score > 0.70
- **Interpretability**: Clear feature importance insights
- **Reproducibility**: All results can be recreated with provided code
- **Business Value**: Actionable recommendations that can increase CTR by at least 10%

## Error Handling and Edge Cases

Consider and handle:
- New categories in categorical features
- Missing values in production data
- Extreme values in numerical features
- Data drift detection mechanisms
- Model performance monitoring strategy

## Deliverables Checklist

Ensure you provide:
- [ ] Clean, well-commented Python code (Jupyter Notebook or .py files)
- [ ] Trained model file (pickle/joblib)
- [ ] Preprocessing pipeline file
- [ ] Comprehensive EDA report with visualizations
- [ ] Model performance report with all metrics
- [ ] Business insights document
- [ ] README with setup and usage instructions
- [ ] Requirements.txt file with all dependencies
- [ ] Executive presentation (PDF/PPT)

## Best Practices to Follow

1. **Code Quality:**
   - Use meaningful variable names
   - Follow PEP 8 style guidelines
   - Modular code with reusable functions
   - Error handling with try-except blocks

2. **Data Science Workflow:**
   - Set random seeds for reproducibility
   - Document all assumptions
   - Version control (Git recommended)
   - Keep raw data unchanged

3. **Model Development:**
   - Always validate on unseen test data
   - Never tune on test set
   - Document model limitations
   - Consider ethical implications

4. **Visualization:**
   - Clear titles and labels
   - Appropriate color schemes
   - Consistent styling
   - Accessible to colorblind viewers

## Timeline Suggestion

- Data acquisition and understanding: 10%
- EDA and visualization: 25%
- Preprocessing and feature engineering: 20%
- Model development and tuning: 25%
- Evaluation and insights: 15%
- Documentation and deployment prep: 5%

## Final Notes

This is a real-world business problem. Your solution should demonstrate:
- Technical proficiency in machine learning
- Business acumen in translating results to insights
- Communication skills through documentation
- Production-readiness of your code

Focus on creating a solution that a marketing team can actually use to improve their ad campaigns. The model is only valuable if it leads to actionable business decisions.

Good luck with your development! 🎯