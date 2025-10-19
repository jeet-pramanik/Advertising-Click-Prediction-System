# Deployment Guide
## Advertising Click Prediction System

### Table of Contents
1. [Prerequisites](#prerequisites)
2. [Installation](#installation)
3. [Running the Pipeline](#running-the-pipeline)
4. [Deploying the API](#deploying-the-api)
5. [Testing](#testing)
6. [Monitoring](#monitoring)
7. [Troubleshooting](#troubleshooting)

---

## 1. Prerequisites

### System Requirements
- Python 3.8 or higher
- 4GB RAM minimum (8GB recommended)
- 2GB free disk space
- Internet connection (for package installation)

### Required Software
- Python 3.8+
- pip package manager
- Git (optional, for version control)

---

## 2. Installation

### Step 1: Navigate to Project Directory
```bash
cd "c:\Users\JEET PRAMANIK\structura\Advertising Click Prediction System"
```

### Step 2: Create Virtual Environment (Recommended)
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate

# On macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install -r requirements.txt
```

Expected output:
```
Successfully installed numpy-X.X.X pandas-X.X.X scikit-learn-X.X.X ...
```

### Step 4: Verify Installation
```bash
python -c "import sklearn, pandas, numpy; print('All packages installed successfully!')"
```

---

## 3. Running the Pipeline

### Option A: Run Complete Pipeline

Execute the master pipeline script:
```bash
python run_pipeline.py
```

This will:
1. Generate synthetic advertising data
2. Perform data understanding and EDA
3. Engineer features
4. Preprocess data
5. Train multiple models
6. Evaluate model performance
7. Generate business insights
8. Save models and visualizations

**Expected Runtime**: 5-10 minutes

### Option B: Run Jupyter Notebooks

Start Jupyter:
```bash
jupyter notebook
```

Navigate to `notebooks/` and run notebooks in sequence:
1. `01_data_understanding.ipynb`
2. `02_exploratory_analysis.ipynb`
3. `03_preprocessing.ipynb`
4. `04_model_development.ipynb`
5. `05_model_evaluation.ipynb`
6. `06_business_insights.ipynb`

---

## 4. Deploying the API

### Step 1: Ensure Models are Trained
```bash
# Check that models exist
ls models/
# Should see: logistic_regression.pkl, preprocessor.pkl, etc.
```

### Step 2: Start Flask API
```bash
cd deployment
python app.py
```

Expected output:
```
================================================================================
                        AD CLICK PREDICTION API
================================================================================
Starting Flask server...
API will be available at: http://localhost:5000
...
✓ Model and preprocessor loaded successfully
 * Running on http://0.0.0.0:5000
```

### Step 3: Test API

**Health Check**:
```bash
curl http://localhost:5000/health
```

Response:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "preprocessor_loaded": true
}
```

**Get Example Input Format**:
```bash
curl http://localhost:5000/example
```

**Make Prediction**:
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "age": 35,
    "gender": "Male",
    "income": 75000,
    "education": "Bachelor",
    "ad_topic": "Technology",
    "ad_position": "Top",
    "ad_size": "Medium",
    "time_spent_on_site": 180,
    "pages_viewed": 5,
    "previous_clicks": 2,
    "day_of_week": "Tuesday",
    "hour_of_day": 14,
    "season": "Spring",
    "device": "Mobile",
    "os": "Android",
    "browser": "Chrome"
  }'
```

Response:
```json
{
  "prediction": 1,
  "probability": 0.78,
  "confidence": "High",
  "recommendation": "Show ad"
}
```

---

## 5. Testing

### Run Unit Tests
```bash
cd tests
python test_prediction.py
```

Expected output:
```
test_age_range ... ok
test_categorical_values ... ok
test_example_input_format ... ok
...
================================================================================
TEST SUMMARY
================================================================================
Tests run: 15
Successes: 15
Failures: 0
Errors: 0
================================================================================
```

### Manual Testing Checklist

- [ ] Data generation works
- [ ] Models train successfully
- [ ] Predictions are reasonable
- [ ] API responds to all endpoints
- [ ] Visualizations are created
- [ ] Reports are generated

---

## 6. Monitoring

### Model Performance Metrics

Check model performance:
```python
import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score

# Load model and test data
model = joblib.load('models/logistic_regression.pkl')
# ... evaluate on new data
```

### API Monitoring

Monitor API logs:
```bash
tail -f api.log
```

Track key metrics:
- **Response Time**: Should be < 100ms
- **Error Rate**: Should be < 1%
- **Prediction Distribution**: Monitor for drift

### Data Drift Detection

Regularly check:
- Feature distributions
- Prediction probabilities
- Actual vs predicted click rates

---

## 7. Troubleshooting

### Common Issues

#### Issue 1: Import Errors
```
ModuleNotFoundError: No module named 'sklearn'
```

**Solution**:
```bash
pip install -r requirements.txt
```

#### Issue 2: Model Not Found
```
FileNotFoundError: models/logistic_regression.pkl
```

**Solution**:
```bash
# Run pipeline to generate models
python run_pipeline.py
```

#### Issue 3: API Won't Start
```
Address already in use: Port 5000
```

**Solution**:
```bash
# Use different port
python app.py --port 5001

# Or kill process on port 5000
# Windows:
netstat -ano | findstr :5000
taskkill /PID <PID> /F

# Linux/Mac:
lsof -i :5000
kill -9 <PID>
```

#### Issue 4: Low Model Performance
```
ROC-AUC < 0.70
```

**Solution**:
1. Check data quality
2. Retrain with hyperparameter tuning:
   ```python
   trainer.train_all_models(X_train, y_train, tune=True)
   ```
3. Try different models
4. Engineer more features

#### Issue 5: Memory Errors
```
MemoryError
```

**Solution**:
1. Reduce dataset size
2. Use batch processing
3. Increase system RAM
4. Use data sampling

### Getting Help

1. **Check Documentation**: Review technical_report.md
2. **Check Logs**: Review error messages and stack traces
3. **Test Components**: Run unit tests to isolate issues
4. **Update Dependencies**: Ensure all packages are up to date

---

## Production Deployment

### Option A: Docker Deployment

Create `Dockerfile`:
```dockerfile
FROM python:3.8-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

EXPOSE 5000

CMD ["python", "deployment/app.py"]
```

Build and run:
```bash
docker build -t ad-click-prediction .
docker run -p 5000:5000 ad-click-prediction
```

### Option B: Cloud Deployment

#### AWS EC2
1. Launch EC2 instance (t2.medium recommended)
2. SSH into instance
3. Clone repository
4. Install dependencies
5. Run API with systemd service

#### Google Cloud Platform
1. Deploy to Cloud Run or App Engine
2. Configure environment variables
3. Set up Cloud SQL for data storage
4. Enable Cloud Monitoring

#### Azure
1. Deploy to Azure App Service
2. Configure application settings
3. Set up Azure Monitor
4. Enable auto-scaling

### Security Considerations

1. **API Authentication**: Add API keys or OAuth
2. **HTTPS**: Use SSL certificates
3. **Rate Limiting**: Prevent abuse
4. **Input Validation**: Sanitize all inputs
5. **Logging**: Monitor for suspicious activity

---

## Maintenance

### Regular Tasks

#### Daily
- Monitor API health
- Check error logs
- Review prediction distribution

#### Weekly
- Analyze model performance
- Check data quality
- Review business metrics

#### Monthly
- Retrain models with new data
- Update dependencies
- Performance optimization
- Feature engineering review

### Model Retraining

```python
# Retrain with new data
from src.model_training import ModelTrainer

# Load new data
new_data = pd.read_csv('data/new_data.csv')

# Retrain model
trainer = ModelTrainer()
model = trainer.train_logistic_regression(X_new, y_new, tune=True)

# Save updated model
trainer.save_model(model, 'Logistic Regression')
```

---

## Support

For issues or questions:
- **Technical Support**: [technical-team@company.com]
- **Documentation**: See `reports/technical_report.md`
- **Issues**: Create ticket in issue tracker

---

**Document Version**: 1.0  
**Last Updated**: 2024  
**Maintained By**: Data Science Team
