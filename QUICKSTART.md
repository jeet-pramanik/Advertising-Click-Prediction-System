# 🚀 QUICK START GUIDE
## Advertising Click Prediction System

**Get up and running in 5 minutes!**

---

## Prerequisites

✅ Python 3.8+ installed  
✅ Terminal/Command Prompt access  
✅ 5-10 minutes of time

---

## Step 1: Open Terminal

Navigate to the project directory:

```bash
cd "c:\Users\JEET PRAMANIK\structura\Advertising Click Prediction System"
```

---

## Step 2: Install Dependencies (One Time)

```bash
pip install -r requirements.txt
```

⏱️ **Time**: ~2 minutes

---

## Step 3: Run the Complete Pipeline

```bash
python run_pipeline.py
```

⏱️ **Time**: ~5-10 minutes

### What happens:
- ✅ Generates 10,000 advertising samples
- ✅ Analyzes data and creates visualizations
- ✅ Engineers features
- ✅ Trains 4 ML models
- ✅ Evaluates performance
- ✅ Generates business insights
- ✅ Saves everything to disk

### Expected Output:
```
================================================================================
                    ADVERTISING CLICK PREDICTION SYSTEM
                         Complete End-to-End Pipeline
================================================================================

================================================================================
                         PHASE 1: DATA GENERATION
================================================================================

Generating 10000 samples of advertising data...
Dataset generated successfully!
Shape: (10000, 19)
Click rate: 23.45%
Dataset saved to data/raw/advertising_data.csv

...

================================================================================
                    PIPELINE EXECUTION COMPLETE
================================================================================
✓ All phases completed successfully!

📁 Outputs Generated:
   - Dataset: data/raw/advertising_data.csv
   - Models: models/
   - Visualizations: visualizations/
   - Reports: reports/
```

---

## Step 4: View Results

### Check Visualizations
```bash
# Open the visualizations folder
start visualizations\eda\
start visualizations\model_performance\
```

### Read Reports
```bash
# View business insights
type reports\business_insights.txt

# View technical report
start reports\technical_report.md

# View executive summary
start reports\executive_summary.md
```

---

## Step 5: Deploy the API (Optional)

### Start the API Server
```bash
cd deployment
python app.py
```

### Test the API
Open another terminal and run:

```bash
# Health check
curl http://localhost:5000/health

# Get example format
curl http://localhost:5000/example

# Make a prediction
curl -X POST http://localhost:5000/predict ^
  -H "Content-Type: application/json" ^
  -d "{\"age\": 35, \"gender\": \"Male\", \"income\": 75000, \"education\": \"Bachelor\", \"ad_topic\": \"Technology\", \"ad_position\": \"Top\", \"ad_size\": \"Medium\", \"time_spent_on_site\": 180, \"pages_viewed\": 5, \"previous_clicks\": 2, \"day_of_week\": \"Tuesday\", \"hour_of_day\": 14, \"season\": \"Spring\", \"device\": \"Mobile\", \"os\": \"Android\", \"browser\": \"Chrome\"}"
```

**Expected Response:**
```json
{
  "prediction": 1,
  "probability": 0.78,
  "confidence": "High",
  "recommendation": "Show ad"
}
```

---

## Step 6: Run Tests (Optional)

```bash
cd tests
python test_prediction.py
```

**Expected Output:**
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

---

## 🎉 You're Done!

### What You Now Have:

✅ **Trained ML Models** in `models/` folder  
✅ **Visualizations** in `visualizations/` folder  
✅ **Business Insights** in `reports/` folder  
✅ **Working API** at `http://localhost:5000`  
✅ **Complete Documentation** in `reports/`

---

## 📊 Quick Results Summary

After running the pipeline, you'll see:

- **Dataset**: 10,000 samples with 18 features
- **Click Rate**: ~20-25% (typical for advertising)
- **Best Model**: Logistic Regression or Random Forest
- **ROC-AUC Score**: 0.75+ (excellent)
- **Accuracy**: 75%+
- **Key Insight**: Mobile users show different behavior
- **Recommendation**: Target specific demographics for +25% CTR

---

## 🆘 Troubleshooting

### Issue: "Module not found"
**Solution**: 
```bash
pip install -r requirements.txt
```

### Issue: "Port 5000 already in use"
**Solution**: 
```bash
# Kill process or use different port
python app.py --port 5001
```

### Issue: "Permission denied"
**Solution**: 
```bash
# Run as administrator (Windows) or use sudo (Linux/Mac)
```

---

## 📚 Next Steps

1. **Understand the Code**: Review `src/` folder
2. **Explore Notebooks**: Open `notebooks/` in Jupyter
3. **Read Documentation**: See `reports/technical_report.md`
4. **Customize**: Modify parameters in `run_pipeline.py`
5. **Deploy**: Follow `deployment/deployment_guide.md`

---

## 🔗 Important Files

| File | Purpose |
|------|---------|
| `run_pipeline.py` | Main execution script |
| `README.md` | Full project documentation |
| `PROJECT_SUMMARY.md` | Complete project overview |
| `deployment/app.py` | API server |
| `reports/executive_summary.md` | Business summary |

---

## 💡 Pro Tips

1. **Save Time**: Use pre-trained models in `models/` folder
2. **Customize Data**: Edit parameters in `data_generator.py`
3. **Quick Tests**: Use `notebooks/` for interactive exploration
4. **API First**: Start API before generating new models
5. **Monitor**: Check logs for errors and performance

---

## ✅ Success Checklist

After running the quick start, you should have:

- [ ] Installed all dependencies
- [ ] Generated dataset
- [ ] Trained models
- [ ] Created visualizations
- [ ] Generated reports
- [ ] Started API (optional)
- [ ] Run tests (optional)
- [ ] Reviewed results

---

## 🎓 Learn More

- **Technical Details**: `reports/technical_report.md`
- **Business Context**: `reports/executive_summary.md`
- **Deployment**: `deployment/deployment_guide.md`
- **Full Documentation**: `README.md`

---

**Questions?** Review the documentation files or check the troubleshooting section!

**Ready to deploy?** See `deployment/deployment_guide.md`

---

🚀 **Happy Predicting!** 🎯
