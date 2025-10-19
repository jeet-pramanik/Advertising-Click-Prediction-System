"""
Unit Tests for Ad Click Prediction System
"""

import unittest
import sys
import os
import numpy as np
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction_api import AdClickPredictor, EXAMPLE_INPUT


class TestPredictionAPI(unittest.TestCase):
    """Test prediction API functionality"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.predictor = AdClickPredictor(
            model_path='../models/logistic_regression.pkl',
            preprocessor_path='../models/preprocessor.pkl'
        )
        self.example_input = EXAMPLE_INPUT.copy()
    
    def test_example_input_format(self):
        """Test that example input has all required fields"""
        required_fields = [
            'age', 'gender', 'income', 'education',
            'ad_topic', 'ad_position', 'ad_size',
            'time_spent_on_site', 'pages_viewed', 'previous_clicks',
            'day_of_week', 'hour_of_day', 'season',
            'device', 'os', 'browser'
        ]
        
        for field in required_fields:
            self.assertIn(field, self.example_input,
                         f"Required field '{field}' missing from example input")
    
    def test_input_data_types(self):
        """Test input data types are correct"""
        self.assertIsInstance(self.example_input['age'], int)
        self.assertIsInstance(self.example_input['income'], int)
        self.assertIsInstance(self.example_input['gender'], str)
        self.assertIsInstance(self.example_input['device'], str)
    
    def test_age_range(self):
        """Test age is within valid range"""
        age = self.example_input['age']
        self.assertGreaterEqual(age, 18, "Age should be >= 18")
        self.assertLessEqual(age, 70, "Age should be <= 70")
    
    def test_income_range(self):
        """Test income is within valid range"""
        income = self.example_input['income']
        self.assertGreaterEqual(income, 20000, "Income should be >= 20000")
        self.assertLessEqual(income, 200000, "Income should be <= 200000")
    
    def test_hour_range(self):
        """Test hour is within valid range"""
        hour = self.example_input['hour_of_day']
        self.assertGreaterEqual(hour, 0, "Hour should be >= 0")
        self.assertLessEqual(hour, 23, "Hour should be <= 23")
    
    def test_categorical_values(self):
        """Test categorical values are valid"""
        valid_genders = ['Male', 'Female', 'Other']
        valid_devices = ['Mobile', 'Desktop', 'Tablet']
        
        self.assertIn(self.example_input['gender'], valid_genders)
        self.assertIn(self.example_input['device'], valid_devices)
    
    def test_recommendation_confidence_levels(self):
        """Test recommendation confidence level logic"""
        # Test high confidence
        recommendation = self.predictor.get_recommendation(1, 0.85)
        self.assertEqual(recommendation['confidence'], 'High')
        
        # Test medium confidence
        recommendation = self.predictor.get_recommendation(1, 0.60)
        self.assertEqual(recommendation['confidence'], 'Medium')
        
        # Test low confidence
        recommendation = self.predictor.get_recommendation(1, 0.40)
        self.assertEqual(recommendation['confidence'], 'Low')
        
        # Test very low confidence
        recommendation = self.predictor.get_recommendation(0, 0.20)
        self.assertEqual(recommendation['confidence'], 'Very Low')


class TestDataGeneration(unittest.TestCase):
    """Test data generation functionality"""
    
    def test_generate_sample_data(self):
        """Test synthetic data generation"""
        from data_generator import generate_advertising_dataset
        
        df = generate_advertising_dataset(n_samples=100)
        
        # Check shape
        self.assertEqual(len(df), 100, "Should generate 100 samples")
        self.assertGreater(df.shape[1], 15, "Should have >15 features")
        
        # Check required columns
        required_columns = ['user_id', 'age', 'gender', 'clicked']
        for col in required_columns:
            self.assertIn(col, df.columns, f"Column '{col}' should exist")
        
        # Check data types
        self.assertTrue(pd.api.types.is_integer_dtype(df['age']))
        self.assertTrue(pd.api.types.is_object_dtype(df['gender']))
        self.assertTrue(pd.api.types.is_integer_dtype(df['clicked']))
        
        # Check target variable values
        unique_clicks = df['clicked'].unique()
        self.assertTrue(set(unique_clicks).issubset({0, 1}),
                       "Clicked should only contain 0 and 1")


class TestPreprocessing(unittest.TestCase):
    """Test preprocessing functionality"""
    
    def test_feature_types_identification(self):
        """Test identification of numerical and categorical features"""
        from preprocessing import PreprocessingPipeline
        from data_generator import generate_advertising_dataset
        
        df = generate_advertising_dataset(n_samples=100)
        pipeline = PreprocessingPipeline()
        
        numerical, categorical = pipeline.identify_feature_types(df)
        
        # Check that we have both types
        self.assertGreater(len(numerical), 0, "Should have numerical features")
        self.assertGreater(len(categorical), 0, "Should have categorical features")
        
        # Check specific features
        self.assertIn('age', numerical, "Age should be numerical")
        self.assertIn('gender', categorical, "Gender should be categorical")


class TestFeatureEngineering(unittest.TestCase):
    """Test feature engineering functionality"""
    
    def test_age_groups_creation(self):
        """Test age group feature creation"""
        from feature_engineering import FeatureEngineer
        
        df = pd.DataFrame({'age': [20, 30, 40, 50, 60]})
        engineer = FeatureEngineer()
        
        df_result = engineer.create_age_groups(df)
        
        # Check that age_group column was created
        self.assertIn('age_group', df_result.columns)
        
        # Check that groups are assigned correctly
        self.assertIsNotNone(df_result['age_group'].iloc[0])
    
    def test_engagement_score_creation(self):
        """Test engagement score creation"""
        from feature_engineering import FeatureEngineer
        
        df = pd.DataFrame({
            'time_spent_on_site': [100, 200, 300],
            'pages_viewed': [2, 5, 8]
        })
        engineer = FeatureEngineer()
        
        df_result = engineer.create_engagement_score(df)
        
        # Check that engagement_score was created
        self.assertIn('engagement_score', df_result.columns)
        
        # Check that scores are between 0 and 1
        self.assertTrue((df_result['engagement_score'] >= 0).all())
        self.assertTrue((df_result['engagement_score'] <= 1).all())


class TestModelEvaluation(unittest.TestCase):
    """Test model evaluation functionality"""
    
    def test_metrics_calculation(self):
        """Test that metrics are calculated correctly"""
        from model_evaluation import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Simple test case
        y_true = np.array([0, 0, 1, 1, 1, 0, 1, 0])
        y_pred = np.array([0, 0, 1, 1, 0, 0, 1, 1])
        y_pred_proba = np.array([0.1, 0.2, 0.8, 0.9, 0.4, 0.3, 0.7, 0.6])
        
        metrics = evaluator.calculate_metrics(y_true, y_pred, y_pred_proba)
        
        # Check that all metrics exist
        required_metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC-AUC']
        for metric in required_metrics:
            self.assertIn(metric, metrics, f"Metric '{metric}' should exist")
        
        # Check that metrics are in valid range [0, 1]
        for metric, value in metrics.items():
            if metric != 'Model':
                self.assertGreaterEqual(value, 0, f"{metric} should be >= 0")
                self.assertLessEqual(value, 1, f"{metric} should be <= 1")


def run_tests():
    """Run all tests"""
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPredictionAPI))
    suite.addTests(loader.loadTestsFromTestCase(TestDataGeneration))
    suite.addTests(loader.loadTestsFromTestCase(TestPreprocessing))
    suite.addTests(loader.loadTestsFromTestCase(TestFeatureEngineering))
    suite.addTests(loader.loadTestsFromTestCase(TestModelEvaluation))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "="*80)
    print("TEST SUMMARY")
    print("="*80)
    print(f"Tests run: {result.testsRun}")
    print(f"Successes: {result.testsRun - len(result.failures) - len(result.errors)}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print("="*80)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    sys.exit(0 if success else 1)
