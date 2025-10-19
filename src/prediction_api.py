"""
Prediction API Module
Functions for making predictions with trained models
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


class AdClickPredictor:
    """Class for making ad click predictions"""
    
    def __init__(self, model_path='models/logistic_regression.pkl', 
                 preprocessor_path='models/preprocessor.pkl'):
        """
        Initialize predictor
        
        Parameters:
        -----------
        model_path : str
            Path to trained model
        preprocessor_path : str
            Path to preprocessing pipeline
        """
        self.model = None
        self.preprocessor = None
        self.model_path = model_path
        self.preprocessor_path = preprocessor_path
        
    def load_model(self):
        """Load trained model"""
        try:
            self.model = joblib.load(self.model_path)
            print(f"✓ Model loaded from {self.model_path}")
            return True
        except Exception as e:
            print(f"✗ Error loading model: {str(e)}")
            return False
    
    def load_preprocessor(self):
        """Load preprocessing pipeline"""
        try:
            from preprocessing import PreprocessingPipeline
            self.preprocessor = PreprocessingPipeline.load_pipeline(self.preprocessor_path)
            return True
        except Exception as e:
            print(f"✗ Error loading preprocessor: {str(e)}")
            return False
    
    def preprocess_input(self, user_data):
        """
        Preprocess user input data
        
        Parameters:
        -----------
        user_data : dict or pandas.DataFrame
            User features
            
        Returns:
        --------
        array-like
            Preprocessed features
        """
        # Convert dict to DataFrame if needed
        if isinstance(user_data, dict):
            user_data = pd.DataFrame([user_data])
        
        # Apply preprocessing
        if self.preprocessor:
            X, _ = self.preprocessor.prepare_data(user_data, fit=False)
            return X
        else:
            return user_data
    
    def predict(self, user_data):
        """
        Make prediction
        
        Parameters:
        -----------
        user_data : dict or pandas.DataFrame
            User features
            
        Returns:
        --------
        tuple
            (prediction, probability)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Preprocess input
        X = self.preprocess_input(user_data)
        
        # Make prediction
        prediction = self.model.predict(X)[0]
        
        # Get probability if available
        if hasattr(self.model, 'predict_proba'):
            probability = self.model.predict_proba(X)[0][1]
        else:
            probability = None
        
        return prediction, probability
    
    def predict_batch(self, user_data_list):
        """
        Make predictions for multiple users
        
        Parameters:
        -----------
        user_data_list : list of dicts or pandas.DataFrame
            Multiple user features
            
        Returns:
        --------
        tuple
            (predictions, probabilities)
        """
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Convert to DataFrame if needed
        if isinstance(user_data_list, list):
            user_data_df = pd.DataFrame(user_data_list)
        else:
            user_data_df = user_data_list
        
        # Preprocess input
        X = self.preprocess_input(user_data_df)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            probabilities = self.model.predict_proba(X)[:, 1]
        else:
            probabilities = None
        
        return predictions, probabilities
    
    def get_recommendation(self, prediction, probability):
        """
        Get business recommendation based on prediction
        
        Parameters:
        -----------
        prediction : int
            Predicted class (0 or 1)
        probability : float
            Prediction probability
            
        Returns:
        --------
        dict
            Recommendation details
        """
        if probability is None:
            return {
                'prediction': int(prediction),
                'recommendation': 'Show ad' if prediction == 1 else 'Skip ad'
            }
        
        # Define thresholds
        if probability >= 0.7:
            confidence = 'High'
            recommendation = 'Strongly recommend showing ad'
        elif probability >= 0.5:
            confidence = 'Medium'
            recommendation = 'Show ad'
        elif probability >= 0.3:
            confidence = 'Low'
            recommendation = 'Consider showing ad'
        else:
            confidence = 'Very Low'
            recommendation = 'Skip ad'
        
        return {
            'prediction': int(prediction),
            'probability': float(probability),
            'confidence': confidence,
            'recommendation': recommendation
        }


def predict_ad_click(user_data, model_path='models/logistic_regression.pkl',
                    preprocessor_path='models/preprocessor.pkl'):
    """
    Convenience function to make a single prediction
    
    Parameters:
    -----------
    user_data : dict
        User features
    model_path : str
        Path to trained model
    preprocessor_path : str
        Path to preprocessing pipeline
        
    Returns:
    --------
    dict
        Prediction results and recommendation
    """
    predictor = AdClickPredictor(model_path, preprocessor_path)
    predictor.load_model()
    predictor.load_preprocessor()
    
    prediction, probability = predictor.predict(user_data)
    recommendation = predictor.get_recommendation(prediction, probability)
    
    return recommendation


# Example input template
EXAMPLE_INPUT = {
    'age': 35,
    'gender': 'Male',
    'income': 75000,
    'education': 'Bachelor',
    'ad_topic': 'Technology',
    'ad_position': 'Top',
    'ad_size': 'Medium',
    'time_spent_on_site': 180,
    'pages_viewed': 5,
    'previous_clicks': 2,
    'day_of_week': 'Tuesday',
    'hour_of_day': 14,
    'season': 'Spring',
    'device': 'Mobile',
    'os': 'Android',
    'browser': 'Chrome'
}


if __name__ == "__main__":
    print("Prediction API Module")
    print("\nExample usage:")
    print("predictor = AdClickPredictor()")
    print("predictor.load_model()")
    print("predictor.load_preprocessor()")
    print("prediction, probability = predictor.predict(user_data)")
    print("\nExample input format:")
    print(EXAMPLE_INPUT)
