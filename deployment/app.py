"""
Flask API for Ad Click Prediction
Production-ready REST API for serving predictions
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import sys
import os

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from prediction_api import AdClickPredictor

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Initialize predictor
predictor = AdClickPredictor(
    model_path='../models/logistic_regression.pkl',
    preprocessor_path='../models/preprocessor.pkl'
)

# Load model and preprocessor on startup
try:
    predictor.load_model()
    predictor.load_preprocessor()
    print("✓ Model and preprocessor loaded successfully")
except Exception as e:
    print(f"✗ Error loading model or preprocessor: {str(e)}")


@app.route('/', methods=['GET'])
def home():
    """Home endpoint"""
    return jsonify({
        'message': 'Ad Click Prediction API',
        'version': '1.0.0',
        'endpoints': {
            '/predict': 'POST - Make a single prediction',
            '/predict_batch': 'POST - Make multiple predictions',
            '/health': 'GET - Check API health'
        }
    })


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None,
        'preprocessor_loaded': predictor.preprocessor is not None
    })


@app.route('/predict', methods=['POST'])
def predict():
    """
    Single prediction endpoint
    
    Expected JSON format:
    {
        "age": 35,
        "gender": "Male",
        "income": 75000,
        ...
    }
    """
    try:
        # Get user data from request
        user_data = request.get_json()
        
        if not user_data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Make prediction
        prediction, probability = predictor.predict(user_data)
        
        # Get recommendation
        recommendation = predictor.get_recommendation(prediction, probability)
        
        return jsonify(recommendation)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Batch prediction endpoint
    
    Expected JSON format:
    {
        "users": [
            {"age": 35, "gender": "Male", ...},
            {"age": 28, "gender": "Female", ...}
        ]
    }
    """
    try:
        # Get data from request
        data = request.get_json()
        
        if not data or 'users' not in data:
            return jsonify({'error': 'No user data provided'}), 400
        
        user_data_list = data['users']
        
        # Make predictions
        predictions, probabilities = predictor.predict_batch(user_data_list)
        
        # Format results
        results = []
        for i, (pred, prob) in enumerate(zip(predictions, probabilities)):
            recommendation = predictor.get_recommendation(pred, prob)
            recommendation['user_index'] = i
            results.append(recommendation)
        
        return jsonify({
            'count': len(results),
            'predictions': results
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/example', methods=['GET'])
def example():
    """Return example input format"""
    from prediction_api import EXAMPLE_INPUT
    
    return jsonify({
        'message': 'Example input format for prediction',
        'example': EXAMPLE_INPUT
    })


if __name__ == '__main__':
    print("\n" + "="*80)
    print("AD CLICK PREDICTION API")
    print("="*80)
    print("Starting Flask server...")
    print("API will be available at: http://localhost:5000")
    print("\nEndpoints:")
    print("  GET  /              - API information")
    print("  GET  /health        - Health check")
    print("  POST /predict       - Single prediction")
    print("  POST /predict_batch - Batch predictions")
    print("  GET  /example       - Example input format")
    print("="*80 + "\n")
    
    app.run(debug=True, host='0.0.0.0', port=5000)
