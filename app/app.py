from flask import Flask, request, jsonify
import joblib
import pandas as pd
import sys

# Initialize Flask app
app = Flask(__name__)

# Attempt to load the pre-trained model and pipeline
try:
    model_path = 'Random_Forest_model.joblib'
    pipeline_path = 'preprocessor_pipeline.joblib'
    
    print(f"Loading model from {model_path}")
    model = joblib.load(model_path)
    print("Model loaded successfully.")

    print(f"Loading pipeline from {pipeline_path}")
    pipeline = joblib.load(pipeline_path)
    print("Pipeline loaded successfully.")

except FileNotFoundError as e:
    print(f"FileNotFoundError: {e.filename} not found.")
    sys.exit(1)
except Exception as e:
    print(f"Unexpected error: {e}")
    sys.exit(1)

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for making predictions."""
    try:
        data = request.get_json()
        df = pd.DataFrame(data)
        X_transformed = pipeline.transform(df)
        prediction = model.predict(X_transformed)
        result = {'prediction': 'Yes' if prediction[0] == 1 else 'No'}
        return jsonify(result)
    except Exception as e:
        print(f"Error processing prediction request: {e}")
        return jsonify({'error': 'Failed to process the request.'}), 400

if __name__ == '__main__':
    app.run(debug=True)
