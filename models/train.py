import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
from xgboost import XGBClassifier
import sys

def load_data(features_filepath, target_filepath):
    """
    Load and preprocess the features and target data from CSV files.
    
    Args:
    - features_filepath (str): Path to the features CSV file.
    - target_filepath (str): Path to the target CSV file.
    
    Returns:
    - X (pd.DataFrame): Preprocessed features dataframe.
    - y (pd.Series): Target variable.
    """
    try:
        # Load the datasets
        X = pd.read_csv(features_filepath)
        y = pd.read_csv(target_filepath)
        
        return X, y
    except FileNotFoundError as e:
        print(f"Error: {e.strerror}. Please check the file paths.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def build_model():
    """
    Build and return a dictionary of models to train.

    Returns:
    - models (dict): Dictionary containing model names and model objects.
    """
    try:
        models = {
            'Random Forest': RandomForestClassifier(random_state=42),
            'XGBoost': XGBClassifier(use_label_encoder=False, random_state=42)
        }
        return models
    except Exception as e:
        print(f"Error building models: {e}")
        sys.exit(1)

def evaluate_model(model, X_test, y_test):
    """
    Evaluate the model's performance using accuracy and classification report.

    Args:
    - model (sklearn model): The trained model to evaluate.
    - X_test (pd.DataFrame): The testing features.
    - y_test (pd.Series): The actual target values.

    Returns:
    - None
    """
    try:
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", classification_report(y_test, y_pred))
    except Exception as e:
        print(f"Error evaluating model: {e}")
        sys.exit(1)

def save_model(model, model_filename):
    """
    Save the trained model to a file.

    Args:
    - model (sklearn model): The trained model to save.
    - model_filename (str): The path to save the model.

    Returns:
    - None
    """
    try:
        joblib.dump(model, model_filename)
        print(f"Model saved as {model_filename}")
    except Exception as e:
        print(f"Error saving model: {e}")
        sys.exit(1)

def main():
    """
    Main function to train the model, evaluate, and save it.

    Command-line arguments:
    - features_filepath (str): Filepath of the processed feature data.
    - target_filepath (str): Filepath of the processed target data.

    Returns:
    - None
    """
    try:
        if len(sys.argv) == 3:
            features_filepath, target_filepath = sys.argv[1:]

            # Load data
            X, y = load_data(features_filepath, target_filepath)

            # # Load the preprocessor pipeline
            # preprocessor = joblib.load('preprocessor_pipeline.joblib')

            # # Transform the features using the preprocessor pipeline
            # X_transformed = preprocessor.transform(X)

            # Split data into train/test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

            # Build and train models
            models = build_model()
            for model_name, model in models.items():
                print(f"Training {model_name}...")
                model.fit(X_train, y_train)

                # Save the trained model
                model_filename = f"{model_name.replace(' ', '_')}_model.joblib"
                save_model(model, model_filename)

                # Evaluate the model
                evaluate_model(model, X_test, y_test)
                print("-" * 40)
        else:
            print("Error: Please provide the correct number of arguments.")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
