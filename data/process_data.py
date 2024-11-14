import pandas as pd
import sys
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib

def load_data(file_path):
    """
    Load the dataset from a given file path.

    Args:
    - file_path (str): The path to the dataset file.

    Returns:
    - df (pd.DataFrame): The loaded dataframe.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def clean_data(df):
    """
    Clean the data by dropping irrelevant columns and separating the target.

    Args:
    - df (pd.DataFrame): The loaded dataset.

    Returns:
    - X (pd.DataFrame): Features dataframe.
    - y (pd.Series): Target variable.
    """
    try:
        # Drop the Name column
        df = df.drop('Name', axis=1)
        
        # Define the target variable
        target = 'History of Mental Illness'
        X = df.drop(columns=[target])

        # Drop unnecessary features
        X = X.drop(['History of Substance Abuse', 'Income'], axis=1)

        # Define the target variable
        y = df[target].map({'Yes': 1, 'No': 0})  # Encoding target as binary
        
        return X, y
    except KeyError as e:
        print(f"Error: {e} column not found in the dataset.")
        sys.exit(1)
    except Exception as e:
        print(f"Error cleaning data: {e}")
        sys.exit(1)

def preprocess_data(X):
    """
    Preprocess the feature data using a pipeline with ordinal, one-hot, and binary encoding.

    Args:
    - X (pd.DataFrame): The features dataframe.

    Returns:
    - preprocessor (Pipeline): The preprocessing pipeline for future transformations.
    - X_transformed (pd.DataFrame): Transformed features.
    """
    try:
        # Defining column groups based on encoding strategies
        continuous_features = ['Age']
        ordinal_features = ['Education Level', 'Physical Activity Level', 'Alcohol Consumption', 'Dietary Habits', 'Sleep Patterns']
        ordinal_mappings = [
            ["Bachelor's Degree", 'High School', 'Associate Degree', "Master's Degree", 'PhD'],  # Education Level in order
            ['Sedentary', 'Moderate', 'Active'],  # Physical Activity Level
            ['Low', 'Moderate', 'High'],  # Alcohol Consumption levels
            ['Unhealthy', 'Moderate', 'Healthy'],  # Dietary habits in order
            ['Poor', 'Fair', 'Good']  # Sleep patterns in order
        ]
        onehot_features = ['Marital Status', 'Smoking Status']
        binary_features = ['Employment Status',  'Family History of Depression', 'Chronic Medical Conditions']

        # Ordinal encoding for ordered categorical features
        ordinal_transformer = Pipeline(steps=[
            ('ordinal', OrdinalEncoder(categories=ordinal_mappings))
        ])

        # One-hot encoding for unordered categorical features
        onehot_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(drop='first'))  # drop='first' to avoid multicollinearity
        ])

        # Binary encoding for binary features
        # For binary features, we map values to 0/1
        X[binary_features] = X[binary_features].apply(lambda x: x.map({'Yes': 1, 'No': 0, 'Employed': 1, 'Unemployed': 0}))

        # Applying all transformations in a ColumnTransformer
        preprocessor = ColumnTransformer(
            transformers=[
                ('ord', ordinal_transformer, ordinal_features),  # Only ordinal encoding
                ('onehot', onehot_transformer, onehot_features),  # One-hot encoding
                ('passthrough', 'passthrough', binary_features)  # Keep binary features as they are
            ],
            remainder='passthrough'  # Pass through continuous features as they are (without scaling)
        )

        # Apply the preprocessor to the data
        X_transformed = preprocessor.fit_transform(X)

        # Save the preprocessor for future use
        joblib.dump(preprocessor, 'preprocessor_pipeline.joblib')

        return preprocessor, X_transformed
    except Exception as e:
        print(f"Error preprocessing data: {e}")
        sys.exit(1)

def save_data(X, y, features_filepath, target_filepath):
    """
    Save the processed features and target to separate files.

    Args:
    - X (pd.DataFrame): Features dataframe.
    - y (pd.Series): Target variable.
    - features_filepath (str): Path where features will be saved.
    - target_filepath (str): Path where the target variable will be saved.

    Returns:
    - None
    """
    try:
        pd.DataFrame(X).to_csv(features_filepath, index=False)
        y.to_csv(target_filepath, index=False)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")
        sys.exit(1)

def main():
    """
    Main function to load, clean, preprocess, and save data.

    Command-line arguments:
    - dataset_filepath (str): Filepath of the dataset.
    - features_filepath (str): Path to save processed features.
    - target_filepath (str): Path to save the target variable.

    Returns:
    - None
    """
    try:
        if len(sys.argv) == 4:
            dataset_filepath, features_filepath, target_filepath = sys.argv[1:]

            # Load, clean, preprocess, and save the data
            df = load_data(dataset_filepath)
            X, y = clean_data(df)
            preprocessor, X_transformed = preprocess_data(X)
            save_data(X_transformed, y, features_filepath, target_filepath)
        else:
            print("Error: Please provide the correct number of arguments.")
            sys.exit(1)
    except Exception as e:
        print(f"An error occurred: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()
