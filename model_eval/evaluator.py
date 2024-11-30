import os
import numpy as np
import pandas as pd
import logging
from tensorflow.keras.models import load_model
from sklearn.metrics import classification_report, f1_score
import json
import argparse

def FPR(pred, ground_truth):
    """False positive rate"""
    pred = np.asarray(pred)  # Ensure NumPy array
    ground_truth = np.asarray(ground_truth)  # Ensure NumPy array
    return np.sum((pred == 1) & (ground_truth == 0)) / max(1, np.sum(ground_truth == 0))  # Avoid division by zero

def FNR(pred, ground_truth):
    """False negative rate"""
    pred = np.asarray(pred)  # Ensure NumPy array
    ground_truth = np.asarray(ground_truth)  # Ensure NumPy array
    return np.sum((pred == 0) & (ground_truth == 1)) / max(1, np.sum(ground_truth == 1))  # Avoid division by zero

def evaluate_model(model_path, csv_file):
    """Load a saved model and evaluate it on the test data."""
    # Load the model
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Preprocess test data
    data = pd.read_csv(csv_file, sep='\t')
    data['embed'] = data['embed'].apply(json.loads)  # Ensure 'embed' is JSON-decoded

    # Extract test features and labels
    X_test = np.array(data['embed'].tolist())
    y_test = data['label']

    # Evaluate the model
    print("Evaluating model on test data...")
    predictions = (model.predict(X_test) > 0.5).astype(int)

    # Initialize results dictionary
    results = {"model": os.path.basename(model_path)}

    # Gender-based analysis
    if 'gender' in data.columns:
        for gender_value, gender_label in [("0", "male"), ("1", "female")]:
            gender_data = data[data['gender'] == gender_value]
            if gender_data.empty:
                print(f"No data for gender == {gender_value}. Skipping...")
                results[f"F1_{gender_label}"] = None
                results[f"FP_{gender_label}"] = None
                results[f"FN_{gender_label}"] = None
                continue

            X_gender = np.array(gender_data['embed'].tolist())
            y_gender = gender_data['label'].to_numpy()  # Convert to NumPy array
            pred_gender = (model.predict(X_gender) > 0.5).astype(int).flatten()

            gender_macro_f1 = f1_score(y_gender, pred_gender, average='macro')
            gender_fp = FPR(pred_gender, y_gender)
            gender_fn = FNR(pred_gender, y_gender)

            results[f"F1_{gender_label}"] = gender_macro_f1
            results[f"FP_{gender_label}"] = gender_fp
            results[f"FN_{gender_label}"] = gender_fn

    # Ethnicity-based analysis
    if 'ethnicity' in data.columns:
        for race_value, race_label in [("0", "white"), ("1", "nonwhite")]:
            race_data = data[data['ethnicity'] == race_value]
            if race_data.empty:
                print(f"No data for ethnicity == {race_value}. Skipping...")
                results[f"F1_{race_label}"] = None
                results[f"FP_{race_label}"] = None
                results[f"FN_{race_label}"] = None
                continue

            X_race = np.array(race_data['embed'].tolist())
            y_race = race_data['label'].to_numpy()  # Convert to NumPy array
            pred_race = (model.predict(X_race) > 0.5).astype(int).flatten()

            race_macro_f1 = f1_score(y_race, pred_race, average='macro')
            race_fp = FPR(pred_race, y_race)
            race_fn = FNR(pred_race, y_race)

            results[f"F1_{race_label}"] = race_macro_f1
            results[f"FP_{race_label}"] = race_fp
            results[f"FN_{race_label}"] = race_fn

    return results

def evaluate_all_models_in_directory(directory, csv_file):
    """Evaluate all models in a directory on the given test data and save results to a CSV."""
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not model_files:
        print(f"No .h5 model files found in directory: {directory}")
        return

    results = []
    for model_file in model_files:
        model_path = os.path.join(directory, model_file)
        print(f"\n=== Evaluating Model: {model_file} ===")
        model_results = evaluate_model(model_path, csv_file)
        results.append(model_results)

    # Save results to CSV
    results_df = pd.DataFrame(results)
    os.makedirs("./results", exist_ok=True)
    results_path = "./results/evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"\nResults saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models in a directory on test data.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing model files.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the test dataset (CSV format).")

    args = parser.parse_args()
    evaluate_all_models_in_directory(args.model_dir, args.csv_file)
