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
    return len(pred[(pred == 1) & (ground_truth == 0)]) / len(ground_truth[ground_truth == 0])

def FNR(pred, ground_truth):
    """False negative rate"""
    return len(pred[(pred == 0) & (ground_truth == 1)]) / len(ground_truth[ground_truth == 1])

def nFPED(pred, ground_truth, targets, num_target_groups):
    """Average FPR absolute distance"""
    return np.sum(np.abs([FPR(pred, ground_truth) -
                          FPR(pred[np.array(targets[i]).astype(bool)],
                              ground_truth[np.array(targets[i]).astype(bool)])
                          for i in range(num_target_groups)])) / num_target_groups

def nFNED(pred, ground_truth, targets, num_target_groups):
    """Average FNR absolute distance"""
    return np.sum(np.abs([FNR(pred, ground_truth) -
                          FNR(pred[np.array(targets[i]).astype(bool)],
                              ground_truth[np.array(targets[i]).astype(bool)])
                          for i in range(num_target_groups)])) / num_target_groups

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
    macro_f1 = f1_score(y_test, predictions, average='macro')
    print(f"Overall Macro-F1 Score: {macro_f1:.4f}")

    # Gender-based analysis
    if 'gender' in data.columns:
        for gender_value in ["0","1"]:
            print(f"\nEvaluating for gender == {gender_value}...")
            gender_data = data[data['gender'] == gender_value]
            if gender_data.empty:
                print(f"No data for gender == {gender_value}. Skipping...")
                continue

            X_gender = np.array(gender_data['embed'].tolist())
            y_gender = gender_data['label']
            pred_gender = (model.predict(X_gender) > 0.5).astype(int)
            gender_macro_f1 = f1_score(y_gender, pred_gender, average='macro')
            print(f"Macro-F1 Score (gender == {gender_value}): {gender_macro_f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_gender, pred_gender))

    # Gender-based analysis
    if 'ethnicity' in data.columns:
        for race_value in ["0","1"]:
            print(f"\nEvaluating for race == {race_value}...")
            race_data = data[data['ethnicity'] == race_value]
            if race_data.empty:
                print(f"No data for race == {race_value}. Skipping...")
                continue

            X_race = np.array(race_data['embed'].tolist())
            y_race = race_data['label']
            pred_race = (model.predict(X_race) > 0.5).astype(int)
            race_macro_f1 = f1_score(y_race, pred_race, average='macro')
            print(f"Macro-F1 Score (race == {race_value}): {race_macro_f1:.4f}")
            print("\nClassification Report:")
            print(classification_report(y_race, pred_race))

    # Classification report for overall data
    print("\nClassification Report (Overall):")
    print(classification_report(y_test, predictions))

def evaluate_all_models_in_directory(directory, csv_file):
    """Evaluate all models in a directory on the given test data."""
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not model_files:
        print(f"No .h5 model files found in directory: {directory}")
        return

    for model_file in model_files:
        model_path = os.path.join(directory, model_file)
        print(f"\n=== Evaluating Model: {model_file} ===")
        evaluate_model(model_path, csv_file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models in a directory on test data.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing model files.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the test dataset (CSV format).")

    args = parser.parse_args()
    evaluate_all_models_in_directory(args.model_dir, args.csv_file)
