import os
import numpy as np
import pandas as pd
import json
from tensorflow.keras.models import load_model
from sklearn.metrics import f1_score
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
    print(f"Loading model from {model_path}...")
    model = load_model(model_path)

    # Load dataset
    data = pd.read_csv(csv_file, sep='\t')
    data['embed'] = data['embed'].apply(json.loads)  # Ensure 'embed' is JSON-decoded

    # Extract test features and labels
    X_test = np.array(data['embed'].tolist())
    y_test = data['label'].to_numpy()

    # Predict model outputs
    predictions = model.predict(X_test).flatten()  # Continuous predictions
    binary_predictions = (predictions > 0.5).astype(int)  # Apply threshold

    # Add prediction and error columns
    data[f"prediction_{os.path.basename(model_path)}"] = predictions
    data[f"false_positive_{os.path.basename(model_path)}"] = ((binary_predictions == 1) & (y_test == 0)).astype(int)
    data[f"false_negative_{os.path.basename(model_path)}"] = ((binary_predictions == 0) & (y_test == 1)).astype(int)

    # Compute overall metrics
    overall_f1 = f1_score(y_test, binary_predictions, average='macro')
    overall_fp = np.sum((binary_predictions == 1) & (y_test == 0))
    overall_fn = np.sum((binary_predictions == 0) & (y_test == 1))

    # Return appended data and model results
    return data, {"model": os.path.basename(model_path), "F1": overall_f1, "FP": overall_fp, "FN": overall_fn}

def evaluate_all_models_in_directory(directory, csv_file):
    """Evaluate all models in a directory on the given test data and save results to CSV."""
    model_files = [f for f in os.listdir(directory) if f.endswith('.h5')]
    if not model_files:
        print(f"No .h5 model files found in directory: {directory}")
        return

    all_data = pd.read_csv(csv_file, sep='\t')  # Original dataset
    all_data['embed'] = all_data['embed'].apply(json.loads)  # Ensure 'embed' is JSON-decoded
    overall_results = []

    for model_file in model_files:
        model_path = os.path.join(directory, model_file)
        print(f"\n=== Evaluating Model: {model_file} ===")
        model_data, model_results = evaluate_model(model_path, csv_file)
        overall_results.append(model_results)

        # Append results to the original dataset
        all_data = all_data.merge(
            model_data[[f"prediction_{model_file}", f"false_positive_{model_file}", f"false_negative_{model_file}"]],
            left_index=True,
            right_index=True,
            how="left"
        )

    # Add overall metrics to the results DataFrame
    results_df = pd.DataFrame(overall_results)

    # Add overall metrics as columns to the original dataset
    all_data["overall_F1"] = results_df["F1"].mean()  # Average F1 across models
    all_data["overall_FP"] = results_df["FP"].sum()  # Sum of false positives across models
    all_data["overall_FN"] = results_df["FN"].sum()  # Sum of false negatives across models

    # Save the updated dataset with predictions and errors
    os.makedirs("./results", exist_ok=True)
    updated_dataset_path = "./results/updated_dataset.csv"
    all_data.to_csv(updated_dataset_path, index=False)
    print(f"Updated dataset saved to {updated_dataset_path}")

    # Save the results for all models
    results_path = "./results/evaluation_results.csv"
    results_df.to_csv(results_path, index=False)
    print(f"Model results saved to {results_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models in a directory on test data.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to the directory containing model files.")
    parser.add_argument("--csv_file", type=str, required=True, help="Path to the test dataset (CSV format).")

    args = parser.parse_args()
    evaluate_all_models_in_directory(args.model_dir, args.csv_file)
