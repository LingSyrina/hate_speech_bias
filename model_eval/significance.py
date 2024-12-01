import pandas as pd
print(pd.__version__)
import statsmodels.api as sm
import os

print(sm.__version__)

def perform_significance_testing(data, models):
    """
    Perform logistic and linear regression to test if false positive, false negative,
    and error can be predicted by gender for each model.
    """
    # Drop rows where gender is 'x'
    data = data[data['ethnicity'] != 'x']
    data['ethnicity'] = pd.to_numeric(data['ethnicity'], errors='coerce')

    results = []

    for model in models:
        # Define columns for current model
        fp_col = f"false_positive_{model}"
        fn_col = f"false_negative_{model}"
        pred_col = f"prediction_{model}"

        if fp_col not in data.columns or fn_col not in data.columns or pred_col not in data.columns:
            print(f"Columns for {model} not found in data. Skipping...")
            continue

        # Calculate error as abs(prediction - label)
        data[f"error_{model}"] = abs(data[pred_col] - data['label'])

        # Gender predictor
        X = sm.add_constant(data['ethnicity'])  # Add intercept for regression

        # Logistic regression for false positive
        if data[fp_col].sum() > 0:  # Ensure non-zero variance
            logistic_fp = sm.Logit(data[fp_col], X).fit(disp=0)
            gender_fp = logistic_fp.summary2().tables[1].loc['ethnicity']  # Extract gender row
            gender_fp['Outcome'] = 'False Positive'
            gender_fp['Model'] = model
        else:
            gender_fp = pd.Series({
                'Outcome': 'False Positive',
                'Model': model,
                'Coef.': None,
                'P>|z|': None
            })

        # Logistic regression for false negative
        if data[fn_col].sum() > 0:  # Ensure non-zero variance
            logistic_fn = sm.Logit(data[fn_col], X).fit(disp=0)
            gender_fn = logistic_fn.summary2().tables[1].loc['ethnicity']  # Extract gender row
            gender_fn['Outcome'] = 'False Negative'
            gender_fn['Model'] = model
        else:
            gender_fn = pd.Series({
                'Outcome': 'False Negative',
                'Model': model,
                'Coef.': None,
                'P>|z|': None
            })

        # Linear regression for error
        linear_error = sm.OLS(data[f"error_{model}"], X).fit()
        gender_error = linear_error.summary2().tables[1].loc['ethnicity']  # Extract gender row
        gender_error['Outcome'] = 'Error'
        gender_error['Model'] = model

        # Append results
        results.append(gender_fp)
        results.append(gender_fn)
        results.append(gender_error)

    # Concatenate all results into a single DataFrame
    full_results = pd.DataFrame(results)

    return full_results


# Load updated dataset
data_path = "/Users/leasyrin/Downloads/IUB_PhD/CS/L715:B659/Final_project/code_for_bash/results/updated_dataset.csv"
data = pd.read_csv(data_path)

# Extract model names from columns
models = [col[11:] for col in data.columns if col.startswith("prediction_")]
print(models)
# Perform significance testing
results = perform_significance_testing(data, models)

# Save results
output_path = "/Users/leasyrin/Downloads/IUB_PhD/CS/L715:B659/Final_project/code_for_bash/results/race_significance_results.csv"
results.to_csv(output_path, index=False)
print(f"Significance testing results saved to {output_path}")
