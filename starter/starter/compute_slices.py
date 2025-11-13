import pandas as pd
import pickle
from ml.model import inference, compute_model_metrics
from ml.data import process_data


def compute_slice_metrics(data, model, categorical_features, label, encoder, lb):
    """
    Compute model performance on slices of data for categorical features.

    For each categorical feature, compute metrics for each unique value.
    """
    results = []

    # Loop through each categorical feature
    for feature in categorical_features:
        unique_values = data[feature].unique()

        # For each unique value of the feature
        for value in unique_values:
            slice_data = data[data[feature] == value]

            # Skip if slice is too small
            if len(slice_data) < 10:
                continue

            # Process the slice
            X_slice, y_slice, _, _ = process_data(
                slice_data,
                categorical_features=categorical_features,
                label=label,
                training=False,
                encoder=encoder,
                lb=lb,
            )

            # Get predictions and metrics
            preds = inference(model, X_slice)
            precision, recall, fbeta = compute_model_metrics(y_slice, preds)

            results.append(
                {
                    "feature": feature,
                    "value": value,
                    "n_samples": len(slice_data),
                    "precision": precision,
                    "recall": recall,
                    "fbeta": fbeta,
                }
            )

    return results


if __name__ == "__main__":
    # Load data
    data = pd.read_csv("../data/census.csv")

    # Clean data
    data.columns = data.columns.str.strip()
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = data[col].str.strip()

    # Load model and encoders
    with open("../model/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("../model/encoder.pkl", "rb") as f:
        encoder = pickle.load(f)
    with open("../model/lb.pkl", "rb") as f:
        lb = pickle.load(f)

    # Define categorical features
    cat_features = [
        "workclass",
        "education",
        "marital-status",
        "occupation",
        "relationship",
        "race",
        "sex",
        "native-country",
    ]

    # Compute slice metrics
    results = compute_slice_metrics(data, model, cat_features, "salary", encoder, lb)

    # Write to file
    df_results = pd.DataFrame(results)
    df_results.to_csv("slice_output.txt", index=False, sep="\t")

    print(f"Slice metrics computed for {len(results)} slices")
    print("Results written to slice_output.txt")
