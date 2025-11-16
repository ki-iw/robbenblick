import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import argparse
from pathlib import Path
from robbenblick import logger


def evaluate_counts(gt_csv_path, pred_csv_path):
    """
    Loads ground truth and prediction CSVs, merges them, computes aggregate
    metrics, and saves a detailed per-image error report.
    """
    # Check if ground truth file exists
    if not gt_csv_path.exists():
        logger.error(f"Ground truth file not found: {gt_csv_path}")
        return

    # Check if prediction file exists
    if not pred_csv_path.exists():
        logger.error(f"Prediction file not found: {pred_csv_path}")
        return

    try:
        # Load the CSV files
        gt_df = pd.read_csv(gt_csv_path)
        pred_df = pd.read_csv(pred_csv_path)

        logger.info(f"Ground truth file loaded: {len(gt_df)} entries.")
        logger.info(f"Prediction file loaded: {len(pred_df)} entries.")

        # Ensure the count columns are numeric, coercing errors to NaN
        gt_df["count"] = pd.to_numeric(gt_df["count"], errors="coerce")
        pred_df["count"] = pd.to_numeric(pred_df["count"], errors="coerce")

        # Merge the DataFrames using 'image_name' as the key
        eval_df = pd.merge(
            gt_df,
            pred_df,
            on="image_name",
            suffixes=("_gt", "_pred"),  # Add suffixes to distinguish count columns
        )

        # Check for missing values after merging or conversion
        initial_rows = len(eval_df)
        eval_df = eval_df.dropna(subset=["count_gt", "count_pred"])
        final_rows = len(eval_df)

        # Handle case with no matching rows
        if final_rows == 0:
            logger.error(
                "No matching 'image_name' entries found between files or all values were invalid. Evaluation not possible."
            )
            return

        # Warn if rows were dropped
        if initial_rows > final_rows:
            logger.warning(
                f"{initial_rows - final_rows} rows were removed due to missing values (NaN)."
            )

        logger.info(f"Evaluation will be performed on {final_rows} matching images.")

        # Calculate error (prediction - ground_truth)
        eval_df["error"] = eval_df["count_pred"] - eval_df["count_gt"]
        # Calculate absolute error for sorting
        eval_df["absolute_error"] = eval_df["error"].abs()

        # Sort by the worst offenders (highest absolute error)
        eval_df_sorted = eval_df.sort_values(by="absolute_error", ascending=False)

        # Save the new CSV in the same directory as the prediction CSV
        output_csv_path = pred_csv_path.parent / "evaluation_details.csv"
        try:
            # Define columns for the report
            report_columns = [
                "image_name",
                "count_gt",
                "count_pred",
                "error",
                "absolute_error",
            ]
            eval_df_sorted.to_csv(output_csv_path, index=False, columns=report_columns)
            logger.info(f"Detailed per-image report saved to: {output_csv_path}")
        except Exception as e:
            logger.error(f"Failed to save detailed evaluation CSV: {e}")

        y_true = eval_df["count_gt"]
        y_pred = eval_df["count_pred"]

        # Calculate the metrics
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)

        print("\n--- Aggregate Counting Metrics ---")
        print("=" * 35)
        print(f"Images compared: {final_rows}")
        print(f"  Mean Absolute Error (MAE):   {mae:.4f}")
        print(f"  Mean Squared Error (MSE):    {mse:.4f}")
        print(f"  Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"  R-squared (R²):            {r2:.4f}")
        print("=" * 35)

        N_WORST = 10
        print(f"\n--- Top {N_WORST} Worst Performing Images (by Abs Error) ---")
        # Get the top N_WORST rows
        top_n_worst = eval_df_sorted.head(N_WORST)
        # Select and format columns for printing
        print_df = top_n_worst[["image_name", "count_gt", "count_pred", "error"]]
        # Use to_string for nice console alignment
        print(print_df.to_string(index=False))
        print("=" * 50)

    except pd.errors.EmptyDataError:
        logger.error("One of the CSV files is empty.")
    except KeyError as e:
        logger.error(
            f"Missing column in CSV file: {e}. Ensure both 'image_name' and 'count' columns exist."
        )
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")


def main():
    # Set up the argument parser
    parser = argparse.ArgumentParser(
        description="Evaluates counting performance by comparing two CSV files."
    )

    # Argument for the ground truth CSV file
    parser.add_argument(
        "--gt-csv",
        type=str,
        required=True,
        help="Path to the ground truth CSV file (e.g., ground_truth_counts.csv).",
    )

    # Argument for the prediction CSV file
    parser.add_argument(
        "--pred-csv",
        type=str,
        required=True,
        help="Path to the prediction CSV file (e.g., detection_counts.csv).",
    )

    args = parser.parse_args()

    # Convert string paths to Path objects (more robust for handling spaces)
    gt_path = Path(args.gt_csv)
    pred_path = Path(args.pred_csv)

    # Run the evaluation
    evaluate_counts(gt_path, pred_path)


if __name__ == "__main__":
    main()
