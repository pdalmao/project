import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler # Using RobustScaler as it's less sensitive to outliers than StandardScaler
import joblib
import os
import argparse
import sys

# --- Configuration ---
DEFAULT_MODEL_DIR = "ml_models"

# Expected CSV columns - adjust if format differs
COLUMNS = ["Time Stamp", "ID", "Extended", "Dir", "Bus", "LEN"] + [f"D{i}" for i in range(1, 9)]
PAYLOAD_FEATURES = [f"D{i}" for i in range(1, 9)] # D1 to D8
MIN_MSGS_FOR_FREQ = 2 # Minimum data points needed for a valid model training

# --- Preprocessing Function ---
def preprocess(df):
    """Converts relevant columns (ID, Timestamp, Payload) to numeric types."""
    print("Preprocessing data...")
    # Safely convert hex string or float to int, handling errors
    def safe_hex_to_int(x):
        try:
            if isinstance(x, float): return int(x)
            return int(str(x), 16)
        except (ValueError, TypeError): return None

    # Convert ID column, dropping rows with invalid IDs
    df['ID'] = df['ID'].apply(safe_hex_to_int)
    initial_rows = len(df)
    df.dropna(subset=['ID'], inplace=True)
    df['ID'] = df['ID'].astype(int)
    if initial_rows > len(df):
        print(f"[WARN] Dropped {initial_rows - len(df)} rows due to invalid ID format.")

    # Convert payload bytes (D1-D8) to numeric, fill NaNs/missing with 0
    for i in range(1, 9):
        col = f'D{i}'
        if col in df.columns:
            # Handle strings (assume hex), numbers, and NaNs
            df[col] = df[col].apply(lambda x: int(str(x), 16) if isinstance(x, str) else (int(x) if pd.notnull(x) else 0))
        else:
            df[col] = 0 # Create column if missing

    # Convert timestamp
    try:
        df['Time Stamp'] = df['Time Stamp'].astype(float)
    except ValueError as e:
        print(f"[ERROR] Could not convert 'Time Stamp' column to float: {e}. Check data.")
        return None

    print("Preprocessing complete.")
    # Return only necessary columns
    return df[['Time Stamp', 'ID'] + PAYLOAD_FEATURES].copy()

# --- Time Delta Calculation ---
def add_time_deltas(df_id):
    """Calculates time difference between consecutive messages for a specific ID."""
    if df_id.empty: return df_id
    df_id = df_id.sort_values('Time Stamp') # Ensure correct order
    df_id['Time Delta'] = df_id['Time Stamp'].diff().fillna(0.0001) # Fill first NaN, avoid zero delta
    df_id.loc[df_id['Time Delta'] <= 0, 'Time Delta'] = 0.0001 # Ensure positive delta
    return df_id

# --- Main Training Function ---
def train_and_save_models(normal_files, target_ids, model_dir, svm_nu):
    """Loads data, trains OneClassSVM models per ID, and saves them."""
    all_normal_dfs = []
    print("Loading normal data files...")
    for file_path in normal_files:
        print(f"Loading: {file_path}")
        try:
            # Load CSV, assuming first 6 cols + 8 data cols based on COLUMNS list
            # low_memory=False helps with mixed types. 'warn' on bad lines. Skip header row.
            num_prefix_cols = len(COLUMNS) - len(PAYLOAD_FEATURES)
            df = pd.read_csv(file_path, names=COLUMNS, usecols=range(num_prefix_cols + 8), skiprows=1, on_bad_lines='warn', low_memory=False)
            if not df.empty: all_normal_dfs.append(df)
            else: print(f"[WARN] File {file_path} was empty or unreadable.")
        except FileNotFoundError: print(f"[ERROR] Normal data file not found: {file_path}")
        except Exception as e: print(f"[ERROR] Failed to read CSV file {file_path}: {e}")
        # Continue processing other files even if one fails

    if not all_normal_dfs:
        print("[ERROR] No valid normal data loaded. Cannot train.")
        return

    print("Concatenating data...")
    normal_df_combined = pd.concat(all_normal_dfs, ignore_index=True)
    print(f"Total rows for training: {len(normal_df_combined)}")

    normal_df_processed = preprocess(normal_df_combined)
    if normal_df_processed is None or normal_df_processed.empty:
        print("[ERROR] Preprocessing failed or resulted in empty DataFrame. Cannot train.")
        return

    os.makedirs(model_dir, exist_ok=True) # Create output directory if needed
    print(f"Models will be saved to: {model_dir}")

    # --- Loop through each target ID for training ---
    for current_id in target_ids:
        print(f"\n--- Training models for ID 0x{current_id:X} ---")
        df_id_normal = normal_df_processed[normal_df_processed['ID'] == current_id].copy()

        if len(df_id_normal) < MIN_MSGS_FOR_FREQ:
            print(f"Skipping ID 0x{current_id:X}: Not enough data ({len(df_id_normal)} rows).")
            continue

        df_id_normal = add_time_deltas(df_id_normal) # Calculate time deltas for this ID

        # --- Time Model ---
        print(f"Training Time model for 0x{current_id:X}...")
        try:
            X_time_normal = df_id_normal[['Time Delta']].values
            if X_time_normal.shape[0] >= MIN_MSGS_FOR_FREQ:
                scaler_time = RobustScaler()
                X_time_normal_scaled = scaler_time.fit_transform(X_time_normal)

                model_time = OneClassSVM(kernel='rbf', gamma='scale', nu=svm_nu)
                model_time.fit(X_time_normal_scaled)

                joblib.dump(scaler_time, os.path.join(model_dir, f'scaler_0x{current_id:X}_time.joblib'))
                joblib.dump(model_time, os.path.join(model_dir, f'model_0x{current_id:X}_time.joblib'))
                print(f"Saved Time model and scaler for 0x{current_id:X}")
            else: print(f"Skipping Time model for 0x{current_id:X}: Insufficient valid data points.")
        except Exception as e: print(f"[ERROR] Failed training Time model for 0x{current_id:X}: {e}")

        # --- Payload Model ---
        print(f"Training Payload model for 0x{current_id:X}...")
        try:
            X_payload_normal = df_id_normal[PAYLOAD_FEATURES].values
            if X_payload_normal.shape[0] >= MIN_MSGS_FOR_FREQ:
                scaler_payload = RobustScaler()
                X_payload_normal_scaled = scaler_payload.fit_transform(X_payload_normal)

                model_payload = OneClassSVM(kernel='rbf', gamma='scale', nu=svm_nu)
                model_payload.fit(X_payload_normal_scaled)

                joblib.dump(scaler_payload, os.path.join(model_dir, f'scaler_0x{current_id:X}_payload.joblib'))
                joblib.dump(model_payload, os.path.join(model_dir, f'model_0x{current_id:X}_payload.joblib'))
                print(f"Saved Payload model and scaler for 0x{current_id:X}")
            else: print(f"Skipping Payload model for 0x{current_id:X}: Insufficient valid data points.")
        except Exception as e: print(f"[ERROR] Failed training Payload model for 0x{current_id:X}: {e}")

    print("\n--- Model Training Script Finished ---")

# --- Argument Parsing & Execution ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train OneClassSVM Anomaly Detection Models for CAN IDs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Shows defaults in help message
        )
    parser.add_argument("normal_data_files", nargs='+', # Accept one or more files
                        help="Path(s) to the CSV file(s) containing normal CAN traffic.")
    parser.add_argument("-m", "--models_dir", default=DEFAULT_MODEL_DIR,
                        help="Directory to save trained models and scalers.")
    parser.add_argument("-ids", "--target_ids", required=True, nargs='+',
                        help="List of CAN IDs (hex or dec) to train models for (e.g., 0x188 244 0x1A0).")
    parser.add_argument("--nu", type=float, default=0.005, # Default nu, assumes small fraction of anomalies/noise
                        help="OneClassSVM nu parameter (expected fraction of outliers). Controls sensitivity.")
    args = parser.parse_args()

    # Convert IDs from string (hex/dec) to integer list
    try:
        target_ids_int = [int(id_str, 0) for id_str in args.target_ids] # Base 0 auto-detects hex
    except ValueError as e:
        print(f"[ERROR] Invalid CAN ID format: {e}. Use hex (0x...) or decimal.")
        sys.exit(1)

    # Validate nu parameter
    if not (0 < args.nu <= 1.0):
        print(f"[ERROR] nu must be > 0 and <= 1. Got: {args.nu}")
        sys.exit(1)

    train_and_save_models(args.normal_data_files, target_ids_int, args.models_dir, args.nu)