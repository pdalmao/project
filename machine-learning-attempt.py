import pandas as pd
import matplotlib.pyplot as plt
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import RobustScaler

# --- Constants & Config ---
COLUMNS = ["Time Stamp", "ID", "Extended", "Dir", "Bus", "LEN",
           "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8"]
FEATURES = [f"D{i}" for i in range(1, 9)]
RANDOM_STATE = 42
TARGET_ID = 0x244  # Target ID for analysis

# --- Data Loading ---
def load_data(normal_file, anomaly_file):
    try:
        normal_df = pd.read_csv(normal_file, names=COLUMNS, usecols=range(14), skiprows=1, on_bad_lines='skip')
        anomaly_df = pd.read_csv(anomaly_file, names=COLUMNS, usecols=range(14), skiprows=1, on_bad_lines='skip')
        return normal_df, anomaly_df
    except Exception as e:
        print(f"Error reading CSV: {e}")
        exit()

# --- Preprocessing ---
def preprocess(df):
    def safe_hex_to_int(x):
        try:
            return int(str(x), 16)
        except:
            return None
    df['ID'] = df['ID'].apply(safe_hex_to_int)
    df.dropna(subset=['ID'], inplace=True)
    df['ID'] = df['ID'].astype(int)

    for i in range(1, 9):
        col = f'D{i}'
        if col in df.columns:
            df[col] = df[col].apply(lambda x: int(str(x), 16) if pd.notnull(x) and isinstance(x, str) else (int(x) if pd.notnull(x) else 0))
        else:
            df[col] = 0
    return df.drop(columns=['Extended', 'Dir', 'Bus'], errors='ignore')

# --- One-Class SVM Training & Prediction ---
def train_predict_svm(X_train, X_test):
    model = OneClassSVM(kernel='rbf', gamma=0.1, nu=0.005)
    model.fit(X_train)
    scores = model.decision_function(X_test)
    predictions = model.predict(X_test)
    return scores, predictions

# --- Time Delta Feature Creation ---
def add_time_deltas(df):
    df['Time Stamp'] = df['Time Stamp'].astype(float)
    df['Time Delta'] = df['Time Stamp'].diff().fillna(0)
    return df

# --- Plotting ---
def plot_results(df, byte_col, id_hex):
    fig, axes = plt.subplots(1, 2, figsize=(24, 5))

    # Normalize anomaly scores between 0 and 1 for color mapping
    # Added small epsilon to prevent division by zero if all scores are identical
    df['norm_score'] = (df['score'] - df['score'].min()) / (df['score'].max() - df['score'].min() + 1e-9)
    df['norm_time_score'] = (df['time_score'] - df['time_score'].min()) / (df['time_score'].max() - df['time_score'].min() + 1e-9)

    # --- Payload Anomaly Plot ---
    # Plot all points using the score for color
    sc0 = axes[0].scatter(df.index, df[byte_col],
                          c=df['norm_score'], cmap='coolwarm_r', s=50)

    axes[0].set_title(f"Payload Anomaly Score (ID 0x{id_hex:X}, Byte: {byte_col})")
    axes[0].set_xlabel("Message Index")
    axes[0].set_ylabel(f"{byte_col} Value")
    cbar0 = plt.colorbar(sc0, ax=axes[0])
    cbar0.set_label("Normalized Payload Anomaly Score") # Updated label
    axes[0].grid(True)

    # --- Timestamp Anomaly Plot ---
    # Plot all points using the time score for color
    sc1 = axes[1].scatter(df.index, df['Time Delta'],
                          c=df['norm_time_score'], cmap='coolwarm_r', s=50)

    axes[1].set_title(f"Timestamp Anomaly Score (ID 0x{id_hex:X})")
    axes[1].set_xlabel("Message Index")
    axes[1].set_ylabel("Time Delta (s)")
    cbar1 = plt.colorbar(sc1, ax=axes[1])
    cbar1.set_label("Normalized Timestamp Anomaly Score") # Updated label
    # Removed the explicit legend as color now represents the score gradient
    axes[1].grid(True)

    plt.tight_layout()
    # Consider creating the figs directory if it doesn't exist
    import os
    os.makedirs("figs", exist_ok=True)
    plt.savefig(f"figs/anomaly_analysis_{id_hex}_{byte_col}.png")
    plt.show()




# --- Main Analysis for Target ID ---
def analyze_target_id(normal_df, anomaly_df):
    print(f"\n=== Analyzing CAN ID: 0x{TARGET_ID:X} ===")
    df_normal = normal_df[normal_df['ID'] == TARGET_ID].copy()
    df_anomaly = anomaly_df[anomaly_df['ID'] == TARGET_ID].copy()

    if df_normal.empty or df_anomaly.empty:
        print("Not enough data for this ID, skipping.")
        return

    df_normal = add_time_deltas(df_normal)
    df_anomaly = add_time_deltas(df_anomaly)

    # Analyze each byte separately
    for byte in FEATURES:
        # if df_normal[byte].nunique() <= 1:
        #     continue  # Skip unchanging bytes

        print(f"\nAnalyzing byte: {byte}")

        X_normal = df_normal[[byte]].values
        X_anomaly = df_anomaly[[byte]].values

        scaler = RobustScaler()
        X_normal_scaled = scaler.fit_transform(X_normal)
        X_anomaly_scaled = scaler.transform(X_anomaly)

        scores, preds = train_predict_svm(X_normal_scaled, X_anomaly_scaled)
        df_anomaly['prediction'] = preds
        df_anomaly['score'] = scores

        # Time delta anomaly detection
        time_scaler = RobustScaler()
        X_time_normal = time_scaler.fit_transform(df_normal[['Time Delta']])
        X_time_anomaly = time_scaler.transform(df_anomaly[['Time Delta']])
        time_scores, time_preds = train_predict_svm(X_time_normal, X_time_anomaly)
        df_anomaly['time_prediction'] = time_preds
        df_anomaly['time_score'] = time_scores

        # Combined anomalies
        both_anomalies = df_anomaly[(df_anomaly['prediction'] == -1) & (df_anomaly['time_prediction'] == -1)]

        print(f"Payload anomalies: {(df_anomaly['prediction'] == -1).sum()} | "
              f"Timestamp anomalies: {(df_anomaly['time_prediction'] == -1).sum()} | "
              f"Both: {len(both_anomalies)}")

        if not both_anomalies.empty:
            print("\nSample combined anomalies:")
            print(both_anomalies[['Time Stamp', 'Time Delta', 'ID', byte, 'score', 'time_score']].head().to_string(index=False))

        # Visualization
        plot_results(df_anomaly.copy(), byte, TARGET_ID)

# --- Execute ---
if __name__ == "__main__":
    normal_df, anomaly_df = load_data('data/normal-data-longer', 'data/244-normal-running')
    print("Preprocessing...")
    normal_df = preprocess(normal_df)
    anomaly_df = preprocess(anomaly_df)
    analyze_target_id(normal_df, anomaly_df)
