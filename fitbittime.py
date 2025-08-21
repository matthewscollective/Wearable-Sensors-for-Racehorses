import pandas as pd
import numpy as np
from scipy.signal import correlate

# === File paths ===
# or Calibration_skin/trial_data_all.csv
proto_file = "Skin_trial/trial_data_all.csv"
fitbit_file = "heart_rate_2025-08-20.csv"

# === Load prototype ===
proto = pd.read_csv(proto_file)
if "py_timestamp" in proto.columns:
    proto["timestamp"] = pd.to_datetime(
        proto["py_timestamp"], unit="s", utc=True)
elif "real_time" in proto.columns:
    # fallback if no py_timestamp
    proto["timestamp"] = pd.to_datetime(
        "2025-08-20 " + proto["real_time"], utc=True)

proto = proto.set_index("timestamp").sort_index()
proto_hr = proto["PPG_HR"].astype(float)

# === Load Fitbit ===
fitbit = pd.read_csv(fitbit_file)
fitbit["timestamp"] = pd.to_datetime(fitbit["timestamp"], utc=True)
fitbit = fitbit.set_index("timestamp").sort_index()
fitbit_hr = fitbit["beats per minute"].astype(float)

# === Resample both to 1-second bins ===
proto_hr = proto_hr.resample("1S").mean().interpolate()
fitbit_hr = fitbit_hr.resample("1S").mean().interpolate()

# === Align overlapping window ===
start = max(proto_hr.index.min(), fitbit_hr.index.min())
end = min(proto_hr.index.max(), fitbit_hr.index.max())
proto_hr = proto_hr.loc[start:end]
fitbit_hr = fitbit_hr.loc[start:end]

# === Ensure equal length ===
if len(proto_hr) != len(fitbit_hr):
    idx = proto_hr.index.intersection(fitbit_hr.index)
    proto_hr = proto_hr.loc[idx]
    fitbit_hr = fitbit_hr.loc[idx]

# === Cross-correlation to estimate offset ===
proto_vals = proto_hr.values - np.nanmean(proto_hr.values)
fitbit_vals = fitbit_hr.values - np.nanmean(fitbit_hr.values)

corr = correlate(proto_vals, fitbit_vals, mode="full")
lags = np.arange(-len(proto_vals)+1, len(proto_vals))
best_lag = lags[np.nanargmax(corr)]

# Convert lag to seconds (since resampled at 1 Hz)
offset_seconds = int(best_lag)

print("=== Estimated Time Offset ===")
print(f"Best lag: {best_lag} seconds")
if offset_seconds > 0:
    print(f"→ Prototype HR is ahead of Fitbit by {offset_seconds} seconds")
elif offset_seconds < 0:
    print(f"→ Prototype HR is behind Fitbit by {-offset_seconds} seconds")
else:
    print("→ No offset detected (already aligned)")
