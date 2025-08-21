import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from scipy.signal import correlate
import os

# ==============================
# Ultra-Script with Auto Sync + Robust Diagnostics (CLEANED)
# ==============================
# - Loads Calibration + Trial prototype CSVs and Fitbit CSV
# - Estimates optimal time offset via cross-correlation, then refines with a grid-search
# - Shifts Fitbit data to align with prototype (positive lag = shift Fitbit forward)
# - Computes stats (incl. high-confidence & low-motion subsets) and saves plots and CSVs
# - Extra robust against tz issues, nulls, poor overlap, and indentation bugs

# ------------
# Config
# ------------
CALIBRATION_PATH = "Calibration_skin/trial_data_all.csv"
TRIAL_PATH = "Skin_trial/trial_data_all.csv"
FITBIT_PATH = "heart_rate_2025-08-20.csv"
OUTPUT_DIR = "analysis_outputs"
PLOT_DPI = 140
CONF_THRESH = 90            # high-confidence cutoff
GRID_LAG_RANGE = 180        # seconds to search around auto offset
GRID_LAG_STEP = 1           # seconds
MIN_OVERLAP = 60            # min aligned samples to accept a metric
MERGE_TOLERANCE_SEC = 2     # nearest-neighbour tolerance
RESAMPLE_RATE = "1S"       # resampling for cross-corr and alignment

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ------------
# Helpers
# ------------


def load_prototype_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "py_timestamp" in df.columns:
        df["py_timestamp"] = pd.to_numeric(df["py_timestamp"], errors="coerce")
        df = df.dropna(subset=["py_timestamp"]).copy()
        # Convert epoch seconds -> UTC, drop tz -> naive UTC
        dt = pd.to_datetime(df["py_timestamp"], unit="s", utc=True).dt.tz_convert(
            "UTC").dt.tz_localize(None)
        df.insert(0, "dt_index", dt)
        df = df.set_index("dt_index").sort_index()
    else:
        raise ValueError("Prototype CSV must contain 'py_timestamp'.")

    # Numeric coercion for key columns
    for col in ["PPG_HR", "PPG_O2", "PPG_Conf", "MPU_AccelX", "MPU_AccelY", "MPU_AccelZ"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_fitbit_csv(path: str) -> pd.DataFrame:
    fb = pd.read_csv(path)
    fb["timestamp"] = pd.to_datetime(
        fb["timestamp"], errors="coerce", utc=True)
    fb = fb.dropna(subset=["timestamp"]).copy()
    fb["timestamp"] = fb["timestamp"].dt.tz_convert("UTC").dt.tz_localize(None)
    fb = fb.set_index("timestamp").sort_index()

    if "beats per minute" in fb.columns:
        fb = fb.rename(columns={"beats per minute": "Fitbit_HR"})
    elif "bpm" in fb.columns:
        fb = fb.rename(columns={"bpm": "Fitbit_HR"})
    else:
        raise ValueError(
            "Fitbit CSV must contain 'beats per minute' or 'bpm'.")

    fb["Fitbit_HR"] = pd.to_numeric(fb["Fitbit_HR"], errors="coerce")
    fb = fb.dropna(subset=["Fitbit_HR"])  # keep valid readings only
    return fb


def estimate_offset(proto: pd.Series, fitbit: pd.Series) -> int:
    """Estimate lag (seconds) where proto aligns best with Fitbit using cross-correlation.
    Positive lag means shift Fitbit FORWARD by that many seconds to align.
    """
    # Resample to 1 Hz
    proto_hr = proto.resample(RESAMPLE_RATE).mean().interpolate()
    fitbit_hr = fitbit.resample(RESAMPLE_RATE).mean().interpolate()

    # Overlapping window
    start = max(proto_hr.index.min(), fitbit_hr.index.min())
    end = min(proto_hr.index.max(), fitbit_hr.index.max())
    if pd.isna(start) or pd.isna(end) or start >= end:
        return 0
    proto_hr = proto_hr.loc[start:end]
    fitbit_hr = fitbit_hr.loc[start:end]

    # Intersect exactly
    idx = proto_hr.index.intersection(fitbit_hr.index)
    if len(idx) < 2:
        return 0

    pv = proto_hr.loc[idx].values
    fv = fitbit_hr.loc[idx].values
    pv = pv - np.nanmean(pv)
    fv = fv - np.nanmean(fv)

    corr = correlate(pv, fv, mode="full")
    lags = np.arange(-len(pv) + 1, len(pv))
    best_lag = lags[int(np.nanargmax(corr))]
    return int(best_lag)


def align_and_merge(proto: pd.DataFrame, fitbit: pd.DataFrame) -> pd.DataFrame:
    fb_resampled = fitbit[["Fitbit_HR"]].resample(
        RESAMPLE_RATE).mean().interpolate()
    proto = proto.sort_index().dropna(subset=["PPG_HR"]).copy()
    fb_resampled = fb_resampled.sort_index().copy()

    merged = pd.merge_asof(
        proto,
        fb_resampled,
        left_index=True,
        right_index=True,
        direction="nearest",
        tolerance=pd.Timedelta(seconds=MERGE_TOLERANCE_SEC)
    )

    # Motion magnitude if available
    if set(["MPU_AccelX", "MPU_AccelY", "MPU_AccelZ"]).issubset(merged.columns):
        ax = pd.to_numeric(merged["MPU_AccelX"], errors="coerce")
        ay = pd.to_numeric(merged["MPU_AccelY"], errors="coerce")
        az = pd.to_numeric(merged["MPU_AccelZ"], errors="coerce")
        merged["motion_mag"] = np.sqrt(ax**2 + ay**2 + az**2)

    return merged.dropna(subset=["PPG_HR", "Fitbit_HR"])  # keep aligned rows


def safe_corr(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    if len(x) < 2 or len(y) < 2:
        return np.nan
    try:
        r, _ = pearsonr(x, y)
    except Exception:
        r = np.nan
    return r


def compute_stats(df: pd.DataFrame, label: str = "") -> dict:
    stats = {"Label": label}
    if df is None or df.empty or len(df) < MIN_OVERLAP:
        # still return label so printing/export works
        stats.update({
            "N": 0,
            "Mean Proto HR": np.nan,
            "SD Proto HR": np.nan,
            "Mean Fitbit HR": np.nan,
            "MAE": np.nan,
            "RMSE": np.nan,
            "Correlation r": np.nan,
            "Slope a": np.nan,
            "Intercept b": np.nan,
            ">50% Conf %": np.nan,
            ">90% Conf %": np.nan,
            "Mean SpO2": np.nan,
            "SD SpO2": np.nan,
            "HC_N": 0, "HC_MAE": np.nan, "HC_RMSE": np.nan, "HC_r": np.nan,
            "LM_N": 0, "LM_MAE": np.nan, "LM_RMSE": np.nan, "LM_r": np.nan,
        })
        return stats

    # Subsets
    hc = df[df.get("PPG_Conf", 0) >= CONF_THRESH]
    if "motion_mag" in df:
        motion_thresh = df["motion_mag"].median()
        lm = df[df["motion_mag"] <= motion_thresh]
    else:
        lm = pd.DataFrame()

    def _metrics(sub):
        if sub is None or sub.empty or len(sub) < MIN_OVERLAP:
            return {
                "N": len(sub) if sub is not None else 0,
                "Mean Proto HR": np.nan,
                "SD Proto HR": np.nan,
                "Mean Fitbit HR": np.nan,
                "MAE": np.nan,
                "RMSE": np.nan,
                "Correlation r": np.nan,
                "Slope a": np.nan,
                "Intercept b": np.nan,
            }
        ph = sub["PPG_HR"].astype(float)
        fh = sub["Fitbit_HR"].astype(float)
        corr = safe_corr(fh.dropna().values, ph.dropna().values)
        try:
            a, b = np.polyfit(fh.values, ph.values, 1)
        except Exception:
            a, b = np.nan, np.nan
        return {
            "N": len(sub),
            "Mean Proto HR": float(ph.mean()),
            "SD Proto HR": float(ph.std(ddof=1)),
            "Mean Fitbit HR": float(fh.mean()),
            "MAE": float(np.mean(np.abs(fh - ph))),
            "RMSE": float(np.sqrt(np.mean((fh - ph) ** 2))),
            "Correlation r": float(corr) if not np.isnan(corr) else np.nan,
            "Slope a": float(a) if not np.isnan(a) else np.nan,
            "Intercept b": float(b) if not np.isnan(b) else np.nan,
        }

    base = _metrics(df)
    confm = _metrics(hc)
    lowmm = _metrics(lm)

    stats.update({
        **base,
        ">50% Conf %": float((df["PPG_Conf"] > 50).mean() * 100) if "PPG_Conf" in df else np.nan,
        ">90% Conf %": float((df["PPG_Conf"] > 90).mean() * 100) if "PPG_Conf" in df else np.nan,
        "Mean SpO2": float(df["PPG_O2"].mean()) if "PPG_O2" in df else np.nan,
        "SD SpO2": float(df["PPG_O2"].std(ddof=1)) if "PPG_O2" in df else np.nan,
        # Subsets
        "HC_N": confm["N"], "HC_MAE": confm["MAE"], "HC_RMSE": confm["RMSE"], "HC_r": confm["Correlation r"],
        "LM_N": lowmm["N"], "LM_MAE": lowmm["MAE"], "LM_RMSE": lowmm["RMSE"], "LM_r": lowmm["Correlation r"],
    })

    return stats


def print_stats(stats: dict):
    print(f"\n=== {stats.get('Label', 'Unnamed')} ===")
    keys = [
        "N", "Mean Proto HR", "SD Proto HR", "Mean Fitbit HR", "MAE", "RMSE", "Correlation r", "Slope a", "Intercept b",
        ">50% Conf %", ">90% Conf %", "Mean SpO2", "SD SpO2",
        "HC_N", "HC_MAE", "HC_RMSE", "HC_r", "LM_N", "LM_MAE", "LM_RMSE", "LM_r"
    ]
    for k in keys:
        v = stats.get(k, np.nan)
        if isinstance(v, (int, float, np.floating)) and not np.isnan(v):
            print(f"{k}: {v:.3f}")
        else:
            print(f"{k}: {v}")


def save_stats(rows, name):
    if not rows:
        return
    out = pd.DataFrame(rows)
    path = os.path.join(OUTPUT_DIR, name)
    out.to_csv(path, index=False)
    print(f"Saved: {path}")


def plot_timeseries(df, title, name):
    if df is None or df.empty:
        print(f"[plot] Skipped {title} (no data)")
        return
    plt.figure(figsize=(12, 5))
    plt.plot(df.index, df["PPG_HR"], label="Prototype HR")
    plt.plot(df.index, df["Fitbit_HR"], label="Fitbit HR", alpha=0.7)
    plt.legend()
    plt.xlabel("Time")
    plt.ylabel("Heart Rate (bpm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=PLOT_DPI)
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, name)}")
    plt.close()


def plot_scatter(df, title, name):
    if df is None or df.empty:
        print(f"[plot] Skipped {title} (no data)")
        return
    plt.figure(figsize=(6, 6))
    plt.scatter(df["Fitbit_HR"], df["PPG_HR"], alpha=0.6)
    mn, mx = float(df["Fitbit_HR"].min()), float(df["Fitbit_HR"].max())
    plt.plot([mn, mx], [mn, mx], linestyle='--')
    plt.xlabel("Fitbit HR (bpm)")
    plt.ylabel("Prototype HR (bpm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=PLOT_DPI)
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, name)}")
    plt.close()


def plot_bland_altman(df, title, name):
    if df is None or df.empty:
        print(f"[plot] Skipped {title} (no data)")
        return
    proto = df["PPG_HR"].astype(float)
    fitbit = df["Fitbit_HR"].astype(float)
    mean_vals = (fitbit + proto) / 2
    diff_vals = (proto - fitbit)
    bias = float(diff_vals.mean())
    loa = 1.96 * float(diff_vals.std(ddof=1)) if len(diff_vals) > 1 else np.nan

    plt.figure(figsize=(8, 6))
    plt.scatter(mean_vals, diff_vals, alpha=0.6)
    if not np.isnan(bias):
        plt.axhline(bias, linestyle='--')
    if not np.isnan(loa):
        plt.axhline(bias + loa, linestyle='--')
        plt.axhline(bias - loa, linestyle='--')
    plt.xlabel("Mean HR (bpm)")
    plt.ylabel("Prototype - Fitbit (bpm)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, name), dpi=PLOT_DPI)
    print(f"Saved plot: {os.path.join(OUTPUT_DIR, name)}")
    plt.close()


# ------------
# Main workflow
# ------------
if __name__ == "__main__":
    # Load
    proto_calib = load_prototype_csv(CALIBRATION_PATH)
    proto_trial = load_prototype_csv(TRIAL_PATH)
    fitbit = load_fitbit_csv(FITBIT_PATH)

    # Quick diagnostics
    print("CALIB prototype range:", proto_calib.index.min(),
          "->", proto_calib.index.max())
    print("TRIAL prototype range:", proto_trial.index.min(),
          "->", proto_trial.index.max())
    print("Fitbit range:", fitbit.index.min(), "->", fitbit.index.max())

    # --- Auto-sync: cross-correlation then grid search refinement ---
    offset_seconds = estimate_offset(
        proto_trial["PPG_HR"], fitbit["Fitbit_HR"])
    print(
        f"[Auto-sync] Cross-corr offset: {offset_seconds} s (positive = shift Fitbit forward)")

    best_rmse = np.inf
    best_lag = offset_seconds
    for lag in range(offset_seconds - GRID_LAG_RANGE, offset_seconds + GRID_LAG_RANGE + 1, GRID_LAG_STEP):
        fb_shift = fitbit.copy()
        fb_shift.index = fb_shift.index + pd.to_timedelta(lag, unit="s")
        test_merge = align_and_merge(proto_trial, fb_shift)
        if len(test_merge) >= MIN_OVERLAP:
            rmse = float(np.sqrt(np.mean((test_merge["Fitbit_HR"].astype(
                float) - test_merge["PPG_HR"].astype(float))**2)))
            if rmse < best_rmse:
                best_rmse = rmse
                best_lag = lag
    print(
        f"[Auto-sync] Grid-search best lag: {best_lag} s (RMSE={best_rmse:.3f})")

    # Apply best lag to Fitbit
    fitbit.index = fitbit.index + pd.to_timedelta(best_lag, unit="s")

    # Align datasets
    df_calib = align_and_merge(proto_calib, fitbit)
    df_trial = align_and_merge(proto_trial, fitbit)

    # Stats collection
    all_stats = []
    phase_stats = []

    s_cal = compute_stats(df_calib, label="Calibration (Skin Contact)")
    print_stats(s_cal)
    all_stats.append(s_cal)

    s_all = compute_stats(df_trial, label="Skin Trial (Overall)")
    print_stats(s_all)
    all_stats.append(s_all)

    if not df_trial.empty:
        t0 = df_trial.index.min()
        phases = [
            ("Walking", t0, t0 + pd.Timedelta(minutes=5)),
            ("Jogging", t0 + pd.Timedelta(minutes=5), t0 + pd.Timedelta(minutes=10)),
            ("Rest",    t0 + pd.Timedelta(minutes=10),
             t0 + pd.Timedelta(minutes=15)),
        ]
        for name, a, b in phases:
            sub = df_trial[(df_trial.index >= a) & (df_trial.index < b)]
            ps = compute_stats(sub, label=f"Skin Trial – {name}")
            print_stats(ps)
            phase_stats.append(ps)

    # Save stats
    save_stats(all_stats, "summary_overall.csv")
    save_stats(phase_stats, "summary_per_phase.csv")

    # Plots
    plot_timeseries(
        df_trial, "Prototype vs Fitbit HR – Full Trial (Aligned)", "trial_timeseries.png")
    plot_scatter(df_trial, "Correlation – Full Trial (Aligned)",
                 "trial_scatter.png")
    plot_bland_altman(
        df_trial, "Bland–Altman – Full Trial (Aligned)", "trial_bland_altman.png")

    plot_timeseries(df_calib, "Calibration – Prototype vs Fitbit HR (Aligned)",
                    "calibration_timeseries.png")
    plot_scatter(df_calib, "Calibration – Correlation (Aligned)",
                 "calibration_scatter.png")
    plot_bland_altman(
        df_calib, "Calibration – Bland–Altman (Aligned)", "calibration_bland_altman.png")
