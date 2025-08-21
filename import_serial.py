"""
PPG + Motion Data Logger & Plotter
==================================

This script connects to an ESP32 (Wi-Fi or Serial), streams PPG and motion sensor data,
logs the results to CSV, and generates a suite of diagnostic plots.

Features:
- Waits until both motion and PPG signals are detected before starting a trial.
- Filters out invalid PPG readings (e.g., HR=0 or SpO2 < threshold).
- Saves raw and filtered signals for later analysis.
- Generates multiple plots (raw + filtered, combined overlays, QC views).
"""

import datetime
import socket
import serial
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ============================================================
# CONFIGURATION
# ============================================================

CONNECTION_MODE = "wifi"   # "wifi" or "serial"

# Wi-Fi connection
ESP32_IP = "192.168.1.45"
ESP32_PORT = 8080

# Serial connection
SERIAL_PORT = "COM5"
BAUD_RATE = 115200

# Trial settings
TRIAL_DURATION = 300       # seconds AFTER valid signals detected
PROGRESS_INTERVAL = 15     # seconds between status prints

# PPG strict validity thresholds (used for filtering)
PPG_HR_MIN, PPG_HR_MAX = 30, 220
SPO2_MIN = 70

# PPG loose detection thresholds (used for detecting finger presence)
PPG_DETECT_HR_MIN = 25
PPG_DETECT_O2_MIN = 60
PPG_DETECT_CONF_MIN = 10
PPG_DETECT_STREAK = 3  # consecutive samples required

# CSV columns (expected order from Arduino/ESP32 stream)
COLUMNS = [
    "arduino_ms", "PPG_HR", "PPG_O2", "PPG_Conf", "PPG_Status",
    "MPU_AccelX", "MPU_AccelY", "MPU_AccelZ",
    "MPU_GyroX", "MPU_GyroY", "MPU_GyroZ",
    "MPU_Temp", "MPU_VelX", "MPU_VelY", "MPU_VelZ"
]

# ============================================================
# UTILITIES
# ============================================================


def parse_csv_line(line: str):
    """Parse a CSV line into a dict of floats (NaN if missing)."""
    try:
        parts = line.strip().split(",")
        if len(parts) != len(COLUMNS):
            return None
        return {k: float(v) if v else np.nan for k, v in zip(COLUMNS, parts)}
    except Exception:
        return None


def read_line(sock, buffer: str):
    """Read one full line from a Wi-Fi socket, handling buffering."""
    while True:
        chunk = sock.recv(1024)
        if not chunk:
            return None, buffer
        buffer += chunk.decode("utf-8", errors="ignore")
        if "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            return line.strip(), buffer


def ensure_dir(path: str):
    """Ensure a directory exists."""
    if not os.path.exists(path):
        os.makedirs(path)

# ============================================================
# PLOTTING
# ============================================================


def save_and_plot(df: pd.DataFrame, outdir="White_brown_tape_trial"):
    """
    Save CSV and generate plots:
    - Raw PPG
    - Filtered PPG
    - Raw Motion
    - Motion + Raw PPG overlay
    - Motion + Filtered PPG overlay
    - Combined raw vs filtered plots
    - QC plots (confidence, dropout inspection)
    """
    ensure_dir(outdir)

    # ---- Add human-readable time ----
    df["real_time"] = df["py_timestamp"].apply(
        lambda t: datetime.datetime.fromtimestamp(t).strftime("%H:%M:%S")
    )

    # ---- Save CSVs ----
    csv_all = os.path.join(outdir, "trial_data_all.csv")
    csv_90 = os.path.join(outdir, "trial_data_conf90.csv")
    csv_50 = os.path.join(outdir, "trial_data_conf50.csv")

    df.to_csv(csv_all, index=False)
    df[df["PPG_Conf"] >= 90].to_csv(csv_90, index=False)
    df[df["PPG_Conf"] >= 50].to_csv(csv_50, index=False)

    print(f"Saved data -> {csv_all}")
    print(f"Saved data -> {csv_90}")
    print(f"Saved data -> {csv_50}")

    # ---- Create validity mask ----
    valid_mask = (df['PPG_HR'].between(PPG_HR_MIN, PPG_HR_MAX)
                  ) & (df['PPG_O2'] >= SPO2_MIN)
    df_valid = df[valid_mask]

    # ---- Raw PPG ----
    plt.figure(figsize=(12, 6))
    plt.plot(df["py_timestamp"], df["PPG_HR"], label="HR (raw)")
    plt.plot(df["py_timestamp"], df["PPG_O2"], label="SpO2 (raw)")
    plt.xlabel("Time (s)")
    plt.ylabel("Raw PPG values")
    plt.legend()
    plt.title("Raw PPG Readings (includes dropouts)")
    plt.savefig(os.path.join(outdir, "raw_ppg.png"))
    plt.close()

    # ---- Filtered PPG ----
    plt.figure(figsize=(12, 6))
    plt.plot(df_valid["py_timestamp"],
             df_valid["PPG_HR"], label="HR (filtered)")
    plt.plot(df_valid["py_timestamp"],
             df_valid["PPG_O2"], label="SpO2 (filtered)")
    plt.xlabel("Time (s)")
    plt.ylabel("Filtered PPG values")
    plt.legend()
    plt.title("Filtered PPG Readings (valid only)")
    plt.savefig(os.path.join(outdir, "filtered_ppg.png"))
    plt.close()

    # ---- Raw Motion ----
    plt.figure(figsize=(12, 6))
    for col in ["MPU_AccelX", "MPU_AccelY", "MPU_AccelZ"]:
        plt.plot(df["py_timestamp"], df[col], label=col)
    for col in ["MPU_GyroX", "MPU_GyroY", "MPU_GyroZ"]:
        plt.plot(df["py_timestamp"], df[col], label=col, linestyle="--")
    plt.xlabel("Time (s)")
    plt.ylabel("Motion values")
    plt.legend()
    plt.title("Raw Motion (Accel + Gyro)")
    plt.savefig(os.path.join(outdir, "raw_motion.png"))
    plt.close()

    # ---- Motion + Raw PPG ----
    plt.figure(figsize=(12, 6))
    plt.plot(df["py_timestamp"], df["MPU_AccelX"], label="AccelX")
    plt.plot(df["py_timestamp"], df["MPU_AccelY"], label="AccelY")
    plt.plot(df["py_timestamp"], df["MPU_AccelZ"], label="AccelZ")
    plt.plot(df["py_timestamp"], df["PPG_HR"], label="HR (raw)", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Mixed signals")
    plt.legend()
    plt.title("Motion + Raw PPG Overlay")
    plt.savefig(os.path.join(outdir, "motion_raw_ppg.png"))
    plt.close()

    # ---- Motion + Filtered PPG ----
    plt.figure(figsize=(12, 6))
    plt.plot(df["py_timestamp"], df["MPU_AccelX"], label="AccelX")
    plt.plot(df["py_timestamp"], df["MPU_AccelY"], label="AccelY")
    plt.plot(df["py_timestamp"], df["MPU_AccelZ"], label="AccelZ")
    plt.plot(df_valid["py_timestamp"], df_valid["PPG_HR"],
             label="HR (filtered)", alpha=0.7)
    plt.plot(df_valid["py_timestamp"], df_valid["PPG_O2"],
             label="SpO2 (filtered)", alpha=0.7)
    plt.xlabel("Time (s)")
    plt.ylabel("Mixed signals")
    plt.legend()
    plt.title("Motion + Filtered PPG Overlay")
    plt.savefig(os.path.join(outdir, "motion_filtered_ppg.png"))
    plt.close()

    # ---- Confidence QC ----
    plt.figure(figsize=(12, 6))
    plt.plot(df["py_timestamp"], df["PPG_Conf"], label="Confidence")
    plt.xlabel("Time (s)")
    plt.ylabel("Confidence score")
    plt.legend()
    plt.title("PPG Confidence vs Time")
    plt.savefig(os.path.join(outdir, "ppg_confidence.png"))
    plt.close()

    # ---- All-in-One Combined ----
    plt.figure(figsize=(14, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df["py_timestamp"], df["PPG_HR"], label="HR (raw)")
    plt.plot(df_valid["py_timestamp"],
             df_valid["PPG_HR"], label="HR (filtered)")
    plt.legend()
    plt.ylabel("HR")

    plt.subplot(3, 1, 2)
    plt.plot(df["py_timestamp"], df["PPG_O2"], label="O2 (raw)")
    plt.plot(df_valid["py_timestamp"],
             df_valid["PPG_O2"], label="O2 (filtered)")
    plt.legend()
    plt.ylabel("SpO2")

    plt.subplot(3, 1, 3)
    for col in ["MPU_AccelX", "MPU_AccelY", "MPU_AccelZ"]:
        plt.plot(df["py_timestamp"], df[col], label=col)
    plt.legend()
    plt.ylabel("Motion (Accel)")
    plt.xlabel("Time (s)")

    plt.suptitle("All Signals (Raw + Filtered)")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "all_signals.png"))
    plt.close()

    print(f"Plots saved in {outdir}/")

# ============================================================
# MAIN LOOP
# ============================================================


def main():
    """Main entry point for data acquisition and plotting."""
    if CONNECTION_MODE == "wifi":
        print(f"Connecting to ESP32 at {ESP32_IP}:{ESP32_PORT} ...")
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect((ESP32_IP, ESP32_PORT))
        print("Connected.")
        buffer = ""
    else:
        print(f"Connecting via serial {SERIAL_PORT} ...")
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
        sock = None
        buffer = None
        print("Connected.")

    samples = []
    trial_started = False
    start_time = None
    last_progress_time = None

    # Flags for detection
    motion_seen = False
    ppg_signal_seen = False
    ppg_detect_streak = 0

    last_status_ping = time.time()
    print("Waiting for sensor data (motion + PPG)...")

    while True:
        # ---- Read a line ----
        if CONNECTION_MODE == "wifi":
            line, buffer = read_line(sock, buffer)
        else:
            raw = ser.readline()
            line = raw.decode(
                "utf-8", errors="ignore").strip() if raw else None

        if not line:
            # Status reminder if nothing yet
            if time.time() - last_status_ping > 15 and not (ppg_signal_seen and motion_seen):
                waiting_for = []
                if not motion_seen:
                    waiting_for.append("motion")
                if not ppg_signal_seen:
                    waiting_for.append("PPG")
                print(f"Still waiting for {', '.join(waiting_for)} ...")
                last_status_ping = time.time()
            continue

        # ---- Parse record ----
        rec = parse_csv_line(line)
        if rec is None:
            continue
        rec['py_timestamp'] = time.time()

        # ---- Detect motion ----
        accel_vals = [rec.get("MPU_AccelX"), rec.get(
            "MPU_AccelY"), rec.get("MPU_AccelZ")]
        gyro_vals = [rec.get("MPU_GyroX"),  rec.get(
            "MPU_GyroY"),  rec.get("MPU_GyroZ")]
        temp_val = rec.get("MPU_Temp")
        if not motion_seen:
            if any(np.isfinite(v) for v in (*accel_vals, *gyro_vals, temp_val)):
                motion_seen = True
                print("Motion data detected (Accel/Gyro/Temp available).")

        # ---- Detect PPG (loose) ----
        hr, o2, conf = rec.get("PPG_HR"), rec.get(
            "PPG_O2"), rec.get("PPG_Conf")
        sample_has_ppg = (
            np.isfinite(hr) and np.isfinite(o2) and np.isfinite(conf) and
            (hr >= PPG_DETECT_HR_MIN or o2 >= PPG_DETECT_O2_MIN) and
            (conf >= PPG_DETECT_CONF_MIN)
        )
        ppg_detect_streak = ppg_detect_streak + 1 if sample_has_ppg else 0
        if not ppg_signal_seen and ppg_detect_streak >= PPG_DETECT_STREAK:
            ppg_signal_seen = True
            print("PPG signal detected (finger on sensor).")

        # ---- Start trial (strict) ----
        is_valid_ppg = (
            np.isfinite(hr) and np.isfinite(o2) and
            (PPG_HR_MIN <= hr <= PPG_HR_MAX) and (o2 >= SPO2_MIN)
        )
        if not trial_started and motion_seen and ppg_signal_seen and is_valid_ppg:
            trial_started = True
            start_time = time.time()
            last_progress_time = start_time
            print(
                f"First valid PPG found: HR={hr:.1f}, O2={o2:.1f}. Starting trial.")

        # ---- Log samples ----
        if trial_started:
            elapsed = time.time() - start_time
            samples.append(rec)

            # Progress update
            if time.time() - last_progress_time >= PROGRESS_INTERVAL:
                df_temp = pd.DataFrame(samples)
                valid_mask = (df_temp['PPG_HR'].between(PPG_HR_MIN, PPG_HR_MAX)) & (
                    df_temp['PPG_O2'] >= SPO2_MIN)
                valid_count = valid_mask.sum()
                mean_hr = df_temp.loc[valid_mask, 'PPG_HR'].mean(
                ) if valid_count else np.nan
                mean_o2 = df_temp.loc[valid_mask, 'PPG_O2'].mean(
                ) if valid_count else np.nan
                latest_temp = df_temp['MPU_Temp'].dropna(
                ).iloc[-1] if not df_temp['MPU_Temp'].dropna().empty else np.nan
                print(f"[{int(elapsed)} s] samples={len(df_temp)}, valid={valid_count}, "
                      f"mean_HR={mean_hr:.1f}, mean_O2={mean_o2:.1f}, latest_temp={latest_temp:.1f}")
                last_progress_time = time.time()

            # End trial
            if elapsed >= TRIAL_DURATION:
                print("Trial period ended.")
                break

    # ---- Wrap up ----
    df = pd.DataFrame(samples)
    save_and_plot(df)


if __name__ == "__main__":
    main()
