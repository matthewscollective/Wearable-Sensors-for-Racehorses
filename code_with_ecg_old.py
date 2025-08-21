"""
trial_run.py

Reads single-line CSV from Arduino (as produced by the Arduino sketch above).
Waits for the first valid PPG (HR 40-200 and SpO2 >= 70), then runs a 3-minute trial
(you can change TRIAL_DURATION). During the trial it collects samples, prints progress
every PROGRESS_INTERVAL seconds, and at the end performs:
 - ECG QRS detection (bandpass + find_peaks) -> ECG HR (instantaneous & mean)
 - Per-sample alignment of ECG HR to PPG samples (nearest-beat)
 - Statistical summaries for: all valid samples, conf>=50%, conf>=90%
 - Bland-Altman and correlation plots comparing ECG HR vs PPG HR
 - Saves CSVs and plots into OUTPUT_FOLDER
"""

import socket
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks
import pandas as pd
from scipy import stats

# # === COM CONFIG ===
# COM_PORT = "COM3"         # set to your COM port
# BAUD = 115200
# TRIAL_DURATION = 180      # seconds (3 minutes)
# PROGRESS_INTERVAL = 10    # seconds
# OUTPUT_FOLDER = "trial_outputs"
# os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === WIFI CONFIG ===
ESP32_IP = "192.168.1.43"   # Arduino Serial Monitor IP
PORT = 8080
TRIAL_DURATION = 180      # seconds
PROGRESS_INTERVAL = 10    # seconds
OUTPUT_FOLDER = "trial_outputs"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# Filters
PPG_HR_MIN, PPG_HR_MAX = 40, 200
SPO2_MIN = 70

# ECG processing params
ECG_HIGHPASS = 5.0    # Hz
ECG_LOWPASS = 40.0    # Hz
ECG_FILTER_ORDER = 3

# === Networking helpers ===


def connect_esp32(ip, port):
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    print(f"Connecting to ESP32 at {ip}:{port} ...")
    sock.connect((ip, port))
    sock.settimeout(1.0)
    print("Connected.")
    return sock


def read_line(sock, buffer):
    """
    Reads one line (ending with \n) from TCP socket.
    Returns (line, buffer).
    """
    while True:
        try:
            data = sock.recv(1024).decode("utf-8")
        except socket.timeout:
            return None, buffer
        if not data:
            return None, buffer
        buffer += data
        if "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            return line.strip(), buffer

# Helper: bandpass filter


def butter_bandpass(lowcut, highcut, fs, order=3):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass_filter(data, lowcut, highcut, fs, order=3):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    return filtfilt(b, a, data)

# Simple Bland-Altman function


def bland_altman_plot(a, b, title, fname):
    a = np.asarray(a)
    b = np.asarray(b)
    diff = a - b
    mean = (a + b) / 2.0
    md = np.mean(diff)
    sd = np.std(diff)
    plt.figure(figsize=(6, 5))
    plt.scatter(mean, diff, alpha=0.6)
    plt.axhline(md, color='gray', linestyle='--', label=f'mean diff={md:.2f}')
    plt.axhline(md + 1.96*sd, color='red', linestyle='--', label=f'+1.96 SD')
    plt.axhline(md - 1.96*sd, color='red', linestyle='--', label=f'-1.96 SD')
    plt.xlabel('Mean')
    plt.ylabel('Difference')
    plt.title(title)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(fname)
    plt.close()

# Parse a CSV line into dict using header mapping


def parse_csv_line(line, header_cols):
    parts = line.strip().split(',')
    if len(parts) != len(header_cols):
        return None
    d = {}
    for k, v in zip(header_cols, parts):
        try:
            d[k] = float(v)
        except:
            d[k] = np.nan
    return d

# Serial
# # Wait for header line from Arduino and return list of column names


# def read_header(ser, timeout=5.0):
#     t0 = time.time()
#     while time.time() - t0 < timeout:
#         raw = ser.readline()
#         if not raw:
#             continue
#         try:
#             line = raw.decode('utf-8', errors='ignore').strip()
#         except:
#             continue
#         if line.startswith("timestamp_ms"):
#             cols = [c.strip() for c in line.split(',')]
#             return cols
#     return None

def read_header(sock, buffer, timeout=5.0):
    t0 = time.time()
    while time.time() - t0 < timeout:
        line, buffer = read_line(sock, buffer)
        if line and line.startswith("timestamp_ms"):
            cols = [c.strip() for c in line.split(',')]
            return cols, buffer
    return None, buffer
# QRS detection -> return peak indices (in samples) and instantaneous HR series


def detect_qrs_and_hr(ecg_signal, ecg_times_s, fs):
    # ecg_signal: 1D numpy array, ecg_times_s: corresponding times in seconds
    if len(ecg_signal) < 50:
        return [], np.array([])

    # Detrend: subtract median
    ecg = np.array(ecg_signal) - np.median(ecg_signal)

    # Bandpass filter
    try:
        ecg_f = bandpass_filter(
            ecg, ECG_HIGHPASS, ECG_LOWPASS, fs, order=ECG_FILTER_ORDER)
    except Exception as e:
        # fallback to raw if filtering fails
        ecg_f = ecg

    # Square the signal to exaggerate peaks (optional)
    ecg_sq = ecg_f ** 2

    # Dynamic threshold: mean + k*std
    thresh = np.mean(ecg_sq) + 0.5 * np.std(ecg_sq)

    # Minimum distance between peaks (in samples). min 0.35s (max HR ~170)
    min_dist_samples = int(0.35 * fs)

    peaks, props = find_peaks(
        ecg_sq, height=thresh, distance=min_dist_samples, prominence=np.std(ecg_sq)*0.3)
    peak_times = ecg_times_s[peaks]

    # Compute RR intervals and instantaneous HR
    if len(peak_times) < 2:
        return peaks, np.array([])

    rr = np.diff(peak_times)   # seconds
    inst_hr = 60.0 / rr        # bpm, length = len(peaks) - 1

    # We'll create a per-peak HR aligned at peak_times[1:]
    peak_hr_times = peak_times[1:]
    peak_hr_values = inst_hr

    return peaks, (peak_hr_times, peak_hr_values, ecg_f)

# Main experiment run


def main():
    # Serial setup
    # print(f"Opening serial port {COM_PORT} @ {BAUD}")
    # ser = serial.Serial(COM_PORT, BAUD, timeout=1)
    # time.sleep(2)

    # # read header
    # header = read_header(ser, timeout=10)
    # if header is None:
    #     print("ERROR: No header found from Arduino. Is the Arduino sketch running and printing a header?")
    #     ser.close()
    #     return
    # print("Header columns:", header)

    # TCP setup
    sock = connect_esp32(ESP32_IP, PORT)
    buffer = ""

    # read header
    header, buffer = read_header(sock, buffer, timeout=10)
    if header is None:
        print("ERROR: No header found from ESP32.")
        sock.close()
        return
    print("Header columns:", header)

    # We'll collect samples into list of dicts
    samples = []
    first_valid_time = None
    start_time = None
    last_progress_time = None

    print("Place finger on the PPG sensor... Waiting for first valid PPG reading.")
    t0 = time.time()
    while True:
        # Serial
        # raw = ser.readline()
        # if not raw:
        #     continue
        # try:
        #     line = raw.decode('utf-8', errors='ignore').strip()
        # except:
        #     continue
        # # parse
        # rec = parse_csv_line(line, header)
        # if rec is None:
        #     # ignore malformed lines, but could print debug
        #     # print("Malformed:", line)
        #     continue

        # # Keep each record as dict; keys are header column names
        # samples.append(rec)

        # TCP
        line, buffer = read_line(sock, buffer)
        if not line:
            continue
        rec = parse_csv_line(line, header)
        if rec is None:
            continue
        samples.append(rec)

        # Check for first valid PPG to start trial
        if first_valid_time is None:
            ppg_hr = rec.get('PPG_HR', 0.0)
            ppg_o2 = rec.get('PPG_O2', 0.0)
            if (not np.isnan(ppg_hr)) and (PPG_HR_MIN <= ppg_hr <= PPG_HR_MAX) and (not np.isnan(ppg_o2)) and (ppg_o2 >= SPO2_MIN):
                first_valid_time = time.time()
                start_time = first_valid_time
                last_progress_time = first_valid_time
                print(
                    f"First valid PPG sample found (HR={ppg_hr:.1f}, O2={ppg_o2:.1f}). Starting trial.")
                break

        # Safety: if waiting too long, inform user
        if time.time() - t0 > 30 and first_valid_time is None:
            print("Still waiting for valid PPG reading... keep finger steady.")
            t0 = time.time()

    # Collect for TRIAL_DURATION seconds from start_time
    print(f"Collecting for {TRIAL_DURATION} seconds...")
    while True:
        # Serial
        # raw = ser.readline()
        # if not raw:
        #     continue
        # try:
        #     line = raw.decode('utf-8', errors='ignore').strip()
        # except:
        #     continue
        # rec = parse_csv_line(line, header)
        # if rec is None:
        #     continue
        # samples.append(rec)

        # TCP
        line, buffer = read_line(sock, buffer)
        if not line:
            continue
        rec = parse_csv_line(line, header)
        if rec is None:
            continue
        samples.append(rec)

        # progress updates
        elapsed = time.time() - start_time
        if time.time() - last_progress_time >= PROGRESS_INTERVAL:
            df_temp = pd.DataFrame(samples)
            total = len(df_temp)
            valid_mask = (df_temp['PPG_HR'].notna()) & (df_temp['PPG_HR'] >= PPG_HR_MIN) & (
                df_temp['PPG_HR'] <= PPG_HR_MAX) & (df_temp['PPG_O2'].notna()) & (df_temp['PPG_O2'] >= SPO2_MIN)
            valid_count = valid_mask.sum()
            mean_hr = df_temp.loc[valid_mask, 'PPG_HR'].mean(
            ) if valid_count > 0 else np.nan
            mean_o2 = df_temp.loc[valid_mask, 'PPG_O2'].mean(
            ) if valid_count > 0 else np.nan
            latest_temp = df_temp['MPU_Temp'].dropna(
            ).iloc[-1] if 'MPU_Temp' in df_temp.columns and df_temp['MPU_Temp'].dropna().size > 0 else np.nan
            print(f"[{int(elapsed)} s] samples={total}, valid={valid_count}, mean_PPG_HR={mean_hr:.2f}, mean_SpO2={mean_o2:.2f}, latest_temp={latest_temp:.2f}")
            last_progress_time = time.time()

        if elapsed >= TRIAL_DURATION:
            print("Trial period ended.")
            break

    # ser.close()
    # print("Serial closed. Processing data...")
    sock.close()
    print("Socket closed. Processing data...")

    # Convert to DataFrame
    df = pd.DataFrame(samples)

    # Convert timestamp to seconds
    if 'timestamp_ms' in df.columns:
        df['t_s'] = df['timestamp_ms'] / 1000.0
    else:
        df['t_s'] = np.arange(len(df)) * 0.01

    # Clean numeric columns: replace NaN or invalid
    # Ensure PPG fields numeric
    df['PPG_HR'] = pd.to_numeric(df['PPG_HR'], errors='coerce')
    df['PPG_O2'] = pd.to_numeric(df['PPG_O2'], errors='coerce')
    df['PPG_Conf'] = pd.to_numeric(df['PPG_Conf'], errors='coerce')

    # Build arrays for ECG processing (use ECG_Centered if present)
    if 'ECG_Centered' in df.columns:
        ecg_vals = pd.to_numeric(df['ECG_Centered'], errors='coerce').values
    else:
        ecg_vals = pd.to_numeric(df['ECG_Raw'], errors='coerce').values - \
            np.nanmedian(pd.to_numeric(df['ECG_Raw'], errors='coerce').values)

    ecg_times = df['t_s'].values

    # Estimate sampling frequency from timestamp differences (use median dt)
    dt_median = np.median(np.diff(ecg_times))
    if dt_median <= 0 or np.isnan(dt_median):
        fs = 100.0
    else:
        fs = 1.0 / dt_median
    print(
        f"Estimated ECG sampling rate: {fs:.1f} Hz (median dt {dt_median:.4f} s)")

    # Detect QRS peaks & ECG HR
    peak_indices, ecg_hr_info = detect_qrs_and_hr_wrapper(
        ecg_vals, ecg_times, fs)

    # ecg_hr_info returns (peak_hr_times, peak_hr_values, ecg_filtered)
    if ecg_hr_info is None or len(ecg_hr_info[0]) == 0:
        print("ECG QRS detection found no beats. ECG HR unavailable.")
        ecg_mean_hr = np.nan
        ecg_hr_at_ppg = np.full(len(df), np.nan)
    else:
        peak_hr_times, peak_hr_values, ecg_filtered = ecg_hr_info
        ecg_mean_hr = np.mean(peak_hr_values)
        # For each PPG sample, find nearest ECG instantaneous HR (by time)
        ppg_times = df['t_s'].values
        ecg_hr_at_ppg = np.full_like(ppg_times, np.nan, dtype=float)
        # peak_hr_times array length = len(peak_indices)-1, times aligned to second and subsequent peaks
        for i, t in enumerate(ppg_times):
            # find nearest peak_hr_time index
            idx = np.argmin(np.abs(peak_hr_times - t))
            # allow a 2s window, otherwise ignore
            if np.abs(peak_hr_times[idx] - t) <= 2.0:
                ecg_hr_at_ppg[i] = peak_hr_values[idx]

    print(f"ECG mean HR (from peaks): {ecg_mean_hr:.2f} bpm" if not np.isnan(
        ecg_mean_hr) else "ECG mean HR: NaN")

    # Create filtered masks
    mask_valid = (df['PPG_HR'].notna()) & (df['PPG_HR'] >= PPG_HR_MIN) & (
        df['PPG_HR'] <= PPG_HR_MAX) & (df['PPG_O2'].notna()) & (df['PPG_O2'] >= SPO2_MIN)
    df_valid = df.loc[mask_valid].copy()
    df_valid['ECG_HR_at_sample'] = ecg_hr_at_ppg[mask_valid.values]

    # Confidence subsets
    df_conf50 = df_valid[df_valid['PPG_Conf'] >= 50].copy()
    df_conf90 = df_valid[df_valid['PPG_Conf'] >= 90].copy()

    # Save CSVs
    df_valid.to_csv(os.path.join(
        OUTPUT_FOLDER, "trial_data_all_valid.csv"), index=False)
    df_conf50.to_csv(os.path.join(
        OUTPUT_FOLDER, "trial_data_conf50.csv"), index=False)
    df_conf90.to_csv(os.path.join(
        OUTPUT_FOLDER, "trial_data_conf90.csv"), index=False)
    print("Saved CSVs.")

    # Stats helper
    def print_stats(dframe, label):
        if dframe.empty:
            print(f"{label}: NO SAMPLES")
            return
        print(f"\n=== Stats: {label} ===")
        print(f"Samples: {len(dframe)}")
        print(
            f"PPG HR mean/std: {dframe['PPG_HR'].mean():.2f} / {dframe['PPG_HR'].std():.2f}")
        print(
            f"SpO2 mean/std: {dframe['PPG_O2'].mean():.2f} / {dframe['PPG_O2'].std():.2f}")
        print(f"PPG Conf mean: {dframe['PPG_Conf'].mean():.2f}")
        # ECG HR comparisons if available
        if 'ECG_HR_at_sample' in dframe.columns and dframe['ECG_HR_at_sample'].notna().any():
            ecg_vals_local = dframe['ECG_HR_at_sample'].dropna().values
            # compare length
            paired_mask = dframe['ECG_HR_at_sample'].notna()
            paired_ppg = dframe.loc[paired_mask, 'PPG_HR'].values
            paired_ecg = dframe.loc[paired_mask, 'ECG_HR_at_sample'].values
            if len(paired_ecg) > 2:
                corr = np.corrcoef(paired_ppg, paired_ecg)[0, 1]
                t_stat, p_val = stats.ttest_rel(
                    paired_ppg, paired_ecg, nan_policy='omit')
                print(
                    f"ECG vs PPG - paired samples: {len(paired_ecg)}, Pearson corr={corr:.3f}, paired t-test p={p_val:.4f}")
            else:
                print("Not enough paired ECG-PPG HR samples for correlation/test.")
        else:
            print("No ECG HR data aligned to PPG samples.")

    # Print stats for each group
    print_stats(df_valid, "All valid samples")
    print_stats(df_conf50, "Confidence >= 50%")
    print_stats(df_conf90, "Confidence >= 90%")

    # Plotting: HR, SpO2, Accel, ECG waveform (filtered) + peaks
    plt.figure(figsize=(14, 12))
    ax1 = plt.subplot(4, 1, 1)
    ax1.plot(df_valid['t_s'].values,
             df_valid['PPG_HR'].values, 'r.-', label='PPG HR')
    if not np.isnan(ecg_mean_hr):
        ax1.hlines(ecg_mean_hr, df_valid['t_s'].min(), df_valid['t_s'].max(
        ), colors='g', linestyles='--', label=f'ECG mean HR {ecg_mean_hr:.1f}')
    ax1.set_ylabel('HR (bpm)')
    ax1.legend()
    ax1.grid(True)

    ax2 = plt.subplot(4, 1, 2, sharex=ax1)
    ax2.plot(df_valid['t_s'].values,
             df_valid['PPG_O2'].values, 'b.-', label='SpO2')
    ax2.set_ylabel('SpO2 (%)')
    ax2.legend()
    ax2.grid(True)

    ax3 = plt.subplot(4, 1, 3, sharex=ax1)
    ax3.plot(df_valid['t_s'].values,
             df_valid['MPU_AccelX'].values, label='Accel X')
    ax3.plot(df_valid['t_s'].values,
             df_valid['MPU_AccelY'].values, label='Accel Y')
    ax3.plot(df_valid['t_s'].values,
             df_valid['MPU_AccelZ'].values, label='Accel Z')
    ax3.set_ylabel('Accel (m/s^2)')
    ax3.legend()
    ax3.grid(True)

    ax4 = plt.subplot(4, 1, 4, sharex=ax1)
    # plot filtered ECG waveform across full record if available
    if 'ecg_filtered' in locals() and len(ecg_filtered) > 0:
        # use ecg_times and ecg_filtered
        ax4.plot(ecg_times, ecg_filtered, 'k-', label='ECG (filtered)')
        if len(peak_indices) > 0:
            ax4.plot(ecg_times[peak_indices], ecg_filtered[peak_indices],
                     'ro', label='Detected R peaks')
    else:
        # fallback: plot centered ECG raw at PPG sample times (coarse)
        if 'ECG_Centered' in df.columns:
            ax4.plot(df['t_s'].values, df['ECG_Centered'].values,
                     'k-', label='ECG (centered)')
    ax4.set_ylabel('ECG (ADC)')
    ax4.set_xlabel('Time (s)')
    ax4.legend()
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_FOLDER, "trial_plots_full.png"))
    plt.show()

    # Bland-Altman ECG HR vs PPG HR (pair only samples where ECG HR available)
    paired_mask = (~np.isnan(df_valid['ECG_HR_at_sample'])) & (
        ~np.isnan(df_valid['PPG_HR']))
    if paired_mask.sum() >= 10:
        ppg_vals = df_valid.loc[paired_mask, 'PPG_HR'].values
        ecg_vals_pairs = df_valid.loc[paired_mask, 'ECG_HR_at_sample'].values
        bland_altman_plot(ppg_vals, ecg_vals_pairs, "PPG HR vs ECG HR", os.path.join(
            OUTPUT_FOLDER, "bland_altman_ecg_vs_ppg.png"))
        print("Saved Bland-Altman plot for ECG vs PPG HR.")
    else:
        print("Not enough paired ECG-PPG HR samples for Bland-Altman plot.")

    print("Done. Outputs saved to", OUTPUT_FOLDER)

# Wrapper to call QRS detection and return peaks and HR info


def detect_qrs_and_hr_wrapper(ecg_vals, ecg_times, fs):
    # convert nan to 0 and ensure arrays
    ecg_vals = np.asarray(ecg_vals, dtype=float)
    ecg_times = np.asarray(ecg_times, dtype=float)

    # Remove NaNs for processing (simple approach: mask)
    mask = ~np.isnan(ecg_vals)
    if mask.sum() < 50:
        return [], None
    ecg_clean = ecg_vals[mask]
    times_clean = ecg_times[mask]

    # High-level detection
    peaks, hr_info = detect_qrs_and_hr(ecg_clean, times_clean, fs)
    if len(peaks) == 0 or hr_info is None:
        return [], None

    # peaks are indices into ecg_clean; we want indices into original ecg array for plotting
    # Build ecg_filtered for plot (filter applied inside detect)
    # For simplicity, reconstruct filtered signal via bandpass filter
    try:
        ecg_filtered = bandpass_filter(
            ecg_clean - np.median(ecg_clean), ECG_HIGHPASS, ECG_LOWPASS, fs, order=ECG_FILTER_ORDER)
    except:
        ecg_filtered = ecg_clean - np.median(ecg_clean)

    # To map peaks back to original times, peaks indices correspond to times_clean[peaks]
    # Return peaks indices for times_clean (we'll use times_clean[peaks] for plotting)
    return peaks, (hr_info[0], hr_info[1], ecg_filtered)

# Lower-level QRS detector used above


def detect_qrs_and_hr(ecg_signal, ecg_times_s, fs):
    # See earlier function; keep as same logic here
    if len(ecg_signal) < 50:
        return [], None
    ecg = np.array(ecg_signal) - np.median(ecg_signal)
    try:
        ecg_f = bandpass_filter(
            ecg, ECG_HIGHPASS, ECG_LOWPASS, fs, order=ECG_FILTER_ORDER)
    except:
        ecg_f = ecg
    ecg_sq = ecg_f ** 2
    thresh = np.mean(ecg_sq) + 0.5 * np.std(ecg_sq)
    min_dist_samples = int(0.35 * fs)
    peaks, props = find_peaks(
        ecg_sq, height=thresh, distance=min_dist_samples, prominence=np.std(ecg_sq)*0.3)
    peak_times = ecg_times_s[peaks]
    if len(peak_times) < 2:
        return peaks, None
    rr = np.diff(peak_times)
    inst_hr = 60.0 / rr
    peak_hr_times = peak_times[1:]
    peak_hr_values = inst_hr
    return peaks, (peak_hr_times, peak_hr_values, ecg_f)


if __name__ == "__main__":
    main()
