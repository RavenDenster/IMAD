import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pywt
from scipy.fft import next_fast_len

DATA_FOLDER = "data"

def load_local_data(folder):
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise FileNotFoundError(f"В папке '{folder}' нет .txt файлов.")
    
    records = []
    sentinels = {-999.0, -9999.0, -99999.0, 9999.0, 99999.0}
    
    for file_path in files:
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "start_date_time" in line:
                    continue
                if ";" not in line:
                    continue
                parts = line.split(";")
                if len(parts) < 2:
                    continue
                dt_str, val_str = parts[0].strip(), parts[1].strip()
                
                if val_str == "" or val_str.lower() in ("null", "nan", "na", "-"):
                    val = np.nan
                else:
                    try:
                        val = float(val_str)
                    except ValueError:
                        continue
                if val in sentinels:
                    val = np.nan
                try:
                    dt = pd.to_datetime(dt_str)
                except Exception:
                    continue
                records.append((dt, val))
                
    if not records:
        raise RuntimeError("Не удалось извлечь данные.")
        
    s = pd.Series(dict(records)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s.fillna(s.median())

def main():
    print("=" * 66)
    print(" Лабораторная работа 5: Вейвлет-анализ (Вариант 7: SOPO, 2 мин)")
    print("=" * 66)

    try:
        series = load_local_data(DATA_FOLDER)
    except Exception as e:
        print(f"[!] Ошибка: {e}")
        return

    signal = series.values
    print(f"[i] Загружено {len(signal)} отсчетов (с разрешением 2 минуты).")

    N_fast = next_fast_len(len(signal))
    pad_total = N_fast - len(signal)
    pad_left = pad_total // 2
    pad_right = pad_total - pad_left
    
    print(f"[i] Дополняем сигнал до {N_fast} точек для быстрого FFT...")
    signal_padded = np.pad(signal, (pad_left, pad_right), mode='symmetric')

    Fs = 1/120.0 
    wavelet_name = 'morl' 
    fc = pywt.central_frequency(wavelet_name)
    
    freq_min = 0.000005
    freq_max = 0.016

    scale_min = max(1, int(fc / (freq_max * (1/Fs))))
    scale_max = int(fc / (freq_min * (1/Fs)))
    
    scales = np.arange(scale_max, scale_min - 1, -1)
    
    print("[i] Выполнение непрерывного вейвлет-преобразования (CWT)...")
    coeffs, freqs = pywt.cwt(signal_padded, scales, wavelet_name)
    
    coeffs_trimmed = coeffs[:, pad_left:-pad_right]
    image = np.abs(coeffs_trimmed)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 2]})

    time_axis = np.arange(len(signal))
    ax1.plot(time_axis, signal, color='blue', linewidth=1)
    ax1.set_title("Исходный временной ряд (SOPO, 9-14 мая 2024)", fontsize=14)
    ax1.set_ylabel("Скорость счета (отн. ед.)")
    ax1.grid(True, alpha=0.3)
    
    im = ax2.pcolormesh(time_axis, scales, image, shading='auto', cmap='jet')
    ax2.set_title("Вейвлет-скейлограмма (CWT, вейвлет 'morl')", fontsize=14)
    ax2.set_ylabel("Масштабы (Scales)")
    ax2.set_xlabel("Время")
    
    date_labels = [
        '09.05\n00:00', '09.05\n08:00', '09.05\n16:00', 
        '10.05\n00:00', '10.05\n08:00', '10.05\n16:00', 
        '11.05\n00:00', '11.05\n08:00', '11.05\n16:00', 
        '12.05\n00:00', '12.05\n08:00', '12.05\n16:00', 
        '13.05\n00:00', '13.05\n08:00', '13.05\n16:00',
        '14.05\n00:00', '14.05\n08:00', '14.05\n16:00'
    ]
    
    tick_step = 240
    tick_positions = np.arange(0, len(signal), tick_step)
    date_labels = date_labels[:len(tick_positions)]
    
    ax1.set_xticks(tick_positions)
    ax2.set_xticks(tick_positions)
    ax2.set_xticklabels(date_labels, rotation=45, ha='right')

    cbar = fig.colorbar(im, ax=ax2, pad=0.01)
    cbar.set_label('Амплитуда коэффициентов (энергия)')

    plt.tight_layout()
    plt.savefig("wavelet_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()