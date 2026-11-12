import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

DATA_FOLDER = "data"

def sine_model(t, A, B, T, phi):
    return A + B * np.sin(2 * np.pi * t / T + phi)

def load_local_data(folder):
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Папка '{folder}' не найдена. Создайте её и положите туда скачанный файл.")
    
    files = glob.glob(os.path.join(folder, "*.txt"))
    if not files:
        raise FileNotFoundError(f"В папке '{folder}' нет .txt файлов. Скачайте данные с сайта NMDB и положите их туда.")
    
    print(f"[i] Найдено файлов: {len(files)}")
    records = []
    sentinels = {-999.0, -9999.0, -99999.0, 9999.0, 99999.0}

    for file_path in files:
        print(f"  -> Чтение {os.path.basename(file_path)}...")
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
        raise RuntimeError("Не удалось извлечь данные из файлов. Возможно, формат файла не ASCII.")
        
    s = pd.Series(dict(records)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s

def preprocess(series: pd.Series) -> pd.Series:
    med = float(series.median())
    return series.fillna(med)

def main():
    print("=" * 66)
    print(f" Лабораторная работа: Нелинейная регрессия (Вариант 7: SOPO)")
    print("=" * 66)

    try:
        series = load_local_data(DATA_FOLDER)
    except Exception as e:
        print(f"[!] Ошибка: {e}")
        return

    series = preprocess(series)
    print(f"[i] Всего загружено {len(series)} суточных отсчётов.")

    X = np.arange(1, len(series) + 1, dtype=float)
    Y = series.values

    initial_guess = [np.mean(Y), np.std(Y), 365.25, 0.0]

    print("[i] Выполнение нелинейной регрессии...")
    try:
        popt, pcov = curve_fit(sine_model, X, Y, p0=initial_guess, maxfev=20000)
    except Exception as e:
        print(f"[!] Ошибка подгонки модели: {e}")
        return

    A_fit, B_fit, T_fit, phi_fit = popt
    print(f"\n[i] Полученные параметры модели:")
    print(f"    A (средний уровень) = {A_fit:.2f}")
    print(f"    B (амплитуда)       = {B_fit:.2f}")
    print(f"    T (период)          = {T_fit:.2f} дней")
    print(f"    phi (фаза)          = {phi_fit:.2f} рад")

    Y_pred = sine_model(X, *popt)
    ss_res = np.sum((Y - Y_pred) ** 2)
    ss_tot = np.sum((Y - np.mean(Y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    print(f"[i] Коэффициент детерминации R^2 = {r_squared:.4f}")

    plt.figure(figsize=(12, 6))
    plt.scatter(X, Y, color='gray', alpha=0.5, label='Исходные данные (NMDB)', s=15)
    plt.plot(X, Y_pred, color='red', linewidth=2, label=f'Модель: A + B * sin(2 * pi * t / T + phi)')
    
    plt.title(f"Нелинейная регрессия данных станции SOPO за 2015 год")
    plt.xlabel("Время (дни)")
    plt.ylabel("Скорость счета (отн. ед.)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("regression_result.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()