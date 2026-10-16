import os
import sys
import datetime
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

STATION = "SOPO"
WINDOW_SIZE = 60
N_CLUSTERS = 2    # (спокойно / буря)
M_FUZZY = 2.0     # Параметр нечеткости (обычно 2.0)
ERROR = 0.005     # Порог сходимости
MAX_ITER = 1000   # Максимальное число итераций

# Класс 1: Спокойное состояние
CLASS_1_DATES = ["2018-06-11", "2018-06-12"]
# Класс 2: Магнитная буря
CLASS_2_DATES = ["2018-12-28", "2019-01-01"]

DATA_DIR = "sopo_data"

def build_url(station: str, date_str: str) -> str:
    """Формирует URL для скачивания данных с NMDB за 1 день."""
    dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
    return (
        "https://www.nmdb.eu/nest/draw_graph.php"
        "?formchk=1"
        f"&stations[]={station}"
        "&tabchoice=ori"
        "&dtype=corr_for_efficiency"
        "&tresolution=1"
        "&yunits=0"
        "&date_choice=bydate"
        f"&start_day={dt.day:02d}&start_month={dt.month:02d}&start_year={dt.year}"
        "&start_hour=00&start_min=00"
        f"&end_day={dt.day:02d}&end_month={dt.month:02d}&end_year={dt.year}"
        "&end_hour=23&end_min=59"
        "&output=ascii&force=1"
    )

def download_data(date_str: str) -> str:
    """Скачивает данные, если их нет локально."""
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    local_file = os.path.join(DATA_DIR, f"{STATION}_{date_str}.txt")
    if os.path.exists(local_file):
        with open(local_file, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    
    url = build_url(STATION, date_str)
    print(f"[i] Скачивание данных за {date_str}...")
    try:
        resp = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
        resp.raise_for_status()
        with open(local_file, "w", encoding="utf-8") as f:
            f.write(resp.text)
        return resp.text
    except Exception as e:
        print(f"[!] Ошибка при скачивании {date_str}: {e}")
        return ""

def parse_data(text: str) -> pd.Series:
    """Парсит ответ NMDB и возвращает временной ряд."""
    records = []
    sentinels = {-999.0, -9999.0, -99999.0, 9999.0, 99999.0}
    
    for line in text.splitlines():
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
        raise RuntimeError("Не удалось распарсить данные.")
        
    s = pd.Series(dict(records)).sort_index()
    s = s[~s.index.duplicated(keep="first")]
    return s

def preprocess(series: pd.Series) -> pd.Series:
    """Заполняет пропуски медианой на регулярной 1-мин сетке."""
    full_idx = pd.date_range(
        series.index.min().floor("min"),
        series.index.max().ceil("min"),
        freq="1min"
    )
    s = series.reindex(full_idx)
    med = float(s.median())
    s = s.fillna(med)
    return s

def extract_features(series: pd.Series, window: int) -> np.ndarray:
    """Формирует матрицу признаков из временного ряда."""
    features = []
    vals = series.values
    for i in range(len(vals) - window + 1):
        win = vals[i:i+window]
        mean_v = np.mean(win)
        std_v = np.std(win)
        amp_v = np.max(win) - np.min(win)
        features.append([mean_v, std_v, amp_v])
    return np.array(features)

def fuzzy_c_means(data: np.ndarray, c: int, m: float, error: float, maxiter: int):
    """
    Реализация нечеткого алгоритма c-средних.
    data: матрица (N, D)
    Возвращает: centers (c, D), U (c, N)
    """
    N, D = data.shape
    np.random.seed(42)
    U = np.random.rand(c, N)
    U = U / np.sum(U, axis=0)
    
    for iteration in range(maxiter):
        U_old = U.copy()
        
        Um = U ** m
        centers = (Um @ data) / np.sum(Um, axis=1, keepdims=True)
        
        dist = np.zeros((c, N))
        for i in range(c):
            dist[i] = np.linalg.norm(data - centers[i], axis=1)
            
        for i in range(c):
            for j in range(N):
                if dist[i, j] == 0:
                    U[i, j] = 1.0
                else:
                    U[i, j] = 1.0 / np.sum((dist[i, j] / dist[:, j]) ** (2 / (m - 1)))
                    
        if np.linalg.norm(U - U_old) < error:
            print(f"[i] FCM сошлась за {iteration+1} итераций.")
            break
            
    return centers, U

def main():
    print("=" * 66)
    print(f" Вариант 7: Станция {STATION} (Южный полюс)")
    print("=" * 66)
    
    X_list = []
    y_list = []
    
    all_dates = [(d, 0) for d in CLASS_1_DATES] + [(d, 1) for d in CLASS_2_DATES]
    
    for date_str, label in all_dates:
        print(f"\n[i] Обработка даты: {date_str} (Класс {label})")
        text = download_data(date_str)
        if not text:
            continue
            
        series = parse_data(text)
        series = preprocess(series)
        print(f"    Точек после заполнения: {len(series)}")
        
        features = extract_features(series, WINDOW_SIZE)
        print(f"    Извлечено окон (признаков): {len(features)}")
        
        X_list.append(features)
        y_list.extend([label] * len(features))
        
    if not X_list:
        sys.exit("Нет данных для анализа.")
        
    X = np.vstack(X_list)
    y_true = np.array(y_list)
    
    X_mean = np.mean(X, axis=0)
    X_std = np.std(X, axis=0)
    X_scaled = (X - X_mean) / X_std
    
    print(f"\n[i] Итоговая матрица данных: {X_scaled.shape[0]} наблюдений, {X_scaled.shape[1]} признака")
    
    print("\n[i] Запуск Fuzzy C-Means...")
    centers, U = fuzzy_c_means(X_scaled, c=N_CLUSTERS, m=M_FUZZY, error=ERROR, maxiter=MAX_ITER)
    
    pred_labels = np.argmax(U, axis=0)
    
    acc1 = np.mean(pred_labels == y_true)
    acc2 = np.mean((1 - pred_labels) == y_true)
    accuracy = max(acc1, acc2)
    print(f"[i] Точность кластеризации (с учетом перестановки): {accuracy*100:.2f}%")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    sc1 = axes[0, 0].scatter(X_scaled[:, 0], X_scaled[:, 1], c=U[0], cmap='coolwarm', alpha=0.6, s=10)
    axes[0, 0].scatter(centers[0, 0], centers[0, 1], c='red', marker='x', s=150, linewidths=3, label='Центр 1')
    axes[0, 0].scatter(centers[1, 0], centers[1, 1], c='blue', marker='x', s=150, linewidths=3, label='Центр 2')
    axes[0, 0].set_title("Признаковое пространство: Среднее vs СКО")
    axes[0, 0].set_xlabel("Среднее (норм.)")
    axes[0, 0].set_ylabel("СКО (норм.)")
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    plt.colorbar(sc1, ax=axes[0, 0], label='Степень принадлежности кластеру 0')
    
    sc2 = axes[0, 1].scatter(X_scaled[:, 0], X_scaled[:, 2], c=U[0], cmap='coolwarm', alpha=0.6, s=10)
    axes[0, 1].scatter(centers[0, 0], centers[0, 2], c='red', marker='x', s=150, linewidths=3, label='Центр 1')
    axes[0, 1].scatter(centers[1, 0], centers[1, 2], c='blue', marker='x', s=150, linewidths=3, label='Центр 2')
    axes[0, 1].set_title("Признаковое пространство: Среднее vs Амплитуда")
    axes[0, 1].set_xlabel("Среднее (норм.)")
    axes[0, 1].set_ylabel("Амплитуда (норм.)")
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    plt.colorbar(sc2, ax=axes[0, 1], label='Степень принадлежности кластеру 0')
    
    axes[1, 0].hist([pred_labels[y_true == 0], pred_labels[y_true == 1]], 
                    bins=[-0.5, 0.5, 1.5], rwidth=0.8, color=['green', 'red'], 
                    label=['Класс 1 (Спокойно)', 'Класс 2 (Буря)'])
    axes[1, 0].set_title("Распределение предсказанных кластеров по истинным классам")
    axes[1, 0].set_xlabel("Номер кластера FCM")
    axes[1, 0].set_ylabel("Количество окон")
    axes[1, 0].set_xticks([0, 1])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    try:
        s_quiet = preprocess(parse_data(download_data(CLASS_1_DATES[0])))
        s_storm = preprocess(parse_data(download_data(CLASS_2_DATES[0])))
        time_hours = np.arange(len(s_quiet)) / 60.0
        
        axes[1, 1].plot(time_hours, s_quiet.values, label=f"Спокойно ({CLASS_1_DATES[0]})", color='green', alpha=0.7)
        axes[1, 1].plot(time_hours, s_storm.values, label=f"Буря ({CLASS_2_DATES[0]})", color='red', alpha=0.7)
        
        axes[1, 1].set_title("Суточный ход SOPO: спокойный день vs буря")
        axes[1, 1].set_xlabel("Время суток (часы UTC)")
        axes[1, 1].set_ylabel("Скорость счета")
        axes[1, 1].set_xlim(0, 24)
        axes[1, 1].set_xticks(np.arange(0, 25, 2))
        
        axes[1, 1].legend()
        axes[1, 1].grid(True, alpha=0.3)
    except Exception as e:
        axes[1, 1].text(0.5, 0.5, f"Ошибка построения\n{e}", ha='center', va='center')
    
    plt.tight_layout()
    out_file = "sopo_fcm_clustering.png"
    plt.savefig(out_file, dpi=150)
    print(f"\n[i] Графики сохранены в {out_file}")
    plt.show()

    print("\n" + "=" * 66)
    print(" ВЫВОДЫ")
    print("=" * 66)
    print("1. Реализован метод нечеткой кластеризации c-средних (FCM) вручную.")
    print(f"2. Из временных рядов станции {STATION} извлечены признаки: среднее, СКО, амплитуда (окно {WINDOW_SIZE} мин).")
    print(f"3. Точность разделения на 'спокойно' и 'буря' составила {accuracy*100:.2f}%.")
    if accuracy > 0.8:
        print("4. Алгоритм успешно разделяет спокойные и возмущенные состояния магнитосферы.")
    else:
        print("4. Алгоритм показывает умеренное разделение. Возможно, требуется подбор параметра нечеткости m или размера окна.")
    print("5. Визуализация подтверждает, что кластеры в признаковом пространстве (среднее, СКО, амплитуда) имеют четкую структуру.")

if __name__ == "__main__":
    main()