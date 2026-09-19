import os
import sys
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import sliding_window_view

STATION    = "MOSC"
START_DAY, START_MONTH, START_YEAR = 1, 8, 2015
END_DAY,   END_MONTH,   END_YEAR   = 9, 8, 2015
LOCAL_FILE = "mosc_2015_08_01_09.txt"

TAU = 180
K   = 3

def build_url() -> str:
    return (
        "https://www.nmdb.eu/nest/draw_graph.php"
        "?formchk=1"
        f"&stations[]={STATION}"
        "&tabchoice=ori"
        "&dtype=corr_for_efficiency"
        "&tresolution=1"
        "&yunits=0"
        "&date_choice=bydate"
        f"&start_day={START_DAY:02d}&start_month={START_MONTH:02d}&start_year={START_YEAR}"
        "&start_hour=00&start_min=00"
        f"&end_day={END_DAY:02d}&end_month={END_MONTH:02d}&end_year={END_YEAR}"
        "&end_hour=23&end_min=59"
        "&output=ascii&force=1"
    )

def download_data() -> str:
    if os.path.exists(LOCAL_FILE):
        print(f"[i] Используется локальный файл: {LOCAL_FILE}")
        with open(LOCAL_FILE, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()
    print("[i] Скачивание данных с NMDB...")
    r = requests.get(build_url(), headers={"User-Agent": "Mozilla/5.0"}, timeout=120)
    r.raise_for_status()
    with open(LOCAL_FILE, "w", encoding="utf-8") as f:
        f.write(r.text)
    print(f"[i] Данные сохранены в {LOCAL_FILE}")
    return r.text

def parse_data(text: str) -> pd.DataFrame:
    rec = []
    sentinels = {-999.0, -9999.0, -99999.0, 9999.0, 99999.0}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "start_date_time" in line:
            continue
        if ";" not in line:
            continue
        dt_str, val_str = line.split(";", 1)
        val_str = val_str.strip()
        if val_str == "" or val_str.lower() in ("null", "nan", "na", "-"):
            v = np.nan
        else:
            try:
                v = float(val_str)
            except ValueError:
                continue
            if v in sentinels:
                v = np.nan
        try:
            dt = pd.to_datetime(dt_str.strip())
        except Exception:
            continue
        rec.append((dt, v))
    if not rec:
        sys.exit("Не удалось распарсить данные.")
    df = pd.DataFrame(rec, columns=["datetime", "value"]).set_index("datetime").sort_index()
    df = df[~df.index.duplicated(keep="first")]
    return df

def fill_gaps(df: pd.DataFrame):
    full_idx = pd.date_range(
        df.index.min().floor("min"),
        df.index.max().ceil("min"),
        freq="1min",
    )
    s = df["value"].reindex(full_idx)
    n_gaps = int(s.isna().sum())
    med = float(s.median())
    s = s.fillna(med)
    return s, med, n_gaps

def dist_euclidean(H, v):  return np.sqrt(np.sum((H - v) ** 2, axis=1))
def dist_manhattan(H, v):  return np.sum(np.abs(H - v), axis=1)
def dist_chebyshev(H, v):  return np.max(np.abs(H - v), axis=1)

METRICS = {
    "Евклидова":    dist_euclidean,
    "Манхэттенская": dist_manhattan,
    "Чебышёва":     dist_chebyshev,
}

def build_histories(values: np.ndarray, tau: int) -> np.ndarray:
    """Матрица (N-tau+1, tau): строка i = values[i:i+tau]."""
    return sliding_window_view(values, tau)

def predict_last(values: np.ndarray, tau: int, k: int, metric_fn):
    N = len(values)
    H = build_histories(values, tau)
    cutoff = N - 2 * tau
    pool = H[:cutoff]
    test_vec = values[-tau:]
    dists = metric_fn(pool, test_vec)
    order = np.argsort(dists)[:k]
    next_vals = values[order + tau]
    pred = float(np.mean(next_vals))
    return pred, order, dists, test_vec

def walk_forward(values: np.ndarray, tau: int, k: int, metric_fn,
                 n_test: int = 1500, step: int = 15):
    N = len(values)
    H = build_histories(values, tau)
    idx_list, pred_list, true_list = [], [], []
    t_start = N - n_test
    for t in range(t_start, N, step):
        cutoff = t - 2 * tau
        if cutoff < 10:
            continue
        pool = H[:cutoff]
        test_vec = values[t - tau:t]
        dists = metric_fn(pool, test_vec)
        order = np.argsort(dists)[:k]
        next_vals = values[order + tau]
        pred_list.append(float(np.mean(next_vals)))
        true_list.append(float(values[t]))
        idx_list.append(t)
    return np.array(idx_list), np.array(pred_list), np.array(true_list)

def main():
    print("=" * 66)
    print(" Вариант 7: MOSC, 01–09 августа 2015, k-NN прогноз")
    print("=" * 66)

    text = download_data()
    df = parse_data(text)
    print(f"[i] Загружено записей: {len(df)}")
    print(f"[i] Диапазон: {df.index.min()} — {df.index.max()}")

    series, med, n_gaps = fill_gaps(df)
    print(f"[i] Регулярная 1-мин сетка: {len(series)} точек")
    print(f"[i] Пропусков заполнено медианой ({med:.3f}): {n_gaps}")

    values = series.values.astype(float)
    print(f"[i] τ = {TAU}, K = {K}")

    print("\nПрогноз следующего значения для тестового вектора")
    results = {}
    for name, fn in METRICS.items():
        pred, order, dists, test_vec = predict_last(values, TAU, K, fn)
        results[name] = {"pred": pred, "order": order, "dists": dists, "test_vec": test_vec}
        d_str = ", ".join(f"{dists[i]:.2f}" for i in order)
        print(f"  {name:15s}: прогноз = {pred:8.3f}  |  "
              f"индексы соседей = {list(order)}  |  d = [{d_str}]")

    print("\nWalk-forward оценка на последних 1500 мин")
    wf = {}
    for name, fn in METRICS.items():
        idx, pred, true = walk_forward(values, TAU, K, fn, n_test=1500, step=15)
        mae = float(np.mean(np.abs(pred - true)))
        rmse = float(np.sqrt(np.mean((pred - true) ** 2)))
        wf[name] = {"idx": idx, "pred": pred, "true": true, "mae": mae, "rmse": rmse}
        print(f"  {name:15s}: MAE = {mae:.4f}   RMSE = {rmse:.4f}   "
              f"точек = {len(idx)}")

    print("\nВлияние числа соседей K (евклидова метрика)")
    for kk in (2, 3, 4, 5):
        idx, pred, true = walk_forward(values, TAU, kk, dist_euclidean,
                                        n_test=1500, step=15)
        mae = float(np.mean(np.abs(pred - true)))
        print(f"  K = {kk}: MAE = {mae:.4f}")

    fig = plt.figure(figsize=(15, 11))

    ax1 = plt.subplot2grid((3, 2), (0, 0), colspan=2)
    ax1.plot(series.index, values, color="steelblue", lw=0.5, label="Ряд MOSC")
    ax1.plot(series.index[-TAU:], values[-TAU:], color="red", lw=1.8,
             label=f"Тестовый вектор (τ = {TAU} мин)")
    ax1.set_title("Временной ряд MOSC, 1–9 августа 2015")
    ax1.set_ylabel("Скорость счёта, MCORR_E")
    ax1.grid(True, alpha=0.3)
    ax1.legend(loc="upper right", fontsize=8)

    ax2 = plt.subplot2grid((3, 2), (1, 0), colspan=2)
    ref = results["Евклидова"]
    for rank, i in enumerate(ref["order"]):
        ax2.plot(np.arange(TAU), values[i:i + TAU], lw=1.0,
                 label=f"Сосед #{rank+1} (i={i}, d={ref['dists'][i]:.2f})")
    ax2.plot(np.arange(TAU), ref["test_vec"], color="red", lw=2.0,
             label="Тестовый вектор")
    ax2.set_title("Тестовый вектор и k ближайших соседей (евклидова метрика)")
    ax2.set_xlabel(f"Смещение внутри вектора (τ = {TAU} мин)")
    ax2.set_ylabel("MCORR_E")
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3 = plt.subplot2grid((3, 2), (2, 0), colspan=2)
    wfe = wf["Евклидова"]
    times = series.index[wfe["idx"]]
    ax3.plot(times, wfe["true"], color="black", lw=1.2, label="Истинные значения")
    for name, marker in zip(METRICS.keys(), ["o", "s", "^"]):
        w = wf[name]
        ax3.plot(series.index[w["idx"]], w["pred"],
                 marker=marker, ms=3.5, lw=0.8, alpha=0.85,
                 label=f"{name}: MAE = {w['mae']:.3f}")
    ax3.set_title("Walk-forward прогноз (последние 1500 мин, K = 3)")
    ax3.set_xlabel("Время (UTC)")
    ax3.set_ylabel("MCORR_E")
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8, loc="upper right")

    plt.tight_layout()
    out = "mosc_knn.png"
    plt.savefig(out, dpi=150)
    print(f"\n[i] Графики сохранены в {out}")
    plt.show()

if __name__ == "__main__":
    main()
