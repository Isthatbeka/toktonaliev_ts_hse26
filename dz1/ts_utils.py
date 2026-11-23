import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf


WEEKDAYS_RU = ['Пн', 'Вт', 'Ср', 'Чт', 'Пт', 'Сб', 'Вс']


# ---------------------------------------------------------------------------
# Загрузка и подготовка данных
# ---------------------------------------------------------------------------

def load_data(path):
    df = pd.read_excel(path, dtype={'InvoiceNo': str, 'StockCode': str})
    df['StockCode'] = df['StockCode'].str.strip()
    return df


def get_trading_days(df):
    days = df['InvoiceDate'].dt.normalize().drop_duplicates().sort_values()
    return pd.DatetimeIndex(days)


def drop_cancelled_orders(sales, cancels, max_days=1):
    """Удаляет заказы, которые были отменены целиком.

    Для каждой отмены ищем заказ того же клиента с тем же количеством,
    сделанный не раньше чем за max_days дней до отмены. Если нашли - считаем,
    что заказ фактически не состоялся (исправление ошибочного заказа), и удаляем его.
    Возвращает (очищенные продажи, таблица найденных пар).
    """
    sales = sales.copy()
    used = set()
    pairs = []
    for _, c in cancels.sort_values('InvoiceDate').iterrows():
        cand = sales[(sales['CustomerID'] == c['CustomerID'])
                     & (sales['Quantity'] == -c['Quantity'])
                     & (sales['InvoiceDate'] <= c['InvoiceDate'])
                     & (c['InvoiceDate'] - sales['InvoiceDate'] <= pd.Timedelta(days=max_days))]
        cand = cand[~cand.index.isin(used)]
        if len(cand) == 0:
            continue
        idx = cand['InvoiceDate'].idxmax()  # самый поздний подходящий заказ
        used.add(idx)
        pairs.append({'заказ': sales.loc[idx, 'InvoiceNo'],
                      'дата заказа': sales.loc[idx, 'InvoiceDate'],
                      'отмена': c['InvoiceNo'],
                      'дата отмены': c['InvoiceDate'],
                      'клиент': c['CustomerID'],
                      'количество': -c['Quantity']})
    sales = sales.drop(index=list(used))
    return sales, pd.DataFrame(pairs)


def prepare_product(df, stock_code, cancel_window_days=1):
    """Отбирает транзакции товара и чистит их.

    - отмены (InvoiceNo начинается с 'C') сами в продажи не идут;
    - если отмена совпадает с конкретным заказом (см. drop_cancelled_orders) - удаляем и заказ;
    - строки с ценой 0 убираем (это не продажи, а служебные корректировки без клиента);
    - строки без CustomerID оставляем: продажа была, просто покупатель не зарегистрирован.
    Возвращает (продажи, словарь с отчётом, таблица отменённых заказов).
    """
    p = df[df['StockCode'] == stock_code].copy()
    is_cancel = p['InvoiceNo'].str.startswith('C')
    cancels = p[is_cancel]
    sales = p[~is_cancel]

    report = {
        'строк по товару': len(p),
        'строк-отмен (C...)': len(cancels),
        'штук в отменах': int(-cancels['Quantity'].sum()),
        'строк с ценой <= 0': int((sales['UnitPrice'] <= 0).sum()),
        'строк с Quantity <= 0 (не отмены)': int((sales['Quantity'] <= 0).sum()),
        'строк без CustomerID': int(sales['CustomerID'].isna().sum()),
    }
    sales = sales[(sales['UnitPrice'] > 0) & (sales['Quantity'] > 0)]
    sales, pairs = drop_cancelled_orders(sales, cancels, max_days=cancel_window_days)
    report['удалено отменённых заказов'] = len(pairs)
    report['штук в удалённых заказах'] = int(pairs['количество'].sum()) if len(pairs) else 0
    report['строк продаж после очистки'] = len(sales)
    report['штук продано после очистки'] = int(sales['Quantity'].sum())
    return sales, report, pairs


# ---------------------------------------------------------------------------
# Формирование рядов
# ---------------------------------------------------------------------------

def make_daily(sales, trading_days):
    """Дневной ряд по календарным дням.

    Дни, когда магазин работал, а товар не покупали -> 0 (это реальные нулевые продажи).
    Дни, когда магазин не работал вообще -> NaN (наблюдения нет, это не ноль спроса).
    """
    qty = sales.groupby(sales['InvoiceDate'].dt.normalize())['Quantity'].sum()
    idx = pd.date_range(trading_days.min(), trading_days.max(), freq='D')
    s = qty.reindex(idx).astype(float)
    open_no_sales = idx.isin(trading_days) & s.isna().values
    s[open_no_sales] = 0.0
    s.name = 'qty'
    s.index.name = 'date'
    return s


def aggregate(daily, freq):
    """Агрегирует дневной ряд по неделям ('W', пн-вс) или месяцам ('M').

    qty            - сумма продаж (NaN, если магазин не работал ни одного дня периода)
    trading_days   - сколько рабочих дней было в периоде
    complete       - период целиком попадает в интервал наблюдений
    Индекс - дата начала периода.
    """
    per = daily.index.to_period(freq)
    g = daily.groupby(per)
    out = pd.DataFrame({'qty': g.sum(min_count=1),
                        'trading_days': g.count()})
    start, end = daily.index.min(), daily.index.max()
    out['complete'] = [(p.start_time >= start) and (p.end_time.normalize() <= end)
                       for p in out.index]
    out.index = out.index.to_timestamp()
    out.index.name = 'period_start'
    return out


def to_six_day_grid(daily):
    """Убирает субботы: магазин по субботам не работает никогда,
    поэтому рабочая неделя = 6 дней (пн-пт + вс) и это регулярная сетка ряда."""
    return daily[daily.index.dayofweek != 5]


def fill_gaps(s, method='linear'):
    """Заполняет NaN (дни/недели, когда магазин был закрыт) интерполяцией.
    Нужно только для методов, которые не работают с пропусками (STL, ADF, KPSS)."""
    return s.interpolate(method=method, limit_direction='both')


# ---------------------------------------------------------------------------
# Пропуски и выбросы
# ---------------------------------------------------------------------------

def missing_report(s, holidays=None):
    """Таблица пропусков (NaN) с днём недели и причиной, если она известна."""
    holidays = holidays or {}
    na = s[s.isna()].index
    rep = pd.DataFrame({'дата': na.date,
                        'день недели': [WEEKDAYS_RU[d.dayofweek] for d in na]})
    reasons = []
    for d in na:
        key = d.strftime('%Y-%m-%d')
        if key in holidays:
            reasons.append(holidays[key])
        elif d.dayofweek == 5:
            reasons.append('суббота - магазин не работает')
        else:
            reasons.append('неизвестно')
    rep['причина'] = reasons
    return rep


def hampel(s, window=13, n_sigmas=3.0):
    """Фильтр Хампеля: выброс, если значение отличается от скользящей медианы
    больше чем на n_sigmas * 1.4826 * MAD. Устойчив к самим выбросам и учитывает
    локальный уровень ряда (в отличие от глобального правила IQR)."""
    x = s.astype(float)
    med = x.rolling(window, center=True, min_periods=1).median()
    mad = (x - med).abs().rolling(window, center=True, min_periods=1).median()
    thr = n_sigmas * 1.4826 * mad
    res = pd.DataFrame({'value': x, 'median': med,
                        'lower': med - thr, 'upper': med + thr})
    res['is_outlier'] = (x - med).abs() > thr
    return res


def iqr_outliers(s, k=1.5):
    """Глобальное правило IQR (для сравнения с Хампелем)."""
    q1, q3 = s.quantile([0.25, 0.75])
    lo, hi = q1 - k * (q3 - q1), q3 + k * (q3 - q1)
    return (s < lo) | (s > hi), lo, hi


def plot_outliers(h, title, ylabel='Продажи, шт.'):
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(h.index, h['value'], lw=0.8, label='ряд')
    ax.plot(h.index, h['median'], lw=1, color='tab:green', label='скользящая медиана')
    ax.fill_between(h.index, h['lower'].clip(lower=0), h['upper'], color='tab:green',
                    alpha=0.15, label='допустимый коридор')
    out = h[h['is_outlier']]
    ax.scatter(out.index, out['value'], color='red', zorder=3, s=20, label='выброс')
    ax.set_title(title)
    ax.set_xlabel('Дата')
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()


# ---------------------------------------------------------------------------
# Описательный анализ
# ---------------------------------------------------------------------------

def describe_series(s):
    """Основные статистики + характеристики, полезные для продаж:
    доля нулей, CV, ADI и CV^2 (классификация спроса Syntetos-Boylan)."""
    x = s.dropna()
    nz = x[x > 0]
    adi = len(x) / len(nz) if len(nz) else np.nan
    cv2 = (nz.std() / nz.mean()) ** 2 if len(nz) > 1 else np.nan
    if adi < 1.32:
        demand = 'smooth (ровный)' if cv2 < 0.49 else 'erratic (неравномерный)'
    else:
        demand = 'intermittent (прерывистый)' if cv2 < 0.49 else 'lumpy (комковатый)'
    return pd.Series({
        'наблюдений': len(x),
        'среднее': x.mean(),
        'медиана': x.median(),
        'ст. отклонение': x.std(),
        'минимум': x.min(),
        'Q1': x.quantile(0.25),
        'Q3': x.quantile(0.75),
        'максимум': x.max(),
        'CV (std/mean)': x.std() / x.mean(),
        'асимметрия': stats.skew(x),
        'эксцесс': stats.kurtosis(x),
        'доля нулей': (x == 0).mean(),
        'ADI': adi,
        'CV^2 ненулевых': cv2,
        'тип спроса': demand,
    })


def plot_series(s, title, ylabel='Продажи, шт.', xlabel='Дата', ax=None, **kw):
    if ax is None:
        _, ax = plt.subplots(figsize=(14, 4))
    ax.plot(s.index, s.values, **kw)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if kw.get('label'):
        ax.legend()
    return ax


def plot_rolling(s, window, title, ylabel='Продажи, шт.'):
    """Скользящие среднее и стандартное отклонение - как меняются уровень и разброс."""
    rm = s.rolling(window, center=True).mean()
    rs = s.rolling(window, center=True).std()
    fig, ax = plt.subplots(figsize=(14, 4))
    ax.plot(s.index, s, lw=0.6, alpha=0.5, label='ряд')
    ax.plot(rm.index, rm, lw=2, label=f'скользящее среднее ({window})')
    ax.plot(rs.index, rs, lw=2, label=f'скользящее ст. откл. ({window})')
    ax.set_title(title)
    ax.set_xlabel('Дата')
    ax.set_ylabel(ylabel)
    ax.legend()
    plt.show()
    return rm, rs


def level_spread_by_month(daily):
    """Среднее и разброс дневных продаж по месяцам + тест Левене на равенство дисперсий."""
    x = daily.dropna()
    grp = x.groupby(x.index.to_period('M'))
    tab = pd.DataFrame({'дней': grp.size(), 'среднее': grp.mean(),
                        'медиана': grp.median(), 'ст. откл.': grp.std()})
    tab['CV'] = tab['ст. откл.'] / tab['среднее']
    lev = stats.levene(*[g.values for _, g in grp], center='median')
    return tab, lev


def weekday_profile(daily):
    """Продажи по дням недели + тест Краскела-Уоллиса
    (H0: распределения продаж во все дни недели одинаковые)."""
    x = daily.dropna()
    wd = x.index.dayofweek
    tab = x.groupby(wd).agg(['count', 'mean', 'median', 'std'])
    tab.index = [WEEKDAYS_RU[i] for i in tab.index]
    kw = stats.kruskal(*[x[wd == d].values for d in sorted(set(wd))])
    return tab, kw


def plot_weekday_box(daily, title, ylabel='Продажи, шт.'):
    x = daily.dropna()
    days = sorted(set(x.index.dayofweek))
    data = [x[x.index.dayofweek == d].values for d in days]
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.boxplot(data, showfliers=True)
    ax.set_xticks(range(1, len(days) + 1))
    ax.set_xticklabels([WEEKDAYS_RU[d] for d in days])
    ax.set_title(title)
    ax.set_xlabel('День недели')
    ax.set_ylabel(ylabel)
    plt.show()


def plot_acf_pacf(s, lags, title):
    fig, axes = plt.subplots(1, 2, figsize=(14, 3.5))
    plot_acf(s.dropna(), lags=lags, ax=axes[0])
    plot_pacf(s.dropna(), lags=lags, ax=axes[1], method='ywm')
    axes[0].set_title(f'ACF: {title}')
    axes[1].set_title(f'PACF: {title}')
    for a in axes:
        a.set_xlabel('Лаг')
        a.set_ylabel('Корреляция')
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Декомпозиция
# ---------------------------------------------------------------------------

def stl_decompose(s, period, seasonal=7, trend=None, robust=True):
    """STL-разложение (Loess).
    seasonal - окно сглаживания сезонности (в периодах, нечётное): чем больше, тем стабильнее профиль;
    trend - окно тренда в наблюдениях (нечётное), None - значение statsmodels по умолчанию;
    robust=True - чтобы крупные заказы уходили в остаток, а не искажали тренд и сезонность."""
    return STL(s, period=period, seasonal=seasonal, trend=trend, robust=robust).fit()


def decomposition_strength(res):
    """Сила тренда и сезонности (Wang, Smith, Hyndman 2006):
    F = max(0, 1 - Var(R) / Var(X + R)). Близко к 0 - компоненты почти нет, к 1 - сильная."""
    r, t, sz = res.resid, res.trend, res.seasonal
    ft = max(0.0, 1 - np.var(r) / np.var(t + r))
    fs = max(0.0, 1 - np.var(r) / np.var(sz + r))
    return {'сила тренда': ft, 'сила сезонности': fs}


def plot_stl(res, title, ylabel='Продажи, шт.'):
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)
    parts = [(res.observed, 'Исходный ряд'), (res.trend, 'Тренд'),
             (res.seasonal, 'Сезонная компонента'), (res.resid, 'Остаток')]
    for ax, (comp, name) in zip(axes, parts):
        if name == 'Остаток':
            ax.scatter(comp.index, comp.values, s=6)
            ax.axhline(0, color='black', lw=0.8)
        else:
            ax.plot(comp.index, comp.values, lw=1)
        ax.set_title(name)
        ax.set_ylabel(ylabel)
    axes[-1].set_xlabel('Дата')
    fig.suptitle(title, y=1.0)
    plt.tight_layout()
    plt.show()


def seasonal_profile(res, s):
    """Средний сезонный эффект по дням недели (из STL)."""
    prof = res.seasonal.groupby(s.index.dayofweek).mean()
    prof.index = [WEEKDAYS_RU[i] for i in prof.index]
    return prof


# ---------------------------------------------------------------------------
# Стационарность
# ---------------------------------------------------------------------------

def adf_test(s, regression='c'):
    """ADF: H0 - есть единичный корень (ряд нестационарен).
    p < 0.05 -> отвергаем H0 -> признак стационарности."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', FutureWarning)  # новые statsmodels ругаются на формат результата
        res = adfuller(s.dropna(), regression=regression, autolag='AIC')
    stat, p, lags, crit = res[0], res[1], res[2], res[4]
    return {'ADF stat': stat, 'ADF p-value': p, 'ADF lags': lags, 'ADF crit 5%': crit['5%']}


def kpss_test(s, regression='c'):
    """KPSS: H0 - ряд стационарен (вокруг константы при 'c' или тренда при 'ct').
    p < 0.05 -> отвергаем H0 -> признак нестационарности.
    statsmodels считает p-value только в диапазоне [0.01, 0.1], за его пределами
    возвращает границу (поэтому предупреждение глушим)."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        stat, p, lags, crit = kpss(s.dropna(), regression=regression, nlags='auto')
    return {'KPSS stat': stat, 'KPSS p-value': p, 'KPSS crit 5%': crit['5%']}


def stationarity_verdict(adf_p, kpss_p, alpha=0.05):
    adf_st = adf_p < alpha       # ADF говорит "стационарен"
    kpss_st = kpss_p >= alpha    # KPSS говорит "стационарен"
    if adf_st and kpss_st:
        return 'стационарен'
    if not adf_st and not kpss_st:
        return 'нестационарен'
    if adf_st and not kpss_st:
        return 'противоречие: ADF - стац., KPSS - нет'
    return 'противоречие: ADF - нестац., KPSS - стац.'


def stationarity_report(series, regression='c'):
    """series: dict {название: ряд}. Возвращает таблицу ADF + KPSS для каждого ряда."""
    rows = {}
    for name, s in series.items():
        r = {'n': int(s.dropna().shape[0])}
        r.update(adf_test(s, regression))
        r.update(kpss_test(s, regression))
        r['вывод'] = stationarity_verdict(r['ADF p-value'], r['KPSS p-value'])
        rows[name] = r
    return pd.DataFrame(rows).T


def detrend_linear(s):
    """Убирает линейный тренд (МНК по номеру наблюдения), возвращает остатки."""
    x = s.dropna()
    t = np.arange(len(x))
    b, a = np.polyfit(t, x.values, 1)
    return pd.Series(x.values - (a + b * t), index=x.index, name=s.name)


def compare_rolling(series, window, title, ylabel):
    """Скользящие среднее и ст. отклонение для нескольких рядов (до/после преобразования)."""
    fig, axes = plt.subplots(1, len(series), figsize=(6 * len(series), 3.8))
    for ax, (name, s) in zip(np.atleast_1d(axes), series.items()):
        ax.plot(s.index, s, lw=0.7, alpha=0.6, label='ряд')
        ax.plot(s.index, s.rolling(window, center=True).mean(), lw=2, label='скользящее среднее')
        ax.plot(s.index, s.rolling(window, center=True).std(), lw=2, label='скользящее ст. откл.')
        ax.set_title(name)
        ax.set_xlabel('Дата')
        ax.set_ylabel(ylabel)
        ax.tick_params(axis='x', rotation=30)
        ax.legend(fontsize=8)
    fig.suptitle(title)
    plt.tight_layout()
    plt.show()


def make_stationary(s, log=True, diff=1, seasonal_diff=None):
    """Преобразование ряда: log1p (стабилизация дисперсии) и разности.
    log1p, а не log, потому что в ряду есть нулевые продажи."""
    x = np.log1p(s) if log else s.copy()
    if diff:
        x = x.diff(diff)
    if seasonal_diff:
        x = x.diff(seasonal_diff)
    return x.dropna()
