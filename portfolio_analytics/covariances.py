import pandas as pd
from typing import Dict
from pypfopt.risk_models import risk_matrix

def expanding_covariance_with_step(
        df_returns: pd.DataFrame,
        step: int = 252,
        cov_method='sample_cov'
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Вычисляет ковариационные матрицы на расширяющемся окне, но с указанным шагом по индексам.

    То есть, вместо того чтобы считать ковариацию для каждой даты, 
    мы двигаемся по DataFrame через каждые `step` строк, 
    берём срез от начала (0) до текущего положения (i) и считаем ковариацию.
    Результат сохраняется в словаре, где ключ — это метка времени (Timestamp) 
    из индекса `df_returns` (на текущем шаге), а значение — ковариационная матрица (DataFrame).

    Пример:
        Если `step=252` и у нас ежедневные данные, 
        то будем получать ковариационную матрицу на конец каждого «года» (252 дня).

    Args:
        df_returns (pd.DataFrame): Таблица с доходностями, 
            где строки — это даты (индекс), а столбцы — различные активы.
        step (int): Шаг, в количестве строк (например, 252), 
            через который будет рассчитываться ковариационная матрица.

    Returns:
        Dict[pd.Timestamp, pd.DataFrame]: 
            Словарь, где:
            - ключ — метка времени (Timestamp) из индекса `df_returns` на текущем шаге,
            - значение — словарь из матрицы ковариаций и доходностей
    """
    cov_dict = {}
    # Начинаем с индекса = step, чтобы влез хотя бы один период
    for i in range(step, len(df_returns) + 1, step):
        current_date = df_returns.index[i-1]  # i-1, т.к. DataFrame идет с 0 до i-1
        # Берём все данные с начала до i
        slice_ = df_returns.iloc[:i]
        cov_matr = risk_matrix(slice_, returns_data=True, method=cov_method)
        mu = slice_.mean()
        # Считаем ковариацию
        cov_dict[current_date] = {'cov': cov_matr, 'mu': mu}
    return cov_dict


def rolling_covariance_with_step(
        df_returns: pd.DataFrame,
        step: int = 252,
        window_size: int = 252,
        cov_method='sample_cov'
    ) -> Dict[pd.Timestamp, pd.DataFrame]:
    """
    Вычисляет ковариационные матрицы на скользящем окне.

    Args:
        df_returns (pd.DataFrame): Таблица с доходностями, 
            где строки — это даты (индекс), а столбцы — различные активы.
        step (int): Шаг, в количестве строк (например, 252), 
            через который будет рассчитываться ковариационная матрица.
        window_size (int): Размер скользящего окна
        cov_method (str): Названия метода расчета ковариации из pypfopt

    Returns:
        Dict[pd.Timestamp, pd.DataFrame]: 
            Словарь, где:
            - ключ — метка времени (Timestamp) из индекса `df_returns` на текущем шаге,
            - значение — словарь из матрицы ковариаций и доходностей

    """
    cov_matrices_rolling = {}

    for current_date in df_returns.index[window_size::step]:
        window_slice = df_returns.loc[:current_date].tail(window_size)
        cov_mat = risk_matrix(window_slice, returns_data=True, method=cov_method)
        mu = window_slice.ewm(span=step).mean().iloc[-1]
        cov_matrices_rolling[current_date] = {'cov' : cov_mat, 'mu': mu}

    return cov_matrices_rolling,