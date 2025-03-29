import pandas as pd
import statsmodels.api as sm
import numpy as np

from portfolio_analytics.optimizer import Optimizer
from portfolio_analytics.utils import plot_efficient_frontier_curve

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def beta_calculation(df_stocks, idx, df_index, window=252, steps=252, expand = False, alpha = 0, ewm=False,  adjust_beta=False):

    """Функция для подсчёта бета коэффициентов доходности акций
    Arg:
        df_stocks (DataFrame): таблица доходностей акций
        df_index (DataFrame): таблица рыночной доходности
        window (int): ширина окна
        steps (int): шаг
        expand (bool): флаг на тип окна скользящее/расширяющееся
        alpha (float): параметр сглаживания
        ewm (bool): флаг на наличие экспоненциального сглаживания
        adjust_beta (bool): флаг на тип беты True = historical beta, False = adjusted beta
    
    """
    df_beta = pd.DataFrame(columns=['name', 'window', 'beta',  'pvalue', 'n_obs'])
    var_r = pd.DataFrame(columns=['window', 'var'])

    start_idx = min(idx)
    end_idx = max(idx)

    for step in range(start_idx, end_idx, steps):
        start = start_idx if expand else step
        end = step + window 
        window_label = f"{start}-{end}"
        market_returns = df_index['IMOEX'][start:end]
        valid_dates = market_returns.dropna().index

        # Расчет весов для экспоненциального сглаживания
        if ewm:
            weights = alpha * (1 - alpha) ** np.arange(len(valid_dates)-1, -1, -1)
            weights /= weights.sum()  # Нормализация
        else:
            weights = np.ones(len(valid_dates))  # Равные веса
            
        # Расчет рыночной дисперсии с весами
        market_var = np.average((market_returns - market_returns.mean())**2, weights=weights)
        var_r = pd.concat([var_r, pd.DataFrame({'window': [window_label], 'var': [market_var]})], ignore_index=True)
        
        for name in df_stocks.columns:
            stock_returns = df_stocks[name].loc[start:end].loc[valid_dates]
            
            X = sm.add_constant(market_returns.loc[valid_dates])  # Рыночная доходность
            y = stock_returns  # Доходность акции

            model = sm.WLS(y, X, weights=weights).fit()

            beta = model.params[1]
            resid_var = model.resid.var() 
            
            beta = 0.67 * beta + 0.33 if adjust_beta else beta
            
            df_beta = pd.concat([df_beta, pd.DataFrame({
                'name': [name],
                'window': [window_label],
                'beta': [beta],
                'pvalue': [model.pvalues[1]],
                'n_obs': [len(y)],
                'resid_var': [resid_var] 
            })], ignore_index=True)
            
    return df_beta, var_r

def cov_with_beta(df_beta, var_r):

    """"
    Функция для расчёта ковариационной матрицы из бет
    Arg:
        df_beta (DataFrame): таблица со значениями бет
        var_r (DataFrame): ковариационная матрица для рыночного индекса
    """
    betas_df = df_beta.pivot(index='window', columns='name', values='beta').reindex(pd.unique(df_beta['window']))
    resid_var_df = df_beta.pivot(index='window', columns='name', values='resid_var')  # Остаточная дисперсия
    var_df = var_r.set_index('window')['var']
    
    cov_matrices = {}
    for window in betas_df.index:
        betas = betas_df.loc[window].values
        var_market = var_df[window]
        
        # Рыночная часть ковариации
        cov_matrix = np.outer(betas, betas) * var_market
        
        # Добавление специфического риска на диагональ
        np.fill_diagonal(cov_matrix, cov_matrix.diagonal() + resid_var_df.loc[window].values)
        
        cov_matrices[window] = pd.DataFrame(
            cov_matrix, 
            index=betas_df.columns, 
            columns=betas_df.columns
        )
    return cov_matrices

def plot_with_pagination(cov_matrices, dates, df_stocks, n_graph = 10, n_cols = 2, title=None,  alpha: float = None):
    # Сортируем окна и сразу сохраняем результаты оптимизации
    results = []
    
    # Первый прогон: собираем данные и результаты оптимизации
    for window, cov_matrix in cov_matrices.items():
        steps = list(map(int, window.split('-')))
        start_idx, end_idx = steps[0], steps[1] if steps[1] < df_stocks.index[-1] else df_stocks.index[-1]

        df_current = df_stocks.loc[start_idx:end_idx]
        if alpha is not None:
            n_obs = len(df_current)
            weights = alpha * (1 - alpha) ** np.arange(n_obs-1, -1, -1)
            weights /= weights.sum()
        else:
            weights = None
                    
        # Общий расчет взвешенных средних для всех методов
        if weights is not None:
            mean_returns = np.average(df_current.values, axis=0, weights=weights)
        else:
            mean_returns = df_current.mean().values
        
        optim = Optimizer(df_current, mean_return=mean_returns, cov_return=cov_matrix.values)
        res = optim.efficient_frontier_curve()
        
        results.append({
            'window': window,
            'start_date': dates[start_idx],
            'end_date': dates[end_idx],
            'returns': [x[0] for x in res],
            'std': [x[1] for x in res]
        })
    
    # Рассчитываем глобальные границы
    all_returns = np.concatenate([r['returns'] for r in results])
    all_std = np.concatenate([r['std'] for r in results])
    
    global_returns_min = np.min(all_returns)
    global_returns_max = np.max(all_returns)
    global_std_min = np.min(all_std)
    global_std_max = np.max(all_std)
    
    # Настройка визуализации
    n_plots = len(results)

    per_page = n_cols * n_graph
    n_rows =  (len(results) + per_page - 1) // per_page
    fig_height = 6 * n_rows
    fig = plt.figure(figsize=(15, fig_height), constrained_layout=True)
    fig.suptitle(title, 
                fontsize=13, 
                y=1.01)
    
    gs = GridSpec(n_rows, n_cols, figure=fig)
    
    current_ax = None
    plot_counter = 0
    
    # Второй прогон: только отрисовка из сохраненных данных
    for idx, result in enumerate(results):
        if idx % n_graph == 0:
            row = (idx // n_graph) // n_cols
            col = (idx // n_graph) % n_cols
            current_ax = fig.add_subplot(gs[row, col])
            
            # Установка глобальных границ
            current_ax.set_xlim(global_std_min, global_std_max)
            current_ax.set_ylim(global_returns_min, global_returns_max)
            
            current_ax.set_ylabel("Expected return, %")
            current_ax.set_xlabel("Standart deviation, %")
            #current_ax.set_title(f'Group {idx//10 + 1}')
            plot_counter = 0
        
        # Отрисовка из сохраненных данных
        plot_efficient_frontier_curve(
            result['std'],
            result['returns'],
            label=f"{result['start_date']} - {result['end_date']}",
            ax=current_ax
        )
        
        plot_counter += 1
        
        if plot_counter == n_graph or idx == len(results)-1:
            current_ax.legend()

    # Скрытие пустых subplots
    total_plots = n_rows * n_cols * n_graph
    for j in range(len(results), total_plots):
        row = (j // n_graph) // n_cols
        col = (j // n_graph) % n_cols
        if row < n_rows and col < n_cols:
            fig.add_subplot(gs[row, col]).axis('off')

    plt.tight_layout()
    plt.show()

def compare_frontier_methods(methods_data, dates, df_stocks, n_cols=2, title=None,
                            alpha: float = None):
    """
    Сравнение методов построения эффективной границы для одних и тех же окон
    
    Parameters:
        methods_data (dict): {method_name: cov_matrices}
        dates (pd.Index): Индекс с датами
        df_stocks (pd.DataFrame): Исходные данные доходностей
        n_cols (int): Количество графиков в строке
        title (str): Общий заголовок
        alpha (float): Коэффициент экспоненциального забывания
    """
    # Сбор данных
    results = {}
    windows = list(next(iter(methods_data.values())).keys())
    
    # Прогон оптимизации для всех методов и окон
    for method in methods_data:
        results[method] = {}
        cov_matrices = methods_data[method]
        
        for window in windows:
            steps = list(map(int, window.split('-')))
            start_idx = steps[0]
            end_idx = steps[1] if steps[1] < len(df_stocks) else len(df_stocks)-1
            df_current = df_stocks.iloc[start_idx:end_idx+1]
            
            # Расчет весов для экспоненциального забывания
            if alpha is not None:
                n_obs = len(df_current)
                weights = alpha * (1 - alpha) ** np.arange(n_obs-1, -1, -1)
                weights /= weights.sum()
            else:
                weights = None
                
            # Общий расчет взвешенных средних для всех методов
            if weights is not None:
                mean_returns = np.average(df_current.values, axis=0, weights=weights)
            else:
                mean_returns = df_current.mean().values
                
            # Расчет ковариации в зависимости от метода
            if method == 'returns':
                if weights is not None:
                    # Взвешенная ковариация для returns-метода
                    centered = df_current.values - mean_returns
                    cov = pd.DataFrame(
                        np.cov(centered.T, aweights=weights),
                        index=df_current.columns,
                        columns=df_current.columns
                    )
                else:
                    cov = df_current.cov()
            else:
                # Для методов с предрасчитанными ковариациями
                cov = cov_matrices[window]
            
            # Оптимизация портфеля
            optim = Optimizer(
                df_current, 
                mean_return=mean_returns, 
                cov_return=cov.values
            )
            res = optim.efficient_frontier_curve()
            
            results[method][window] = {
                'returns': [x[0] for x in res],
                'std': [x[1] for x in res],
                'start_date': dates[start_idx],
                'end_date': dates[end_idx]
            }
    # Рассчет глобальных границ
    all_returns = []
    all_std = []
    for method in results.values():
        for window_data in method.values():
            all_returns.extend(window_data['returns'])
            all_std.extend(window_data['std'])
    
    global_returns_min, global_returns_max = np.nanquantile(all_returns, [0.01, 0.99])
    global_std_min, global_std_max = np.nanquantile(all_std, [0.01, 0.99])

    # Настройка визуализации
    n_windows = len(windows)
    n_rows = (n_windows + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(6*n_cols, 6*n_rows), constrained_layout=True)
    gs = GridSpec(n_rows, n_cols, figure=fig)
    fig.suptitle(title, y=1.03, fontsize=14)
    

    # Отрисовка по окнам
    for idx, window in enumerate(windows):
        ax = fig.add_subplot(gs[idx//n_cols, idx%n_cols])
        steps = list(map(int, window.split('-')))
        start_date = dates[steps[0]]
        end_date = dates[steps[1]] if steps[1] < len(dates) else dates[-1]
        
        # Для каждого метода в текущем окне
        for midx, (method, data) in enumerate(results.items()):
            window_data = data[window]
            
            # Отрисовка кривой
            ax.scatter(
                window_data['std'],
                window_data['returns'],
                label=method,
                alpha=0.6,
                s=20
            )
        

        # Настройка графика
        ax.set_xlim(global_std_min*0.95, global_std_max*1.05)
        ax.set_ylim(global_returns_min*0.95, global_returns_max*1.05)
        ax.set_title(f"{start_date} - {end_date}", fontsize=10)
        ax.set_xlabel("Standard Deviation")
        ax.set_ylabel("Expected Return")
        ax.grid(True, alpha=0.2)
        
        if idx == 0:
            ax.legend()

    # Скрытие пустых subplots
    for j in range(len(windows), n_rows*n_cols):
        fig.add_subplot(gs[j//n_cols, j%n_cols]).axis('off')

    plt.show()