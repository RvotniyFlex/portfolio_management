import pandas as pd
import statsmodels.api as sm
import numpy as np

def beta_calculation(df_stocks, df_index, window=252, steps=252, expand = False, alpha = 0, ewm=False,  adjust_beta=False):

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

    for step in range(0, len(df_stocks), steps):
        start = 0 if expand else step
        end = step + window
        window_label = f"{start}-{end}"
        
        market_returns = df_index['IMOEX'][start:end]
        valid_dates = market_returns.dropna().index

        # Расчет весов для экспоненциального сглаживания
        if expand:
            weights = (1 - alpha) ** np.arange(len(valid_dates))[::-1]
            weights /= weights.sum()  # Нормализация
        else:
            weights = np.ones(len(valid_dates))  # Равные веса
            
        # Расчет рыночной дисперсии с весами
        market_var = np.average((market_returns - market_returns.mean())**2, weights=weights)
        var_r = pd.concat([var_r, pd.DataFrame({'window': [window_label], 'var': [market_var]})], ignore_index=True)
        
        for name in df_stocks.columns:
            stock_returns = df_stocks[name].iloc[start:end].loc[valid_dates]
            
            X = sm.add_constant(market_returns.loc[valid_dates])  # Рыночная доходность
            y = stock_returns  # Доходность акции
            
            # Взвешенная регрессия
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
