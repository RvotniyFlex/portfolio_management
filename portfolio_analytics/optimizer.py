from scipy.optimize import minimize, LinearConstraint
from functools import cached_property
import numpy as np
from tqdm.notebook import tqdm
from collections import defaultdict

class Optimizer:
    """
    Минимизация отклонения доходности при заданнном уровне

    Args:
        data (np.array): n_dates x n_shares матрица доходностей
        seed (int): сид для репрезентативности
        constrs (Optional[List[Union[LinearConstraint, NonlinearConstraint]]]): лист ограничений
        bounds (Optional[Bounds]): границы параметров
        mean_return (np.array): (data.shape[1], ) матрица ожиданий для взвешиваний
        cov_return (np.array): (data.shape[1], data.shape[1]) ковариационная матрица для взвешиваний
    """

    def __init__(self, data, constrs=[], bounds=None, mean_return=None, cov_return=None, seed=42):

        if np.all(np.abs(data) != np.inf):
            self.data = data
        else:
            raise ValueError('В данных замечены значение inf')
        self.seed = seed
        self.constrs = constrs
        self.bounds = bounds

        if mean_return is None:
            self.mean_return = self.data.mean()
        elif mean_return.shape == (data.shape[1],):
            self.mean_return = mean_return
        else:
            raise ValueError(f'Требуется вектор ожиданий размером: {(data.shape[1], )}\nПолучено: {mean_return.shape}')

        if cov_return is None:
            self.cov_return = self.data.cov()
        elif cov_return.shape == (data.shape[1], data.shape[1]):
            self.cov_return = cov_return
        else:
            raise ValueError(f'Требуется коварициоонная матрица размером: {(data.shape[1], data.shape[1])}\nПолучено: {cov_return.shape}')


    def generate_init_w(self):
        sample_w = np.random.uniform(-1, 1, self.data.shape[1]) 
        init_w = sample_w / np.sum(sample_w)
        return init_w
    
    @cached_property
    def return_rate_border(self):
        mean_return = self.mean_return
        min_border = min(mean_return)
        max_border = max(mean_return)
        return min_border, max_border

    @staticmethod
    def portfolio_var(w, cov):
        w_matrix = w.reshape(-1, 1) @ w.reshape(1, -1)
        total_var = (w_matrix * cov).sum().sum()
        return total_var

    def minimize_portfolio_var(self, r_iter):
        init_w = self.generate_init_w()
        w_sum_constr = LinearConstraint(np.ones(init_w.shape), 1, 1)
        expect_return_constr = LinearConstraint(self.mean_return, r_iter, r_iter)
        constrs = [w_sum_constr, expect_return_constr] + self.constrs

        if self.bounds is not None:
            res = minimize(
                self.portfolio_var, 
                x0=init_w, 
                args=(self.cov_return), 
                method='SLSQP', 
                tol=1e-6, 
                constraints=constrs, 
                bounds=self.bounds)
            
        else:
            res = minimize(
                self.portfolio_var, 
                x0=init_w, 
                args=(self.cov_return), 
                method='SLSQP', 
                tol=1e-6, 
                constraints=constrs)
        return res
    
    def efficient_frontier_curve(self, n_point=500, presicion=4):
        """
        Кривая эффективных портфелей.
        
        Args: 
            n_point (int): требуемое кол-во точек
            presicion (int): округление доходности для аггрегации и выбора минимальной дисперсии 

        Returns:
            results (List[(return, std, w)]): список решений
        """
        np.random.seed(self.seed)
        min_rate, max_rate = self.return_rate_border
        step = (max_rate - min_rate) / n_point
        current_rate = min_rate
        results = []
        for i in tqdm(range(n_point)):
            res = self.minimize_portfolio_var(current_rate)
            if res.success:
                results.append((current_rate, res.fun ** (1/2), res.x))
            current_rate += step

        groups = defaultdict(list)
        for r, risk, weights in results:
            rounded_r = round(r, presicion)
            groups[rounded_r].append((r, risk, weights))

        filtered_data = []
        for rounded_r, items in groups.items():
            min_risk = min(risk for _, risk, _ in items)
            filtered_data.extend((r, risk, weights) for r, risk, weights in items if risk == min_risk)

        return filtered_data
    

class OptimizerIS:
    """
    Оптимизатор, который учитывает Implementation Shortfall (IS).
    Наследует логику из Optimizer, но добавляет штраф за ребалансировку.
    """

    def __init__(
        self, 
        data, 
        w_old=None,       # старый портфель
        is_costs=None,    # вектор IS
        alpha=0.01,       # коэффициент штрафа за IS
        constrs=[], 
        bounds=None, 
        mean_return=None, 
        cov_return=None, 
        seed=42
    ):
        """
        Args:
            data (np.array or pd.DataFrame): n_dates x n_shares матрица доходностей 
            w_old (np.array): предыдущие веса (shape = (n_shares, ))
            is_costs (np.array): оценка издержек на единицу изменения веса (shape = (n_shares, ))
            alpha (float): коэффициент, насколько сильно штрафуется IS
            ...
        """
        # Сохраняем все аргументы
        self.data = data
        self.seed = seed
        self.constrs = constrs
        self.bounds = bounds
        self.alpha = alpha

        # Если нет старого портфеля, считаем, что он нулевой:
        if w_old is None:
            w_old = np.zeros(data.shape[1])
        self.w_old = w_old

        # Если нет is_costs, считаем 0 (никакого штрафа)
        if is_costs is None:
            is_costs = np.zeros(data.shape[1])
        self.is_costs = is_costs

        # mean_return, cov_return
        if mean_return is None:
            # Если data — DataFrame, mean() вернёт Series
            # Если np.array, придётся самим считать среднее
            if hasattr(data, 'mean'):
                self.mean_return = data.mean()
            else:
                self.mean_return = np.mean(data, axis=0)
        else:
            self.mean_return = mean_return

        if cov_return is None:
            if hasattr(data, 'cov'):
                self.cov_return = data.cov()
            else:
                self.cov_return = np.cov(data, rowvar=False)
        else:
            self.cov_return = cov_return

    def generate_init_w(self):
        np.random.seed(self.seed)
        n = self.data.shape[1]
        sample_w = np.random.uniform(-1,1,n)
        init_w = sample_w / np.sum(sample_w)
        return init_w
    
    @staticmethod
    def portfolio_var(w, cov):
        """ w^T Sigma w """
        return w.T @ cov @ w
    
    def is_penalty(self, w):
        """
        Implementation Shortfall: суммируем |w_i - w_old_i| * is_costs_i
        """
        return np.sum(np.abs(w - self.w_old) * self.is_costs)

    def portfolio_var_with_is(self, w, cov):
        """
        Итоговая функция: var + alpha * IS
        """
        var_part = self.portfolio_var(w, cov)
        is_part = self.is_penalty(w)
        return var_part + self.alpha * is_part

    def minimize_portfolio_var_is(self, target_return):
        """
        Минимизируем [ var + alpha*IS ] при условии, что E(R) = target_return, sum(w)=1.
        """
        init_w = self.generate_init_w()
        # Ограничение: сумма весов = 1
        w_sum_constr = LinearConstraint(np.ones(len(init_w)), 1, 1)
        # Ограничение: матожидание доходности = target_return
        expect_return_constr = LinearConstraint(self.mean_return, target_return, target_return)
        all_constrs = [w_sum_constr, expect_return_constr] + self.constrs

        res = minimize(
            fun=self.portfolio_var_with_is,
            x0=init_w,
            args=(self.cov_return,),
            method='SLSQP',
            constraints=all_constrs,
            bounds=self.bounds,
            tol=1e-6
        )
        return res

    def efficient_frontier_curve_with_is(self, n_points=100):
        """
        Строим эффективную границу, перебирая различные уровни доходности
        и минимизируя [ var + alpha * IS ].
        
        Returns:
            List of tuples: [(R, sigma, w), ...]
        """
        np.random.seed(self.seed)
        # Оценим min & max доходность из mean_return
        min_r = np.min(self.mean_return)
        max_r = np.max(self.mean_return)
        step = (max_r - min_r) / n_points

        frontier_results = []
        current_r = min_r
        for _ in tqdm(range(n_points)):
            res = self.minimize_portfolio_var_is(current_r)
            if res.success:
                # res.fun = var + alpha*IS (значение целевой)
                # Но для графика "sigma" = sqrt(var) обычно
                # var = w^T Sigma w => выделим var отдельно:
                var_val = self.portfolio_var(res.x, self.cov_return)
                std_val = np.sqrt(var_val)
                frontier_results.append((current_r, std_val, res.x))
            current_r += step
        
        return frontier_results
