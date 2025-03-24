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