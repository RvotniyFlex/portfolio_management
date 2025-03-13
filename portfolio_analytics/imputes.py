from abc import ABC, abstractmethod
import pandas as pd

class BaseTimeSeriesImputer(ABC):
    """
    Абстрактный класс для методов заполнения пропусков во временных рядах.
    Определяет общий интерфейс.
    """
    
    @abstractmethod
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Принимает DataFrame, возвращает DataFrame, 
        в котором пропуски заполнены в соответствии с выбранным методом.
        """
        pass


class ForwardFillImputer(BaseTimeSeriesImputer):
    """
    Заполняет пропуски, копируя последнее известное значение вперёд (pandas ffill).
    """
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.ffill()


class BackwardFillImputer(BaseTimeSeriesImputer):
    """
    Заполняет пропуски, копируя следующее известное значение назад (pandas bfill).
    """
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.bfill()


class ZeroFillImputer(BaseTimeSeriesImputer):
    """
    Заполняет пропуски нулями.
    """
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        return data.fillna(0)


class MeanImputer(BaseTimeSeriesImputer):
    """
    Заполняет пропуски средним значением по каждому столбцу.
    """
    
    def fit_transform(self, data: pd.DataFrame) -> pd.DataFrame:
        # mean() по каждому столбцу => fillna() для каждого столбца индивидуально
        return data.fillna(data.mean())
