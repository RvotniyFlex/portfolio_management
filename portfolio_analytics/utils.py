import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def plot_efficient_frontier_curve(expect, std, label, ax):
    """
    Функция для сравнения кривых

    Args:
        expect (List[float]): список ожиданий
        std (List[float]): список отклонений
        label (str): название для легенды
        ax (): ось графиуов
    """

    expect = np.array(expect)
    std = np.array(std)
    expect_border = np.quantile(expect, 0.01), np.quantile(expect, 0.99)
    std_border = np.quantile(std, 0.01), np.quantile(std, 0.99)

    mask = (expect >= expect_border[0]) & (expect <= expect_border[1]) & (std >= std_border[0]) & (std <= std_border[1])

    sns.scatterplot(
        x=expect[mask],
        y=std[mask],
        label=label,
        ax=ax
    )

    ax.set_ylabel("Expected return, %")
    ax.set_xlabel("Standart deviation, %")
    ax.set_title('Efficient_frontier')