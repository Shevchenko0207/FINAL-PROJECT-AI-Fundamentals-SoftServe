"""
Практичні завдання 1 і 2 з нотатника:
1. Змінити n_samples зі 100 на 300 та порівняти результат.
2. Змінити noise з 20 на 50 і порівняти результат.

Узагальнений скрипт: задайте будь-які значення n_samples/noise для
порівняння — жодних заздалегідь написаних висновків, лише реальні
розраховані MSE та R2 для кожного варіанту.

Цей файл — окремий експеримент, який НЕ впливає на основний model.py
(щоб не змінити .joblib файли, які використовує app.py). Запуск:
    python experiments.py
"""

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

from model import train_regression_model


def run_experiment(n_samples, noise, random_state=42):
    """Навчає модель на синтетичних даних із заданими n_samples/noise і повертає (MSE, R2)."""
    X, y = make_regression(
        n_samples=n_samples, n_features=1, noise=noise, random_state=random_state
    )
    X = np.interp(X, (X.min(), X.max()), (-3, 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=random_state
    )

    model = train_regression_model(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return mse, r2


def compare(label, configs):
    """
    Виводить таблицю MSE/R2 для списку конфігурацій.

    Args:
        label (str): назва порівняння для заголовка.
        configs (list[dict]): список словників з ключами n_samples, noise.
    """
    print(f"=== {label} ===")
    for config in configs:
        mse, r2 = run_experiment(**config)
        print(
            f"n_samples={config['n_samples']}, noise={config['noise']}: MSE={mse:.2f}, R2={r2:.3f}"
        )
    print()


if __name__ == "__main__":
    compare(
        "n_samples: 100 vs 300 (noise=20)",
        [
            {"n_samples": 100, "noise": 20},
            {"n_samples": 300, "noise": 20},
        ],
    )

    compare(
        "noise: 20 vs 50 (n_samples=100)",
        [
            {"n_samples": 100, "noise": 20},
            {"n_samples": 100, "noise": 50},
        ],
    )
