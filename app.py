from __future__ import annotations
import streamlit as st
from joblib import load
import numpy as np
from numpy.typing import ArrayLike
import matplotlib

matplotlib.use(
    "Agg"
)  # неінтерактивний бекенд, щоб уникнути крашу процесу Streamlit на Windows
import matplotlib.pyplot as plt

from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from model import (
    train_regression_model,
    save_regression_model,
    evaluate_regression_model,
    save_initial_datasets,
)


def load_and_predict(
    X: ArrayLike, filename: str = "linear_regression_model.joblib"
) -> ArrayLike:
    """
    Deserialize and load the regression model and use it to predict on user provided data.

    This function takes a file name 'filename' that has a default value.
    It uses Joblib 'load' to load the model using the provided file name.
    When the model is loaded, call its `predict` method on provied data.

    Args:
        X (array-like): User provided data used for prediction.
        filename (str): Name of the file that is used to store the model.

    Returns:
        np.ndarray: Predicted value.
    """

    model = load(filename)
    y = model.predict(X)

    return y


def _model_files_exist():
    """Перевіряє, чи є файли моделі/датасетів на диску."""
    import os

    required_files = ["linear_regression_model.joblib", "X.joblib", "y.joblib"]
    return all(os.path.exists(f) for f in required_files)


@st.cache_resource(show_spinner="Training regression model...")
def _ensure_model_trained():
    """
    Якщо .joblib файли моделі відсутні (наприклад, при першому запуску на
    Streamlit Cloud, де немає окремого build-кроку) — тренує модель
    автоматично, той самий пайплайн, що й у model.py.
    Кешовано через @st.cache_resource, тому виконується лише один раз
    за час життя застосунку, а не при кожному оновленні сторінки.
    """
    if _model_files_exist():
        return

    X, y = make_regression(n_samples=100, n_features=1, noise=20, random_state=42)
    X = np.interp(X, (X.min(), X.max()), (-3, 3))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = train_regression_model(X_train, y_train)
    evaluate_regression_model(model, X_test, y_test)
    save_regression_model(model)
    save_initial_datasets(X, y)


def create_streamlit_app():
    """
    Creates a Streamlit web application for making predictions with a simple regression model.

    This function sets up a Streamlit app with a user interface for inputting a single feature
    value and making predictions using a pre-trained regression model. The app includes:

    - A title displayed at the top of the app.
    - A slider for the user to select an input feature value within a specified range (-3.0 to 3.0).
    - A "Predict value" button that, when clicked, triggers the prediction process.
    - Upon clicking the "Predict value" button, the function:
        - Calls `load_and_predict`, passing the selected feature as input, to load the regression model
          and make a prediction.
        - Displays the prediction result on the app.
        - Calls `visualize_difference`, passing the input feature and the prediction result,
          to visualize the difference between the predicted value and the actual value in the original dataset.

    Note: This function does not return any value. It directly manipulates the Streamlit app's UI by
    writing content and rendering UI elements.
    """
    # Streamlit app title
    st.title("Simple Regression model prediction")

    # Якщо файлів моделі немає (наприклад, чистий деплой без окремого
    # build-кроку) — тренуємо модель автоматично один раз.
    _ensure_model_trained()

    # User input for new prediction using a slider
    input_feature = st.slider(
        "Input Feature for Prediction",
        min_value=-3.0,
        max_value=3.0,
        value=0.0,
    )

    # Практичне завдання 3: альтернативний спосіб введення значення — number_input,
    # синхронізований зі слайдером (введене число одразу оновлює повзунок).
    input_feature = st.number_input(
        "Or type an exact value",
        min_value=-3.0,
        max_value=3.0,
        value=input_feature,
        step=0.01,
    )

    # Button to make a prediction
    if st.button("Predict value"):
        # 1. Call load_and_predict functions.
        # Make sure you convert the input_feature to a matrix before calling load_and_predict, e.g., load_and_predict([[input_feature]])
        prediction = load_and_predict([[input_feature]])

        # 2. Display the prediction.
        st.write(f"Prediction: {prediction[0]}")

        # 4. Call visualize_difference to display a plot visualizing the difference between actual and perdicted value.
        visualize_difference(input_feature, prediction)


def visualize_difference(input_feature: float, prediction: ArrayLike):
    """
    Deserialize and load the initial datasets. Calculate the difference between actual data
    in the 'y' dataset and the predicted value for a given 'input_feature'.

    Visualize the difference by plotting the entire 'X' & 'y' as a Scatter plot. Then add
    a blue dot that represents the actual target value, and a red dot that represents the predicted target value for the given 'input_feature'.
    Add a dashed line connects these points, highlighting the difference between them, which is annotated on the plot.

    Args:
        input_feature (float): User provided data used for prediction.
        prediction (array-like): Predicted value.

    """
    # Load the X and y datasets
    X_filename = "X.joblib"
    y_filename = "y.joblib"

    X = load(X_filename)

    y = load(y_filename)

    actual_target = y[_index_of_closest(X, input_feature)]

    # Calculate difference
    difference = actual_target - prediction

    # Visualization
    fig = plt.figure(figsize=(6, 4))

    # Plot the entire dataset (X, y) as grey dots to visualize the data distribution.
    plt.scatter(X, y, color="grey", label="Dataset")

    # Практичне завдання 4: пряма лінія регресії моделі поверх точок датасету.
    regression_model = load("linear_regression_model.joblib")
    x_line = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    y_line = regression_model.predict(x_line)
    plt.plot(x_line, y_line, color="green", linewidth=2, label="Regression Line")

    # Plot the actual target value for a specific input feature as a blue dot.
    plt.scatter(input_feature, actual_target, color="blue", label="Actual Target")

    # Plot the predicted target value for the same input feature as a red dot.
    plt.scatter(input_feature, prediction, color="red", label="Predicted Target")

    # Display a legend on the plot to label the different scatter points (dataset, actual target, predicted target).
    plt.legend()

    # Set the title of the plot, describing what is being visualized.
    plt.title("Prediction vs Actual Target")

    # Set the label for the x-axis to 'Feature', indicating that the x-axis represents the input features.
    plt.xlabel("Feature")

    # Set the label for the y-axis to 'Target', indicating that the y-axis represents the target values (actual or predicted).
    plt.ylabel("Target")

    # Enable a grid on the plot to improve readability.
    plt.grid(True)

    # Draw a dashed line ('k--' for black dashed line) between the actual and predicted target values to visually represent the difference.
    plt.plot(
        [input_feature, input_feature],
        [float(np.ravel(actual_target)[0]), float(np.ravel(prediction)[0])],
        "k--",
    )

    # Annotate the plot with the difference between the actual and predicted target values, positioned halfway between them and offset slightly for visibility.
    midpoint_y = (
        float(np.ravel(actual_target)[0]) + float(np.ravel(prediction)[0])
    ) / 2
    plt.annotate(
        f"Difference = {float(np.ravel(difference)[0]):.2f}",
        xy=(input_feature, midpoint_y),
        xytext=(input_feature + 0.15, midpoint_y),
    )

    st.pyplot(fig)


# This is a helper function. No need to edit it
def _index_of_closest(X: ArrayLike, k: float) -> int:
    """
    This function takes an array-like object `X` and a float `k`, and returns the index of the
    element in `X` that is closest to `k`. The function first converts `X` into a NumPy array
    (if it isn't one already) to ensure compatibility with NumPy operations. It then calculates
    the absolute difference between each element in `X` and `k`, identifies the minimum value
    among these differences, and returns the index of this minimum difference.

    Args:
        X (ArrayLike): An array-like object containing numerical data. It can be a list, tuple,
      or any object that can be converted to a NumPy array.
        k (float): The target value to which the closest element in `X` is sought.

    Returns:
        int: The index of the element in `X` that is closest to the value `k`.
    Returns:
        int: Index for the closest value to k in X.
    Finds the index of the element in `X` that is closest to the value `k`.

    """
    X = np.asarray(X)
    idx = (np.abs(X - k)).argmin()
    return idx


if __name__ == "__main__":
    create_streamlit_app()
