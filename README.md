# Final Project: Machine Learning Model + Streamlit App

🔗 **Live demo:** https://final-project-ai-fundamentals-softserve.streamlit.app/

A simple linear regression project: generate a synthetic dataset, train and
evaluate a regression model, serialize it, and serve it through an
interactive Streamlit web app that makes predictions and visualizes the
result against the actual data.

## Project structure

```
├── model.py          # Trains, evaluates, and saves the regression model + datasets
├── app.py            # Streamlit app: slider/number input, prediction, visualization
├── experiments.py    # Generalized comparison tool for different n_samples/noise values
└── requirements.txt  # Project dependencies
```

## How it works

1. **`model.py`** generates a synthetic dataset (`make_regression`), trains a
   `LinearRegression` model, evaluates it (MSE and R²), and saves the model
   and datasets with `joblib` (`linear_regression_model.joblib`, `X.joblib`,
   `y.joblib`).
2. **`app.py`** loads the saved model and lets the user pick an input feature
   value (via a slider or a synced number input), predicts the target value,
   and plots it against the full dataset, the actual nearest target value,
   and the regression line, highlighting the difference between predicted
   and actual values.
3. If the `.joblib` files are missing (e.g. on a fresh deploy with no
   separate build step), `app.py` **automatically trains the model on
   first run** and caches it, so no manual `python model.py` step is
   required before deploying.

## Running locally

```bash
pip install -r requirements.txt
python model.py          # optional: pre-train the model
python -m streamlit run app.py
```

## Extra features (practical tasks)

- **R² score** printed alongside MSE in `model.py`.
- **`experiments.py`** — compare MSE/R² across different `n_samples` and
  `noise` configurations without any hardcoded conclusions; run
  `python experiments.py` and interpret the numbers yourself.
- **Number input** in `app.py` as an alternative to the slider for exact
  value entry.
- **Regression line** drawn on the plot alongside the dataset and
  actual/predicted points.
- **Auto-training fallback** — the app trains the model itself on first run
  if the saved model files aren't found, instead of just showing an error.

## Notes

- The prediction for input feature `1.27` is `26.50989902...`, matching the
  reference example in the assignment exactly (same `random_state=42`
  throughout the pipeline).
- Deployed on Streamlit Community Cloud — no separate build step needed
  thanks to the auto-training fallback in `app.py`.
