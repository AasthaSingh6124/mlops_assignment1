# misc.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_data() -> pd.DataFrame:
    """
    Loads the Boston dataset from the CMU mirror (as required by the assignment).
    Returns a DataFrame with features and target column 'MEDV'.
    """
    data_url = "http://lib.stat.cmu.edu/datasets/boston"
    raw_df = pd.read_csv(data_url, sep=r"\s+", skiprows=22, header=None)
    data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
    target = raw_df.values[1::2, 2]
    feature_names = ['CRIM','ZN','INDUS','CHAS','NOX','RM','AGE','DIS','RAD','TAX','PTRATIO','B','LSTAT']
    df = pd.DataFrame(data, columns=feature_names)
    df['MEDV'] = target
    return df

def get_X_y(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    X = df.drop(columns=['MEDV'])
    y = df['MEDV'].astype(float)
    return X, y

def split_data(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

def build_pipeline(model, scale: bool = True) -> Pipeline:
    steps = []
    if scale:
        steps.append(('scaler', StandardScaler()))
    steps.append(('model', model))
    return Pipeline(steps)

def cross_validate_model(X, y, pipeline: Pipeline, cv: int = 5) -> float:
    """Return mean MSE from cross-validation (uses neg_mean_squared_error)."""
    scores = cross_val_score(pipeline, X, y, scoring='neg_mean_squared_error', cv=cv)
    mse_scores = -scores
    return float(mse_scores.mean())

def train_and_evaluate(pipeline: Pipeline, X_train, X_test, y_train, y_test) -> float:
    """Fit on X_train and return MSE on X_test."""
    pipeline.fit(X_train, y_train)
    preds = pipeline.predict(X_test)
    return float(mean_squared_error(y_test, preds))
