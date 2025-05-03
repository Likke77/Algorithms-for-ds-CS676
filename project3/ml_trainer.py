from typing import Dict, Any, List
import pandas as pd
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, r2_score

def is_continuous(series: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(series) and series.nunique() > 20

def train_model(payload: Dict[str, Any]) -> Dict[str, Any]:
    data = pd.DataFrame(payload["data"])
    target_column = payload["target_column"]

    X = data.drop(columns=[target_column])
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if payload["model_type"] == "logistic_regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = accuracy_score(y_test, y_pred)
        metric_name = "Accuracy"
    elif payload["model_type"] == "linear_regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        score = r2_score(y_test, y_pred)
        metric_name = "R2 Score"
    else:
        raise ValueError("Unsupported model type.")

    return {
        "model_type": payload["model_type"],
        "metric_name": metric_name,
        "score": score
    }
