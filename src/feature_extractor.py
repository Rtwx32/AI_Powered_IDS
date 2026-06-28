import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureExtractor:
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names_ = None

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        self.feature_names_ = list(X.columns)
        return self.scaler.fit_transform(X)

    def transform(self, X: pd.DataFrame) -> np.ndarray:

        if self.feature_names_ is not None:
            X = X.reindex(columns=self.feature_names_, fill_value=0)
        return self.scaler.transform(X)

    def to_lstm_shape(self, X_scaled: np.ndarray, timesteps: int = 1) -> np.ndarray:
        return X_scaled.reshape((X_scaled.shape[0], timesteps, X_scaled.shape[1]))

    def save(self, path: str) -> None:
        joblib.dump({"scaler": self.scaler, "feature_names_": self.feature_names_}, path)

    @classmethod
    def load(cls, path: str) -> "FeatureExtractor":
        data = joblib.load(path)
        obj = cls()
        obj.scaler = data["scaler"]
        obj.feature_names_ = data["feature_names_"]
        return obj


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    X_train, X_test, y_train, y_test, _ = run_pipeline()
    fe = FeatureExtractor()
    X_train_scaled = fe.fit_transform(X_train)
    X_train_lstm = fe.to_lstm_shape(X_train_scaled)
    print("Scaled shape:", X_train_scaled.shape)
    print("LSTM shape:", X_train_lstm.shape)
