import numpy as np
from sklearn.ensemble import RandomForestClassifier

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False
    print("[ensemble_model] تحذير: tensorflow غير مثبت، سيتم استخدام RF فقط.")


def build_lstm_model(input_shape):
    model = models.Sequential([
        layers.LSTM(32, input_shape=input_shape),
        layers.Dense(16, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    return model


class EnsembleIDS:
    def __init__(self, lstm_input_shape=None):
        self.rf = RandomForestClassifier(n_estimators=100, random_state=42)
        self.lstm = None
        self.use_lstm = TF_AVAILABLE and lstm_input_shape is not None
        if self.use_lstm:
            self.lstm = build_lstm_model(lstm_input_shape)

    def fit(self, X_train, y_train, X_train_lstm=None, epochs=5):
        self.rf.fit(X_train, y_train)
        if self.use_lstm and X_train_lstm is not None:
            self.lstm.fit(X_train_lstm, y_train, epochs=epochs, batch_size=64, verbose=0)

    def predict_proba(self, X_test, X_test_lstm=None) -> np.ndarray:
        rf_pred = self.rf.predict_proba(X_test)[:, 1]
        if self.use_lstm and X_test_lstm is not None:
            lstm_pred = self.lstm.predict(X_test_lstm, verbose=0).flatten()
            return (rf_pred + lstm_pred) / 2
        return rf_pred

    def predict(self, X_test, X_test_lstm=None, threshold=0.5) -> np.ndarray:
        proba = self.predict_proba(X_test, X_test_lstm)
        return (proba >= threshold).astype(int)


if __name__ == "__main__":
    from sklearn.metrics import accuracy_score, classification_report
    from src.data_pipeline import run_pipeline
    from src.feature_extractor import FeatureExtractor

    X_train, X_test, y_train, y_test, _ = run_pipeline()
    fe = FeatureExtractor()
    X_train_s = fe.fit_transform(X_train)
    X_test_s = fe.transform(X_test)
    X_train_l = fe.to_lstm_shape(X_train_s)
    X_test_l = fe.to_lstm_shape(X_test_s)

    model = EnsembleIDS(lstm_input_shape=(1, X_train_s.shape[1]))
    model.fit(X_train_s, y_train, X_train_l)
    preds = model.predict(X_test_s, X_test_l)

    print("Accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds))
