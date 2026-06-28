import numpy as np

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("[xai_explainer] تحذير: shap غير مثبت (pip install shap)")

try:
    from lime.lime_tabular import LimeTabularExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False
    print("[xai_explainer] تحذير: lime غير مثبت (pip install lime)")


class XAIExplainer:
    def __init__(self, rf_model, X_train, feature_names):
        self.rf_model = rf_model
        self.X_train = X_train
        self.feature_names = feature_names
        self.shap_explainer = shap.TreeExplainer(rf_model) if SHAP_AVAILABLE else None
        self.lime_explainer = (
            LimeTabularExplainer(
                X_train, feature_names=feature_names,
                class_names=["normal", "attack"], mode="classification"
            ) if LIME_AVAILABLE else None
        )

    def explain_with_shap(self, X_sample):
        if not SHAP_AVAILABLE:
            raise RuntimeError("shap غير متاح")
        return self.shap_explainer.shap_values(X_sample)

    def plot_summary(self, X_test):
        if not SHAP_AVAILABLE:
            raise RuntimeError("shap غير متاح")
        shap_values = self.shap_explainer.shap_values(X_test)
        shap.summary_plot(shap_values, X_test, feature_names=self.feature_names)

    def explain_single_lime(self, x_row, predict_fn):
        if not LIME_AVAILABLE:
            raise RuntimeError("lime غير متاح")
        return self.lime_explainer.explain_instance(x_row, predict_fn, num_features=10)

    def top_features_text(self, shap_values_row, top_n=3) -> str:
        idx_sorted = np.argsort(-np.abs(shap_values_row))[:top_n]
        parts = []
        for i in idx_sorted:
            direction = "مرتفع" if shap_values_row[i] > 0 else "منخفض"
            parts.append(f"{self.feature_names[i]} ({direction})")
        return "تم تصنيف هذا السجل كـ attack بسبب: " + "، ".join(parts)


if __name__ == "__main__":
    from src.data_pipeline import run_pipeline
    from src.feature_extractor import FeatureExtractor
    from sklearn.ensemble import RandomForestClassifier

    X_train, X_test, y_train, y_test, _ = run_pipeline()
    fe = FeatureExtractor()
    X_train_s = fe.fit_transform(X_train)
    X_test_s = fe.transform(X_test)

    rf = RandomForestClassifier(n_estimators=50, random_state=42).fit(X_train_s, y_train)

    if SHAP_AVAILABLE:
        xai = XAIExplainer(rf, X_train_s, fe.feature_names_)
        shap_vals = xai.explain_with_shap(X_test_s[:5])
        print(xai.top_features_text(shap_vals[1][0]))
