
import hashlib
import json
import os
import threading
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, confusion_matrix,
)

from .data_pipeline import (
    load_data, clean_data, encode_categorical, encode_labels, split_data,
    NSL_KDD_COLUMNS,
)
from .feature_extractor import FeatureExtractor
from .ensemble_model import EnsembleIDS, TF_AVAILABLE
from .alert_generator import generate_alerts_batch, filter_by_severity

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
LEARNED_DIR = os.path.join(DATA_DIR, "learned")
MODELS_DIR = os.path.join(BASE_DIR, "models")
DEFAULT_DATA_PATH = os.path.join(DATA_DIR, "KDDTrain+.txt")

ENCODERS_PATH = os.path.join(MODELS_DIR, "encoders.joblib")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.joblib")
RF_PATH = os.path.join(MODELS_DIR, "rf_model.joblib")
META_PATH = os.path.join(MODELS_DIR, "meta.json")


FEATURE_COLS = list(NSL_KDD_COLUMNS)
TRAINING_COLS = FEATURE_COLS + ["label"]
TEXT_COLS = {"protocol_type", "service", "flag", "label"}

DISPLAY_FIELDS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "count", "srv_count", "serror_rate", "same_srv_rate", "label",
]


def _native(value):
    """Convert numpy/pandas scalars to plain python types so they survive
    Flask's jsonify() without a TypeError."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def _metrics_from(y_true, proba):
    preds = (np.asarray(proba) >= 0.5).astype(int)
    return {
        "accuracy": round(float(accuracy_score(y_true, preds)), 4),
        "precision": round(float(precision_score(y_true, preds, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, preds, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, preds, zero_division=0)), 4),
        "confusion_matrix": confusion_matrix(y_true, preds).tolist(),
        "test_rows": int(len(y_true)),
    }


class ModelRegistry:
    def __init__(self):
        self.encoders = None
        self.feature_extractor: FeatureExtractor | None = None
        self.model: EnsembleIDS | None = None
        self.metrics = None
        self.trained_at = None
        self.train_rows = None
        self.learned_datasets = 0
        self.total_train_rows = None
        self._demo_alerts_cache = None

        self._train_lock = threading.Lock()
        self._training = False
        self._last_learning = None

   
    @staticmethod
    def _prepare_frame_for_training(df: pd.DataFrame) -> pd.DataFrame:
        """Reduce any parsed frame to the canonical 41 features + label,
        filling missing features with 0 and dropping rows that have no
        label (you can't learn from a row without ground truth)."""
        df = df.reindex(columns=TRAINING_COLS)
        for col in FEATURE_COLS:
            if col not in TEXT_COLS:
                df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df[df["label"].notna()].copy()
        return df

    def _load_learned_frames(self):
        frames = []
        if not os.path.isdir(LEARNED_DIR):
            return frames
        for name in sorted(os.listdir(LEARNED_DIR)):
            if not name.endswith(".csv"):
                continue
            try:
                f = pd.read_csv(os.path.join(LEARNED_DIR, name))
                frames.append(self._prepare_frame_for_training(f))
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[model_registry] تخطّي ملف تعلّم تالف {name}: {exc}")
        return frames

    @staticmethod
    def _anti_join(learned: pd.DataFrame, benchmark: pd.DataFrame) -> pd.DataFrame:
        
        if learned.empty:
            return learned
        keys = TRAINING_COLS
        left = learned.copy()
        right = benchmark[keys].drop_duplicates().copy()
        
        for col in keys:
            if col not in TEXT_COLS:
                left[col] = pd.to_numeric(left[col], errors="coerce").astype("float64")
                right[col] = pd.to_numeric(right[col], errors="coerce").astype("float64")
        merged = left.merge(right, on=keys, how="left", indicator=True)
        mask = merged["_merge"].to_numpy() == "left_only"
        return learned[mask].reset_index(drop=True)


    def _artifacts_exist(self):
        return all(os.path.exists(p) for p in [ENCODERS_PATH, SCALER_PATH, RF_PATH, META_PATH])

    def _load_cached(self):
        self.encoders = joblib.load(ENCODERS_PATH)
        self.feature_extractor = FeatureExtractor.load(SCALER_PATH)
        rf = joblib.load(RF_PATH)
        self.model = EnsembleIDS()
        self.model.rf = rf
        self.model.use_lstm = False
        with open(META_PATH, encoding="utf-8") as f:
            meta = json.load(f)
        self.metrics = meta.get("metrics")
        self.trained_at = meta.get("trained_at")
        self.train_rows = meta.get("train_rows")
        self.learned_datasets = meta.get("learned_datasets", 0)
        self.total_train_rows = meta.get("total_train_rows", self.train_rows)

    def _save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        joblib.dump(self.encoders, ENCODERS_PATH)
        self.feature_extractor.save(SCALER_PATH)
        joblib.dump(self.model.rf, RF_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump({
                "trained_at": self.trained_at,
                "train_rows": self.train_rows,
                "total_train_rows": self.total_train_rows,
                "learned_datasets": self.learned_datasets,
                "metrics": self.metrics,
            }, f, ensure_ascii=False, indent=2)


    def train(self, data_path: str = DEFAULT_DATA_PATH):
        
        base = clean_data(self._prepare_frame_for_training(load_data(data_path)))
        base = encode_labels(base)
        base_train, base_test = train_test_split(
            base, test_size=0.2, random_state=42, stratify=base["label_binary"]
        )

        
        learned_frames = self._load_learned_frames()
        if learned_frames:
            learned = encode_labels(clean_data(pd.concat(learned_frames, ignore_index=True)))
            learned = self._anti_join(learned, base_test)
            train_df = pd.concat([base_train, learned], ignore_index=True)
        else:
            train_df = base_train.copy()
        train_df = train_df.drop_duplicates().reset_index(drop=True)

        n_learned = len(self._learned_files())
        print(f"[model_registry] تدريب على {len(train_df)} سجل "
              f"(أساسي + {n_learned} مجموعة بيانات مُتعلَّمة) ...")

        
        train_enc, encoders = encode_categorical(train_df)
        drop = [c for c in ["label", "label_binary"] if c in train_enc.columns]
        X_train = train_enc.drop(columns=drop)
        y_train = train_df["label_binary"].values

        fe = FeatureExtractor()
        X_train_s = fe.fit_transform(X_train)
        model = EnsembleIDS()
        model.fit(X_train_s, y_train)

        
        test_enc, _ = encode_categorical(base_test, encoders=encoders)
        X_test = test_enc.drop(columns=[c for c in ["label", "label_binary"] if c in test_enc.columns])
        X_test_s = fe.transform(X_test)
        proba = model.predict_proba(X_test_s)
        self.metrics = _metrics_from(base_test["label_binary"].values, proba)

        self.encoders = encoders
        self.feature_extractor = fe
        self.model = model
        self.trained_at = datetime.now().isoformat()
        self.learned_datasets = n_learned
        self.train_rows = int(len(base_train))
        self.total_train_rows = int(len(train_df))
        self._save()
        print(f"[model_registry] انتهى التدريب — benchmark accuracy={self.metrics['accuracy']}")
        return self.metrics

    def _learned_files(self):
        if not os.path.isdir(LEARNED_DIR):
            return []
        return [n for n in os.listdir(LEARNED_DIR) if n.endswith(".csv")]

    def get_or_train(self, data_path: str = DEFAULT_DATA_PATH, force: bool = False):
        if not force and self._artifacts_exist():
            try:
                self._load_cached()
                print("[model_registry] تم تحميل نموذج محفوظ مسبقاً (بدون إعادة تدريب).")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[model_registry] فشل تحميل النموذج المحفوظ ({exc}) — إعادة تدريب.")
                self.train(data_path)
        else:
            self.train(data_path)
        self._prepare_demo_sample(data_path)
        return self.metrics


    def incorporate_labeled_data(self, raw_df: pd.DataFrame, source_name: str = "upload"):
        
        if "label" not in raw_df.columns:
            return {"learned": False, "reason": "no_label"}

        prepared = self._prepare_frame_for_training(raw_df)
        prepared = clean_data(prepared)
        if prepared.empty:
            return {"learned": False, "reason": "no_label"}

        os.makedirs(LEARNED_DIR, exist_ok=True)
        digest = hashlib.md5(prepared.to_csv(index=False).encode()).hexdigest()[:12]
        path = os.path.join(LEARNED_DIR, f"learned_{digest}.csv")
        already = os.path.exists(path)

        before = dict(self.metrics) if self.metrics else None
        if already:
            return {
                "learned": False,
                "reason": "duplicate",
                "rows_in_file": int(len(prepared)),
                "metrics_before": before,
                "metrics_after": before,
                "learned_datasets": self.learned_datasets,
                "total_train_rows": self.total_train_rows,
            }

        prepared.to_csv(path, index=False)
        self.train()  
        self._prepare_demo_sample()
        return {
            "learned": True,
            "reason": "ok",
            "rows_added": int(len(prepared)),
            "metrics_before": before,
            "metrics_after": dict(self.metrics),
            "learned_datasets": self.learned_datasets,
            "total_train_rows": self.total_train_rows,
        }


    def _prepare_demo_sample(self, data_path: str = DEFAULT_DATA_PATH, sample_size: int = 1200):
        df = load_data(data_path)
        df = clean_data(df)
        df, _ = encode_categorical(df, encoders=self.encoders)
        df = encode_labels(df)
        _, X_test, _, y_test = split_data(df)
        X_test_s = self.feature_extractor.transform(X_test)
        proba = self.model.predict_proba(X_test_s)

        n = min(sample_size, len(X_test))
        rng = np.random.RandomState(7)
        idx = rng.choice(len(X_test), size=n, replace=False)

        decoded = self._decode_display_rows(df.loc[X_test.index].iloc[idx])
        sample_proba = proba[idx]
        alerts = generate_alerts_batch(sample_proba, decoded)
        times = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="min")
        for a, t in zip(alerts, times):
            a["timestamp"] = t.isoformat()

        self._demo_alerts_cache = {
            "alerts": alerts,
            "label_distribution": self._label_distribution(df.loc[X_test.index].iloc[idx]),
        }

    def demo_alerts(self):
        if self._demo_alerts_cache is None:
            self.get_or_train()
        return self._demo_alerts_cache


    def _decode_display_rows(self, df: pd.DataFrame):
        rows = []
        for _, row in df.iterrows():
            d = {}
            for col in DISPLAY_FIELDS:
                if col not in df.columns:
                    continue
                val = row[col]
                if col in ("protocol_type", "service", "flag") and col in self.encoders:
                    try:
                        val = self.encoders[col].inverse_transform_one(int(val))
                    except (ValueError, TypeError):
                        val = "unknown"
                d[col] = _native(val)
            rows.append(d)
        return rows

    def _label_distribution(self, df: pd.DataFrame):
        if "label" not in df.columns:
            return None
        return {str(k): int(v) for k, v in df["label"].value_counts().items()}

    def analyze_dataframe(self, raw_df: pd.DataFrame, max_alerts: int = 150):
        if self.model is None:
            self.get_or_train()

        df = clean_data(raw_df)
        df, _ = encode_categorical(df, encoders=self.encoders)
        has_label = "label" in df.columns
        if has_label:
            df = encode_labels(df)

        drop_cols = [c for c in ["label", "label_binary", "difficulty"] if c in df.columns]
        drop_cols += [c for c in df.columns if str(c).startswith("extra_")]
        X = df.drop(columns=drop_cols)
        X_s = self.feature_extractor.transform(X)
        proba = np.asarray(self.model.predict_proba(X_s))


        high = int(np.sum(proba >= 0.85))
        medium = int(np.sum((proba >= 0.6) & (proba < 0.85)))
        low = int(len(proba) - high - medium)

 
        n = len(proba)
        top_idx = np.argsort(-proba)[:max_alerts] if n else np.array([], dtype=int)
        decoded_rows = self._decode_display_rows(df.iloc[top_idx])
        alerts_sample = generate_alerts_batch(proba[top_idx], decoded_rows)

        result = {
            "total_records": int(n),
            "severity_counts": {"high": high, "medium": medium, "low": low},
            "avg_probability": round(float(np.mean(proba)), 4) if n else 0.0,
            "attack_type_distribution": self._label_distribution(df) if has_label else None,
            "has_label": has_label,
            "alerts_sample": alerts_sample,
        }

        if has_label:
            y_true = df["label_binary"].values
            result["evaluation"] = {
                k: v for k, v in _metrics_from(y_true, proba).items()
                if k in ("accuracy", "precision", "recall", "f1")
            }
        return result


    def is_training(self):
        return self._training

    def _save_to_store(self, raw_df: pd.DataFrame):
        """Persist a labeled upload to the learning store. Returns
        (path, already_existed, rows) or (None, False, 0) if unusable."""
        if "label" not in raw_df.columns:
            return None, False, 0
        prepared = clean_data(self._prepare_frame_for_training(raw_df))
        if prepared.empty:
            return None, False, 0
        os.makedirs(LEARNED_DIR, exist_ok=True)
        digest = hashlib.md5(prepared.to_csv(index=False).encode()).hexdigest()[:12]
        path = os.path.join(LEARNED_DIR, f"learned_{digest}.csv")
        already = os.path.exists(path)
        if not already:
            prepared.to_csv(path, index=False)
        return path, already, int(len(prepared))

    def learn_in_background(self, raw_df: pd.DataFrame, source_name: str = "upload"):
        """Save the upload, then retrain on a background thread so the HTTP
        request can return immediately. The page polls /api/status and reads
        the result from self._last_learning once training finishes."""
        if "label" not in raw_df.columns:
            return {"queued": False, "learned": False, "reason": "no_label"}


        with self._train_lock:
            if self._training:
                return {"queued": False, "learned": False, "reason": "busy"}

            path, already, rows = self._save_to_store(raw_df)
            if path is None:
                return {"queued": False, "learned": False, "reason": "no_label"}
            if already:
                return {"queued": False, "learned": False, "reason": "duplicate",
                        "rows_in_file": rows}

            before = dict(self.metrics) if self.metrics else None
            self._training = True  

        def _job():
            try:
                self.train()
                self._prepare_demo_sample()
                self._last_learning = {
                    "learned": True,
                    "source": source_name,
                    "rows_added": rows,
                    "metrics_before": before,
                    "metrics_after": dict(self.metrics),
                    "learned_datasets": self.learned_datasets,
                    "total_train_rows": self.total_train_rows,
                    "finished_at": datetime.now().isoformat(),
                }
            except Exception as exc:  # pragma: no cover - defensive
                self._last_learning = {"learned": False, "reason": "error", "detail": str(exc)}
            finally:
                self._training = False

        threading.Thread(target=_job, daemon=True).start()
        return {"queued": True, "rows_added": rows, "metrics_before": before}

    def last_learning(self):
        return self._last_learning



registry = ModelRegistry()
