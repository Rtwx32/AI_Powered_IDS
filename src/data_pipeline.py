import io
import pandas as pd
from sklearn.model_selection import train_test_split

NSL_KDD_COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds", "is_host_login",
    "is_guest_login", "count", "srv_count", "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate", "same_srv_rate", "diff_srv_rate",
    "srv_diff_host_rate", "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
]  # 41 base feature columns (label + difficulty are appended conditionally)

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]


class UnseenLabelEncoder:
    """Like sklearn's LabelEncoder, but never raises on a category it hasn't
    seen before. Unseen values map to one reserved 'unknown' code instead,
    which is what lets us fit ONCE on the training set and safely reuse the
    exact same mapping on the held-out test split and on any file a user
    uploads later (calling LabelEncoder.fit_transform() again on new data,
    as the original code did, silently reassigns integer codes and corrupts
    the model -- this class is the fix for that bug)."""

    def __init__(self):
        self.classes_ = []
        self._mapping = {}
        self.unknown_code = 0

    def fit(self, values):
        self.classes_ = sorted(set(str(v) for v in values))
        self._mapping = {v: i for i, v in enumerate(self.classes_)}
        self.unknown_code = len(self.classes_)
        return self

    def transform(self, values):
        return [self._mapping.get(str(v), self.unknown_code) for v in values]

    def fit_transform(self, values):
        self.fit(values)
        return self.transform(values)

    def inverse_transform_one(self, code):
        if 0 <= code < len(self.classes_):
            return self.classes_[code]
        return "unknown"


def _columns_for_width(n_cols: int):
    """NSL-KDD files come in three shapes in the wild: 41 cols (raw features
    only, e.g. unlabeled live traffic), 42 (features + label) or 43
    (features + label + difficulty, like KDDTrain+.txt). Detect which one
    we're looking at from the column count instead of hard-coding 43, so
    user-uploaded files of any of these shapes parse correctly."""
    base = NSL_KDD_COLUMNS
    if n_cols <= len(base):
        return base[:n_cols]
    extra = n_cols - len(base)
    names = list(base)
    if extra >= 1:
        names.append("label")
    if extra >= 2:
        names.append("difficulty")
    if extra > 2:
        names += [f"extra_{i}" for i in range(extra - 2)]
    return names


def _read_raw_text(source) -> str:
    """Read upload content as text, transparently handling .zip archives
    (KDD Cup data is commonly distributed zipped) and rejecting binary
    files with a clear message instead of producing garbage rows."""
    if hasattr(source, "read"):
        data = source.read()
        if isinstance(data, str):
            data = data.encode("utf-8", errors="ignore")
    else:
        with open(source, "rb") as f:
            data = f.read()

    # zip archive? pull out the largest member and use that
    if data[:4] == b"PK\x03\x04":
        import zipfile
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            members = [m for m in zf.infolist() if not m.is_dir()]
            if not members:
                raise ValueError("الأرشيف المضغوط فارغ.")
            biggest = max(members, key=lambda m: m.file_size)
            data = zf.read(biggest)

    # binary guard: a real CSV is overwhelmingly printable text. NUL bytes
    # or a high ratio of undecodable bytes means this isn't a data file
    # (e.g. an image, a model, a corrupt download) -- fail clearly.
    if b"\x00" in data[:4096]:
        raise ValueError("هذا ليس ملف بيانات نصي (CSV). الرجاء رفع ملف NSL-KDD/KDD بصيغة نصية أو مضغوطة .zip.")

    return data.decode("utf-8", errors="ignore")


def parse_nsl_kdd(source) -> pd.DataFrame:
    """Parse NSL-KDD- or KDD-Cup-formatted data from a path, a file-like
    object, or a .zip of one. Auto-detects column width (41/42/43),
    tolerates header/comment/blank lines and a few malformed rows, and
    normalizes labels.
    """
    raw = _read_raw_text(source)

    # drop blank lines and ARFF/comment-style lines (@relation, @attribute, %...)
    lines = [
        l for l in raw.splitlines()
        if l.strip() and not l.lstrip().startswith(("@", "#", "%"))
    ]
    if not lines:
        raise ValueError("الملف لا يحتوي على أي سطور بيانات صالحة (تأكد أنه CSV مفصول بفواصل).")

    # decide the column count by majority vote over a sample of lines,
    # preferring real-looking rows (>=10 cols) so short junk/comment lines
    # can't outvote the data.
    from collections import Counter
    sample = lines[:300]
    counts_list = [l.count(",") + 1 for l in sample]
    plausible = [c for c in counts_list if c >= 10]
    pool = plausible if plausible else counts_list
    n_cols = Counter(pool).most_common(1)[0][0]

    # sanity: NSL-KDD/KDD rows have ~41-43 columns. If the dominant width is
    # tiny, this file isn't in the expected format -- say so plainly.
    if n_cols < 10:
        raise ValueError(
            "تنسيق الملف غير متوقع: يجب أن يكون بصيغة NSL-KDD/KDD "
            "(41 إلى 43 عموداً مفصولة بفواصل). تأكد أنك رفعت ملف البيانات الصحيح."
        )

    columns = _columns_for_width(n_cols)
    buffer = io.StringIO("\n".join(lines))
    try:
        df = pd.read_csv(buffer, names=columns, header=None)
    except Exception:
        buffer.seek(0)
        df = pd.read_csv(buffer, names=columns, header=None,
                          engine="python", on_bad_lines="skip")

    # strip a literal header row if the file had one
    if len(df) and "duration" in df.columns:
        if str(df.iloc[0]["duration"]).strip().lower() == "duration":
            df = df.iloc[1:].reset_index(drop=True)

    # normalize labels: KDD Cup '99 writes them with a trailing dot
    # ("normal.", "neptune.") -- strip it so they match NSL-KDD names and so
    # the "normal" check in encode_labels works.
    if "label" in df.columns:
        df["label"] = df["label"].astype(str).str.strip().str.rstrip(".")

    # coerce numeric feature columns; non-numeric -> NaN, dropped later
    text_cols = {"protocol_type", "service", "flag", "label"}
    for col in df.columns:
        if col not in text_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def load_data(path: str) -> pd.DataFrame:
    try:
        df = parse_nsl_kdd(path)
        print(f"[data_pipeline] تم تحميل {len(df)} سجل من {path}")
        return df
    except FileNotFoundError:
        print(f"[data_pipeline] لم يتم إيجاد {path} → سيتم توليد بيانات تجريبية للتجربة")
        return generate_synthetic_data()


def generate_synthetic_data(n_rows: int = 2000) -> pd.DataFrame:
    import numpy as np
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "duration": rng.integers(0, 1000, n_rows),
        "protocol_type": rng.choice(["tcp", "udp", "icmp"], n_rows),
        "service": rng.choice(["http", "ftp", "smtp", "dns"], n_rows),
        "flag": rng.choice(["SF", "S0", "REJ"], n_rows),
        "src_bytes": rng.integers(0, 5000, n_rows),
        "dst_bytes": rng.integers(0, 5000, n_rows),
        "count": rng.integers(0, 500, n_rows),
        "srv_count": rng.integers(0, 500, n_rows),
        "serror_rate": rng.random(n_rows),
        "same_srv_rate": rng.random(n_rows),
        "label": rng.choice(["normal", "dos", "probe", "r2l", "u2r"],
                             n_rows, p=[0.6, 0.2, 0.1, 0.05, 0.05]),
    })
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop_duplicates()
    df = df.dropna()
    return df


def encode_categorical(df: pd.DataFrame, encoders=None):
    """Encode protocol_type/service/flag to integers.

    If `encoders` is None we FIT new encoders on this dataframe (training
    mode) and return them so the caller can persist + reuse them later.
    If `encoders` is given, we only TRANSFORM with the existing mapping
    (inference mode) -- this is what guarantees a record encoded during
    training and the *same* record re-encoded later (test split, or a
    brand-new uploaded file) end up with identical integer codes.
    """
    df = df.copy()
    fit_mode = encoders is None
    encoders = {} if fit_mode else dict(encoders)
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        values = df[col].astype(str)
        if fit_mode:
            enc = UnseenLabelEncoder().fit(values)
            encoders[col] = enc
        else:
            enc = encoders.get(col) or UnseenLabelEncoder().fit(values)
            encoders[col] = enc
        df[col] = enc.transform(values)
    return df, encoders


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "label" in df.columns:
        df["label_binary"] = df["label"].apply(lambda x: 0 if str(x) == "normal" else 1)
    return df


def split_data(df: pd.DataFrame, target_col: str = "label_binary", test_size: float = 0.2):
    drop_cols = [c for c in ["label", "label_binary", "difficulty"] if c in df.columns]
    X = df.drop(columns=drop_cols)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)


def run_pipeline(path: str = "data/KDDTrain+.txt"):
    """End-to-end prep used for training. Returns the train/test split AND
    the fitted categorical encoders, since downstream code (model_registry)
    needs to persist those encoders for consistent inference later."""
    df = load_data(path)
    df = clean_data(df)
    df, encoders = encode_categorical(df)
    df = encode_labels(df)
    X_train, X_test, y_train, y_test = split_data(df)
    print(f"[data_pipeline] train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test, encoders


if __name__ == "__main__":
    run_pipeline()
