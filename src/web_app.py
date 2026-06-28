
import threading
import webbrowser

from flask import Flask, jsonify, render_template, request

from .data_pipeline import parse_nsl_kdd
from .ensemble_model import TF_AVAILABLE
from .model_registry import registry

ALLOWED_EXTENSIONS = {".txt", ".csv", ".data"}
MAX_UPLOAD_ROWS_PREVIEW = 300


def create_app():
    app = Flask(__name__, template_folder="../templates", static_folder="../static")
    app.config["MAX_CONTENT_LENGTH"] = 200 * 1024 * 1024  # 200 MB uploads

    @app.errorhandler(413)
    def too_large(_e):
        return jsonify({"error": "الملف كبير جداً (الحد 200MB). جرّب رفع الملف مضغوطاً .zip أو عيّنة أصغر."}), 413

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/api/status")
    def status():
        if registry.model is None:
            registry.get_or_train()
        return jsonify({
            "trained_at": registry.trained_at,
            "train_rows": registry.train_rows,
            "total_train_rows": registry.total_train_rows,
            "learned_datasets": registry.learned_datasets,
            "metrics": registry.metrics,
            "training": registry.is_training(),
            "last_learning": registry.last_learning(),
            "tf_available": TF_AVAILABLE,
        })

    @app.route("/api/train", methods=["POST"])
    def train():
        metrics = registry.get_or_train(force=True)
        return jsonify({"metrics": metrics, "trained_at": registry.trained_at})

    @app.route("/api/demo")
    def demo():
        data = registry.demo_alerts()
        alerts = data["alerts"]
        severity_counts = {
            s: sum(1 for a in alerts if a["severity"] == s) for s in ["high", "medium", "low"]
        }
        return jsonify({
            "total_records": len(alerts),
            "severity_counts": severity_counts,
            "label_distribution": data["label_distribution"],
            "alerts_sample": sorted(alerts, key=lambda a: a["timestamp"])[-200:],
        })

    @app.route("/api/analyze", methods=["POST"])
    def analyze():
        if "file" not in request.files:
            return jsonify({"error": "لم يتم رفع أي ملف (file)."}), 400
        f = request.files["file"]
        if not f.filename:
            return jsonify({"error": "اسم الملف غير صالح."}), 400

        try:
            df = parse_nsl_kdd(f.stream)
        except Exception as exc:
            return jsonify({"error": f"تعذّر قراءة الملف كبيانات NSL-KDD: {exc}"}), 400

        if df.empty:
            return jsonify({"error": "الملف لا يحتوي على أي سجلات صالحة."}), 400

        try:
            result = registry.analyze_dataframe(df)
        except Exception as exc:
            return jsonify({"error": f"تعذّر تحليل الملف: {exc}"}), 400

        result["filename"] = f.filename


        learn_flag = str(request.form.get("learn", "")).lower() in ("1", "true", "on", "yes")
        if learn_flag:
            if result.get("has_label"):
                try:
                    result["learning"] = registry.learn_in_background(df, f.filename)
                except Exception as exc:
                    result["learning"] = {"queued": False, "learned": False,
                                          "reason": "error", "detail": str(exc)}
            else:
                result["learning"] = {"queued": False, "learned": False, "reason": "no_label"}

        return jsonify(result)

    return app


def run(host="127.0.0.1", port=5050, open_browser=True):
    app = create_app()

    if registry.model is None:
        registry.get_or_train()

    if open_browser:
        url = f"http://{host}:{port}/"
        def _open():
            try:
                webbrowser.open(url)
            except Exception:
                pass
        threading.Timer(1.0, _open).start()

    app.run(host=host, port=port, debug=False, use_reloader=False, threaded=True)
