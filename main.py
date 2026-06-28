"""
main.py
-------
Entry point for the AI-IDS project.

Running `python main.py` will:
  1. Train (or load a cached) RandomForest detector on data/KDDTrain+.txt
  2. Print a quick console summary (accuracy / alert counts), same spirit
     as the original pipeline script
  3. Launch the web dashboard at http://127.0.0.1:5050/ and open it in
     your default browser automatically — no separate `streamlit run` step.

Use --no-browser to just start the server without opening a tab, or
--retrain to force a fresh training run even if a cached model exists.
"""
import argparse

from src.model_registry import registry
from src.alert_generator import filter_by_severity
from src import web_app


def print_summary():
    demo = registry.demo_alerts()
    alerts = demo["alerts"]
    high = filter_by_severity(alerts, "high")

    print("\n=== ملخص النموذج ===")
    print(f"دقة النموذج (accuracy) على بيانات الاختبار: {registry.metrics['accuracy']}")
    print(f"F1-score: {registry.metrics['f1']} | Precision: {registry.metrics['precision']} | Recall: {registry.metrics['recall']}")
    print(f"\n=== عينة Alerts (n={len(alerts)}) ===")
    print(f"إجمالي alerts: {len(alerts)} | عالية الخطورة: {len(high)}")
    for a in high[:5]:
        print(a)


def main():
    parser = argparse.ArgumentParser(description="AI-IDS — train + launch dashboard")
    parser.add_argument("--retrain", action="store_true", help="force retraining even if a cached model exists")
    parser.add_argument("--no-browser", action="store_true", help="start the server without opening a browser tab")
    parser.add_argument("--port", type=int, default=5050)
    parser.add_argument("--host", default="127.0.0.1", help="use 0.0.0.0 when running inside Docker")
    args = parser.parse_args()

    print("=== 1) تحميل/تدريب النموذج ===")
    registry.get_or_train(force=args.retrain)

    print_summary()

    print(f"\n=== 2) تشغيل لوحة التحكم على http://{args.host}:{args.port}/ ===")
    web_app.run(host=args.host, port=args.port, open_browser=not args.no_browser)


if __name__ == "__main__":
    main()
