# 🛡️ AI-IDS Project

AI-powered Intrusion Detection System with explainability and SOC automation,
built as part of a 60-day AI Security learning plan.

## ما الجديد

- **لوحة تحكم HTML/JS** بدل Streamlit — صفحة واحدة سريعة، بدون أمر تشغيل منفصل.
- **تشتغل تلقائياً عند تشغيل `main.py`**: تدريب/تحميل النموذج ثم فتح المتصفح على
  `http://127.0.0.1:5050/` تلقائياً.
- **رفع ملف بياناتك الخاص** من نفس الصفحة وتحليله بنفس النموذج والمعايرة
  المستخدمة على KDDTrain+.txt. يدعم:
  - صيغ **NSL-KDD وKDD Cup '99** (41/42/43 عمود، مع تطبيع labels التي تنتهي
    بنقطة مثل `normal.`).
  - الملفات **النصية والمضغوطة `.zip`** (يفك الضغط تلقائياً)، حتى الملفات
    الكبيرة (مئات آلاف السجلات).
  - يتجاهل سطور الـheader/التعليقات والفراغات، ويرفض الملفات غير النصية
    (صور/أرشيف تالف) أو غير المتوافقة برسالة واضحة بدل أن تتعطّل الصفحة.
- **التعلّم في الخلفية (Background Incremental Learning)**: كل ملف مرفوع فيه
  `label` يُضاف لمخزن تدريب متنامٍ (`data/learned/`) ويُعاد تدريب النموذج على
  **اتحاد كل البيانات** — لكن إعادة التدريب تجري **في خيط منفصل** فلا تجمّد
  الصفحة. التحليل يرجع فوراً، والصفحة تعرض شريط تقدّم وتُحدّث الأداء (الدقة
  قبل ← بعد على معيار ثابت) لحظة انتهاء التدريب. الملفات بدون `label` تُحلَّل
  فقط.
- **تصحيح خلل في التشفير**: النموذج الأصلي كان يعيد `fit_transform()` على كل
  dataset فيعطي أكواد مختلفة لنفس الفئة — وهذا يكسر النموذج بصمت. الحل
  `UnseenLabelEncoder` يُدرَّب مرة ويُحفظ ويُعاد استخدامه، ويتعامل بأمان مع
  أي فئة غير معروفة.
- **تدريب لمرة واحدة + كاش**: النموذج/الـscaler/الـencoders تُحفظ في `models/`،
  فلا يُعاد تدريب 126 ألف سجل كل مرة تُفتح الصفحة.

> **عن خطأ Chrome `RESULT_CODE_KILLED_BAD_MESSAGE`**: كان سببه أن إعادة
> تدريب النموذج كانت تجري داخل نفس طلب الرفع فتُجمّد الخادم ~20 ثانية، فيقتل
> Chrome صفحة العرض. تم حلّه جذرياً: التحليل يرجع فوراً، وإعادة التدريب تجري
> **في الخلفية**، والخادم **متعدد الخيوط (threaded)**، وكل ردود الـAPI صغيرة
> (بضعة كيلوبايت) مهما كبر الملف، وسجل التنبيهات يُبنى عبر `DocumentFragment`
> بسقف صفوف ثابت. لو ظهر الخطأ مع متصفحك لسبب خارجي: اضغط Reload أو جرّب
> متصفحاً آخر.

## Architecture

```
ai_ids_project/
├── src/
│   ├── data_pipeline.py        # load + clean + flexible/robust NSL-KDD parsing + UnseenLabelEncoder
│   ├── feature_extractor.py    # scaling + reshaping, with save()/load() persistence
│   ├── ensemble_model.py       # RandomForest (+ optional LSTM) ensemble detector
│   ├── model_registry.py       # train once, cache, INCREMENTAL LEARNING + fixed benchmark
│   ├── web_app.py              # Flask app: HTML dashboard + JSON API (analyze / learn)
│   ├── alert_generator.py      # turns model output into structured alerts
│   ├── xai_explainer.py        # SHAP/LIME explainability for predictions
│   ├── threat_nlp_analyzer.py  # NER + regex IOC extraction from threat reports
│   ├── red_blue_ai_demo.py     # red/blue team AI concepts (alert triage, UBA)
│   ├── auto_incident_response.py # SOAR-style automated IR playbook
│   └── dashboard_streamlit_legacy.py  # the old Streamlit dashboard (kept for reference)
├── templates/index.html        # dashboard page
├── static/{style.css,app.js}   # dashboard styling + logic
├── models/                     # cached model/scaler/encoders + meta.json (auto-created)
├── data/
│   ├── KDDTrain+.txt           # base dataset (included)
│   └── learned/                # accumulated labeled uploads the model learned from
├── tests/                      # unit tests (pytest)
├── main.py                     # train/load + auto-launch dashboard in the browser
└── requirements.txt
```

## Quick start

```bash
pip install -r requirements.txt
python main.py                  # trains/loads the model AND opens the dashboard automatically
pytest tests/ -v                # run tests
```

Flags:

```bash
python main.py --retrain        # force a fresh training run
python main.py --no-browser     # start the server without opening a tab
python main.py --port 8080      # use a different port
```

Once it's running, open the **"حلّل ملف بياناتك"** panel on the page and
upload any NSL-KDD-formatted file (your own capture/export, or
`KDDTest+.txt`). It's scored with the exact same model and preprocessing
used on the training data, and if the file has a `label` column you'll also
get accuracy/precision/recall/F1 for that file.

**Teaching the model.** Leave the **"علّم النموذج من هذا الملف"** checkbox
ticked when you upload a *labeled* file and the system will add it to the
training store and retrain on everything seen so far — the panel then shows
the benchmark accuracy *before → after* so you can tell whether the new data
helped. To wipe what it has learned and go back to the base model:

```bash
rm -rf data/learned models/*.joblib models/meta.json
python main.py --retrain
```

## Project Proposal

**Problem.** Traditional signature-based intrusion detection systems struggle
to catch novel or evolving network attacks, and produce high volumes of
low-context alerts that overwhelm security analysts.

**Goal.** Build an AI-IDS that combines classical ML (Random Forest) and
sequence-aware deep learning (LSTM, optional) into an ensemble detector,
evaluated on the NSL-KDD benchmark dataset. Beyond raw detection, the system
prioritizes explainability (SHAP/LIME) so analysts understand *why* a record
was flagged, and integrates automation (SOAR-style playbook) to reduce manual
triage time.

**Architecture.** The pipeline is modular: `data_pipeline` handles ingestion,
cleaning and consistent categorical encoding; `feature_extractor` prepares
numeric/sequential inputs and persists its scaler; `model_registry` owns
training/caching/inference so the rest of the app never has to think about
when or how the model was fit; `ensemble_model` performs detection;
`alert_generator` formats results; `xai_explainer` adds interpretability; and
`web_app` exposes everything through a single-page HTML dashboard. An
`auto_incident_response` module demonstrates how high-confidence alerts can
trigger automated containment actions (e.g., IP blocking) while
lower-confidence ones are routed to a human analyst via a ticketing workflow.

**Tech stack.** Python, scikit-learn, Flask, Chart.js, TensorFlow/Keras
(optional), SHAP, LIME, pytest, Docker.

**Expected outcome.** A working, tested, explainable IDS prototype with a
self-contained web dashboard that anyone can run with one command — suitable
as a portfolio project demonstrating applied AI security + data engineering
skills.

## Docker

```bash
docker compose build
docker compose up        # dashboard at http://localhost:5050
```

ضع ملفات NSL-KDD إضافية داخل `data/` إن رغبت — مجلد `data/` و `models/`
متصلان بالكونتينر عبر volumes، فأي ملف تضعه محلياً يظهر تلقائياً داخل
الكونتينر بدون إعادة بناء الصورة.
