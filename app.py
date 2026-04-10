from flask import Flask, render_template, request, redirect, url_for, session
import numpy as np
import joblib
import os
import json
from datetime import datetime

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', os.urandom(24))

LANGUAGE_MAP = {
    "en": {
        "code": "en",
        "name": "English",
        "page_title": "AYP - Smart Crop Yield Predictor",
        "subtitle": "AI-Powered Crop Yield Prediction",
        "description": "Helping farmers make better decisions with precision agriculture",
        "header": "Enter Field & Soil Details",
        "fertilizer": "Fertilizer (kg/hectare)",
        "temperature": "Temperature (°C)",
        "nitrogen": "Nitrogen (N)",
        "phosphorus": "Phosphorus (P)",
        "potassium": "Potassium (K)",
        "predict_button": "Predict Crop Yield",
        "footer": "Our AI model is trained on real agricultural data for high accuracy",
        "language_label": "Language",
        "help_text": "Select a language and enter your field details.",
        "info1": "Field data driven predictions",
        "info2": "Trusted by smart farms",
        "info3": "Fast and reliable results",
        "result_title": "Predicted Yield",
        "result_subtitle": "kg per hectare",
        "recommendation": "This yield looks promising with current conditions!",
        "recommendation2": "Maintain balanced nutrition and irrigation for better results.",
        "model_prefix": "Selected model:",
        "error_message": "Please enter valid numeric values for all fields."
    },
    "hi": {
        "code": "hi",
        "name": "हिंदी",
        "page_title": "एग्रीप्रिडिक्ट - स्मार्ट फसल भविष्यवाणी",
        "subtitle": "AI संचालित फसल उत्पादन पूर्वानुमान",
        "description": "सटीक कृषि के साथ किसानों को बेहतर निर्णय लेने में मदद",
        "header": "क्षेत्र और मिट्टी का विवरण दर्ज करें",
        "fertilizer": "उर्वरक (किग्रा/हेक्टेयर)",
        "temperature": "तापमान (°C)",
        "nitrogen": "नाइट्रोजन (N)",
        "phosphorus": "फॉस्फोरस (P)",
        "potassium": "पोटैशियम (K)",
        "predict_button": "फसल उत्पादन अनुमान",
        "footer": "हमारा AI मॉडल वास्तविक कृषि डेटा पर प्रशिक्षित है",
        "language_label": "भाषा",
        "help_text": "भाषा चुनें और विवरण दर्ज करें।",
        "info1": "फ़ील्ड डेटा आधारित भविष्यवाणियाँ",
        "info2": "स्मार्ट फार्म के लिए भरोसेमंद",
        "info3": "तेज़ और सटीक परिणाम",
        "result_title": "अनुमानित उत्पादन",
        "result_subtitle": "किग्रा प्रति हेक्टेयर",
        "recommendation": "वर्तमान परिस्थितियों में यह उत्पादन बहुत अच्छा दिखता है!",
        "recommendation2": "बेहतर परिणामों के लिए संतुलित पोषण और सिंचाई बनाए रखें।",
        "model_prefix": "चयनित मॉडल:",
        "error_message": "कृपया सभी फ़ील्डों के लिए मान्य संख्यात्मक मान दर्ज करें।"
    },
    "es": {
        "code": "es",
        "name": "Español",
        "page_title": "AYP - Predicción Inteligente de Cultivos",
        "subtitle": "Predicción de rendimiento de cultivos con IA",
        "description": "Ayudando a los agricultores a tomar decisiones más inteligentes",
        "header": "Ingrese detalles del campo y del suelo",
        "fertilizer": "Fertilizante (kg/ha)",
        "temperature": "Temperatura (°C)",
        "nitrogen": "Nitrógeno (N)",
        "phosphorus": "Fósforo (P)",
        "potassium": "Potasio (K)",
        "predict_button": "Predecir rendimiento",
        "footer": "Nuestro modelo de IA está entrenado con datos agrícolas reales",
        "language_label": "Idioma",
        "help_text": "Seleccione un idioma y complete los datos del campo.",
        "info1": "Predicciones basadas en datos de campo",
        "info2": "Confiable para granjas modernas",
        "info3": "Resultados rápidos y precisos",
        "result_title": "Rendimiento Predicho",
        "result_subtitle": "kg por hectárea",
        "recommendation": "Este rendimiento se ve prometedor con las condiciones actuales.",
        "recommendation2": "Mantenga la nutrición y el riego equilibrados para mejores resultados.",
        "model_prefix": "Modelo seleccionado:",
        "error_message": "Por favor ingrese valores numéricos válidos en todos los campos."
    },
    "pt": {
        "code": "pt",
        "name": "Português",
        "page_title": "AYP - Previsão Inteligente de Safras",
        "subtitle": "Previsão de rendimento de culturas com IA",
        "description": "Ajudando agricultores a tomar decisões melhores",
        "header": "Insira os detalhes do campo e do solo",
        "fertilizer": "Fertilizante (kg/ha)",
        "temperature": "Temperatura (°C)",
        "nitrogen": "Nitrogênio (N)",
        "phosphorus": "Fósforo (P)",
        "potassium": "Potássio (K)",
        "predict_button": "Prever rendimento",
        "footer": "Nosso modelo de IA é treinado com dados agrícolas reais",
        "language_label": "Idioma",
        "help_text": "Selecione um idioma e informe os dados do campo.",
        "info1": "Previsões baseadas em dados de campo",
        "info2": "Confiável para fazendas inteligentes",
        "info3": "Resultados rápidos e precisos",
        "result_title": "Rendimento Previsto",
        "result_subtitle": "kg por hectare",
        "recommendation": "Este rendimento parece promissor com as condições atuais.",
        "recommendation2": "Mantenha nutrição equilibrada e irrigação constante.",
        "model_prefix": "Modelo selecionado:",
        "error_message": "Por favor, insira valores numéricos válidos para todos os campos."
    },
    "fr": {
        "code": "fr",
        "name": "Français",
        "page_title": "AYP - Prédiction Intelligente des Cultures",
        "subtitle": "Prédiction de rendement des cultures avec IA",
        "description": "Aidez les agriculteurs à prendre de meilleures décisions",
        "header": "Entrez les détails du champ et du sol",
        "fertilizer": "Engrais (kg/ha)",
        "temperature": "Température (°C)",
        "nitrogen": "Azote (N)",
        "phosphorus": "Phosphore (P)",
        "potassium": "Potassium (K)",
        "predict_button": "Prédire le rendement",
        "footer": "Notre modèle IA est entraîné sur des données agricoles réelles",
        "language_label": "Langue",
        "help_text": "Sélectionnez une langue et saisissez les détails du champ.",
        "info1": "Prédictions basées sur les données du champ",
        "info2": "Fiable pour les fermes intelligentes",
        "info3": "Résultats rapides et précis",
        "result_title": "Rendement Prédit",
        "result_subtitle": "kg par hectare",
        "recommendation": "Ce rendement semble prometteur dans les conditions actuelles.",
        "recommendation2": "Maintenez une nutrition équilibrée et une irrigation régulière.",
        "model_prefix": "Modèle sélectionné:",
        "error_message": "Veuillez saisir des valeurs numériques valides pour tous les champs."
    },
    "ar": {
        "code": "ar",
        "name": "العربية",
        "page_title": "AYP - التنبؤ الذكي بالمحاصيل",
        "subtitle": "تنبؤ إنتاجية المحاصيل باستخدام الذكاء الاصطناعي",
        "description": "مساعدة المزارعين على اتخاذ قرارات أفضل",
        "header": "أدخل تفاصيل الحقل والتربة",
        "fertilizer": "الأسمدة (كجم/هكتار)",
        "temperature": "درجة الحرارة (°م)",
        "nitrogen": "النيتروجين (N)",
        "phosphorus": "الفوسفور (P)",
        "potassium": "البوتاسيوم (K)",
        "predict_button": "تنبؤ محصول",
        "footer": "نموذج الذكاء الاصطناعي لدينا مدرب على بيانات زراعية حقيقية",
        "language_label": "اللغة",
        "help_text": "اختر لغة وأدخل تفاصيل الحقل.",
        "info1": "تنبؤات قائمة على بيانات الحقل",
        "info2": "موثوق للمزارع الذكية",
        "info3": "نتائج سريعة ودقيقة",
        "result_title": "المحصول المتوقع",
        "result_subtitle": "كجم لكل هكتار",
        "recommendation": "يبدو هذا المحصول واعدًا في الظروف الحالية.",
        "recommendation2": "حافظ على تغذية متوازنة وري منتظم للحصول على نتائج أفضل.",
        "model_prefix": "النموذج المحدد:",
        "error_message": "يرجى إدخال قيم رقمية صالحة لجميع الحقول.",
        "dir": "rtl"
    },
    "zh": {
        "code": "zh",
        "name": "中文",
        "page_title": "AYP - 智能作物产量预测",
        "subtitle": "基于 AI 的作物产量预测",
        "description": "帮助农民做出更好的决策",
        "header": "输入田间和土壤信息",
        "fertilizer": "肥料 (kg/公顷)",
        "temperature": "温度 (°C)",
        "nitrogen": "氮 (N)",
        "phosphorus": "磷 (P)",
        "potassium": "钾 (K)",
        "predict_button": "预测作物产量",
        "footer": "我们的 AI 模型基于真实农业数据训练",
        "language_label": "语言",
        "help_text": "请选择语言并输入田间数据。",
        "info1": "基于田间数据的预测",
        "info2": "现代农场信赖",
        "info3": "快速且准确的结果",
        "result_title": "预测产量",
        "result_subtitle": "千克/公顷",
        "recommendation": "当前条件下，该产量看起来很有希望。",
        "recommendation2": "保持营养均衡并定期灌溉以获得更好结果。",
        "model_prefix": "所选模型：",
        "error_message": "请输入所有字段的有效数字值。"
    },
    "bn": {
        "code": "bn",
        "name": "বাংলা",
        "page_title": "AYP - স্মার্ট ফসল পূর্বাভাস",
        "subtitle": "এআই-ভিত্তিক ফসল উৎপাদন পূর্বাভাস",
        "description": "কৃষকদের আরও ভাল সিদ্ধান্ত নিতে সহায়তা করে",
        "header": "ক্ষেত্র এবং মাটির বিবরণ লিখুন",
        "fertilizer": "সার (কেজি/হেক্টর)",
        "temperature": "তাপমাত্রা (°C)",
        "nitrogen": "নাইট্রোজেন (N)",
        "phosphorus": "ফসফরাস (P)",
        "potassium": "পটাসিয়াম (K)",
        "predict_button": "ফসল পূর্বানুমান করুন",
        "footer": "আমাদের AI মডেল বাস্তব কৃষি ডেটায় প্রশিক্ষিত",
        "language_label": "ভাষা",
        "help_text": "একটি ভাষা নির্বাচন করুন এবং ক্ষেত্রের বিবরণ দিন।",
        "info1": "ক্ষেত্র-ভিত্তিক ডেটা ভবিষ্যদ্বাণী",
        "info2": "স্মার্ট খামারের জন্য নির্ভরযোগ্য",
        "info3": "দ্রুত এবং সঠিক ফলাফল",
        "result_title": "পূর্বানুমানিত উৎপাদন",
        "result_subtitle": "কেজি প্রতি হেক্টর",
        "recommendation": "বর্তমান পরিস্থিতিতে এই উৎপাদন প্রতিশ্রুতিশীল দেখাচ্ছে।",
        "recommendation2": "ভাল ফলাফলের জন্য সমতুল্য পুষ্টি এবং সেচ বজায় রাখুন।",
        "model_prefix": "নির্বাচিত মডেল:",
        "error_message": "অনুগ্রহ করে সমস্ত ক্ষেত্রের জন্য বৈধ সংখ্যাসূচক মান লিখুন।"
    },
    "kn": {
        "code": "kn",
        "name": "ಕನ್ನಡ",
        "page_title": "AYP - ಎಐ ಬೆಳೆ ಉತ್ಪಾದನೆ ಭವಿಷ್ಯವಾಣಿ",
        "subtitle": "AI ಆಧಾರಿತ ಬೆಳೆ ಉತ್ಪಾದನೆ ಭವಿಷ್ಯವಾಣಿ",
        "description": "ಕೃಷಕರಿಗೆ ಉತ್ತಮ ನಿರ್ಧಾರಮಾಡಲು ಸಹಾಯ",
        "header": "ಕ್ಷೇತ್ರ ಮತ್ತು ಮಣ್ಣಿನ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ",
        "fertilizer": "ಸಾರ (ಕಿಗ್ರಾ/ಹೆಕ್ಟೇರ್)",
        "temperature": "ತಾಪಮಾನ (°C)",
        "nitrogen": "ನೈಟ್ರೋಜನ್ (N)",
        "phosphorus": "ಫಾಸ್ಫೋರಸ್ (P)",
        "potassium": "ಪೊಟ್ಯಾಸಿಯಂ (K)",
        "predict_button": "ಬೆಳೆ ಉತ್ಪನ್ನವನ್ನು ಭವಿಷ್ಯವಾಣಿ ಮಾಡಿ",
        "footer": "ನಮ್ಮ AI ಮಾದರಿ ವಾಸ್ತವ ಕೃಷಿ ಡೇಟಾದೊಂದಿಗೆ ತರಬೇತುಗೊಂಡಿದೆ",
        "language_label": "ಭಾಷೆ",
        "help_text": "ಭಾಷೆಯನ್ನು ಆಯ್ಕೆಮಾಡಿ ಮತ್ತು ಕ್ಷೇತ್ರದ ವಿವರಗಳನ್ನು ನಮೂದಿಸಿ.",
        "info1": "ಕ್ಷೇತ್ರದ ಡೇಟಾ ಆಧಾರಿತ ಭವಿಷ್ಯಗಳು",
        "info2": "ಸ್ಮಾರ್ಟ್ ಫಾರ್ಮ್ಗಳಿಗೆ ವಿಶ್ವಾಸಾರ್ಹ",
        "info3": "ವೇಗವಾದ ಮತ್ತು ನಿಖರ ಫಲಿತಾಂಶಗಳು",
        "result_title": "ಭವಿಷ್ಯವಾಣಿ ಫಲವು",
        "result_subtitle": "ಕಿಗ್ರಾ ಪ್ರತಿ ಹೆಕ್ಟೇರ್",
        "recommendation": "ಈ ಉತ್ಪಾದನೆ ಪ್ರಸ್ತುತ ಪರಿಸ್ಥಿತಿಗೆ ಅನుకೂಲವಾಗಿದೆ!",
        "recommendation2": "ಉತ್ತಮ ಫಲಿತಾಂಶಕ್ಕಾಗಿ ಸಮತೋಲನ ಪೋಷಣೆ ಮತ್ತು ನಿರಂತರ ಜಲಸिंಚನ ನಿರ್ವಹಿಸಿ.",
        "model_prefix": "ಆಯ್ದ ಮಾದರಿ:",
        "error_message": "ದಯವಿಟ್ಟು ಎಲ್ಲಾ ಕ್ಷೇತ್ರಗಳಿಗೆ ಮಾನ್ಯ ಸಂಖ್ಯಾ ಮೌಲ್ಯಗಳನ್ನು ನಮೂದಿಸಿ."
    }
}

DEFAULT_LANG = "en"
model = None
pipeline_mode = False
model_name = "AYP Forecast"
scaler = None
poly = None
prediction_history = []
HISTORY_FILE = os.path.join('data', 'prediction_history.json')


def normalize_lang(lang):
    return lang if lang in LANGUAGE_MAP else DEFAULT_LANG


def save_prediction_history():
    try:
        os.makedirs(os.path.dirname(HISTORY_FILE), exist_ok=True)
        with open(HISTORY_FILE, 'w', encoding='utf-8') as f:
            json.dump(prediction_history, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print('Error saving prediction history:', e)


def load_prediction_history():
    global prediction_history
    try:
        if os.path.exists(HISTORY_FILE):
            with open(HISTORY_FILE, 'r', encoding='utf-8') as f:
                prediction_history = json.load(f)
    except Exception as e:
        print('Error loading prediction history:', e)
        prediction_history = []


load_prediction_history()


# ====================== LOAD MODEL & SCALER ======================
try:
    if os.path.exists('models/best_pipeline.pkl'):
        model = joblib.load('models/best_pipeline.pkl')
        pipeline_mode = True
        model_name = type(model.named_steps['model']).__name__ if hasattr(model, 'named_steps') else type(model).__name__
        print(f"Pipeline model loaded: {model_name}")
    elif os.path.exists('models/best_model.pkl'):
        model = joblib.load('models/best_model.pkl')
        scaler = joblib.load('models/scaler.pkl')
        if os.path.exists('models/poly_features.pkl'):
            poly = joblib.load('models/poly_features.pkl')
            print("Polynomial features loaded")
        else:
            print("No polynomial features (using original 5 features)")
        model_name = type(model).__name__
        print(f"Model loaded successfully: {model_name}")
        print(f"Scaler expects {scaler.n_features_in_} features")
    else:
        raise FileNotFoundError('No model file found. Train the model and create models/best_pipeline.pkl or models/best_model.pkl')
except Exception as e:
    print("Error loading model:", e)
    model = None

# ====================== ROOT REDIRECT ======================
@app.route('/')
def root():
    return redirect(url_for('login'))

# ====================== HOME PAGE ======================
@app.route('/home')
def home():
    lang = normalize_lang(request.args.get('lang', DEFAULT_LANG))
    user = request.args.get('user', '')
    show_welcome = request.args.get('welcome') == '1'
    labels = LANGUAGE_MAP[lang]
    return render_template('index.html', lang=lang, labels=labels, languages=LANGUAGE_MAP, user=user, show_welcome=show_welcome)

# ====================== ABOUT PAGE ======================
@app.route('/about')
def about():
    lang = normalize_lang(request.args.get('lang', DEFAULT_LANG))
    user = request.args.get('user', '')
    labels = LANGUAGE_MAP[lang]
    return render_template('about.html', lang=lang, labels=labels, languages=LANGUAGE_MAP, user=user)

# ====================== DASHBOARD PAGE ======================
@app.route('/dashboard')
def dashboard():
    lang = normalize_lang(request.args.get('lang', DEFAULT_LANG))
    labels = LANGUAGE_MAP[lang]
    total = len(prediction_history)
    avg_yield = round(sum(item['prediction'] for item in prediction_history) / total, 2) if total else 0
    latest = prediction_history[0] if total else None
    user = request.args.get('user', '')
    return render_template(
        'dashboard.html',
        lang=lang,
        labels=labels,
        languages=LANGUAGE_MAP,
        history=prediction_history,
        total=total,
        avg_yield=avg_yield,
        latest=latest,
        user=user,
    )

# ====================== CLEAR DASHBOARD HISTORY ======================
@app.route('/dashboard/clear', methods=['POST'])
def clear_dashboard():
    global prediction_history
    prediction_history = []
    save_prediction_history()
    lang = normalize_lang(request.form.get('lang', DEFAULT_LANG))
    user = request.form.get('user', '')
    return redirect(url_for('dashboard', lang=lang, user=user))

# ====================== CONTACT FORM ======================
@app.route('/contact', methods=['POST'])
def contact():
    lang = normalize_lang(request.args.get('lang', request.form.get('lang', DEFAULT_LANG)))
    labels = LANGUAGE_MAP[lang]
    user = request.form.get('user', '').strip()
    name = request.form.get('name', '').strip()
    email = request.form.get('email', '').strip()
    message = request.form.get('message', '').strip()

    print('📩 Contact request received:', {'user': user, 'name': name, 'email': email, 'message': message})

    return render_template('contact_thankyou.html', lang=lang, labels=labels, name=name, user=user)

# ====================== LOGIN PAGE ======================
@app.route('/login', methods=['GET', 'POST'])
def login():
    lang = normalize_lang(request.args.get('lang', request.form.get('lang', DEFAULT_LANG)))
    labels = LANGUAGE_MAP[lang]
    message = None

    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        if username:
            return redirect(url_for('home', lang=lang, user=username, welcome='1'))
        message = labels.get('error_message', 'Please enter a username.')

    return render_template('login.html', lang=lang, labels=labels, languages=LANGUAGE_MAP, message=message)

# ====================== PREDICT PAGE ======================
@app.route('/predict', methods=['POST'])
def predict():
    lang = normalize_lang(request.form.get('lang', DEFAULT_LANG))
    labels = LANGUAGE_MAP[lang]

    if model is None:
        return f"<h3 style='color:red'>Model not loaded. Please train the model first.</h3><br><a href='/home?lang={lang}'>Go Back</a>"

    try:
        field_names = ["Fertilizer", "temp", "N", "P", "K"]
        input_values = [float(request.form.get(field, 0)) for field in field_names]
        user = request.form.get('user', '').strip()
        print("📥 Inputs received from form:", input_values)

        final_features = np.array([input_values])

        if pipeline_mode:
            prediction = model.predict(final_features)[0]
        else:
            if poly is not None:
                final_features = poly.transform(final_features)
                print("📊 Applied Polynomial Features")
            final_features_scaled = scaler.transform(final_features)
            prediction = model.predict(final_features_scaled)[0]

        prediction_value = round(prediction, 2)
        prediction_history.insert(0, {
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M'),
            'inputs': {
                'Fertilizer': input_values[0],
                'Temperature': input_values[1],
                'Nitrogen': input_values[2],
                'Phosphorus': input_values[3],
                'Potassium': input_values[4],
            },
            'prediction': prediction_value,
            'user': user,
        })
        if len(prediction_history) > 12:
            prediction_history.pop()
        save_prediction_history()

        session['last_prediction'] = prediction_value
        session['last_model_name'] = model_name
        session['last_user'] = user
        session['last_lang'] = lang
        session['last_inputs'] = {
            'Fertilizer': input_values[0],
            'Temperature': input_values[1],
            'Nitrogen': input_values[2],
            'Phosphorus': input_values[3],
            'Potassium': input_values[4],
        }

        print(f"🎯 Predicted Yield: {prediction_value:.2f} kg/hectare")

        return redirect(url_for('result', lang=lang, user=user))

    except Exception as e:
        error_msg = str(e)
        print("❌ Prediction Error:", error_msg)
        return f"""
        <h3 style='color:red'>{labels['error_message']}</h3>
        <p>{error_msg}</p>
        <br>
        <a href='/home?lang={lang}' class='btn btn-primary'>Go Back to Form</a>
        """

# ====================== RESULT PAGE ======================
@app.route('/result')
def result():
    lang = normalize_lang(request.args.get('lang', session.get('last_lang', DEFAULT_LANG)))
    user = request.args.get('user', session.get('last_user', ''))
    prediction = session.get('last_prediction')
    model_name_session = session.get('last_model_name', model_name)
    labels = LANGUAGE_MAP[lang]

    if prediction is None:
        return redirect(url_for('home', lang=lang, user=user))

    last_inputs = session.get('last_inputs', {})

    return render_template(
        'result.html',
        prediction=prediction,
        labels=labels,
        lang=lang,
        model_name=model_name_session,
        user=user,
        inputs=last_inputs,
    )

if __name__ == "__main__":
    app.run(
        host='0.0.0.0',
        port=int(os.environ.get('PORT', 5000)),
        debug=False,
    )