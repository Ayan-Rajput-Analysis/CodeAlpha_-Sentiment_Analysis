import streamlit as st
import pickle
import re
import os
import nltk
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Alexa Sentiment Analyzer",
    page_icon="🎙️",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main { background: #F4F6FB; }

.header-box {
    background: linear-gradient(135deg, #1B3A6B 0%, #4C72B0 100%);
    border-radius: 16px;
    padding: 32px 28px 24px;
    margin-bottom: 24px;
    color: white;
}
.header-box h1 { font-size: 28px; font-weight: 700; margin: 0 0 4px; }
.header-box p  { font-size: 14px; opacity: 0.75; margin: 0; }

.result-positive {
    background: #EAF7F0;
    border: 1.5px solid #2A9D5C;
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 20px;
}
.result-negative {
    background: #FEF0EE;
    border: 1.5px solid #E05C4B;
    border-radius: 14px;
    padding: 24px 28px;
    margin-top: 20px;
}
.result-label {
    font-size: 22px;
    font-weight: 700;
    margin-bottom: 6px;
}
.result-sub {
    font-size: 14px;
    opacity: 0.7;
    margin-bottom: 16px;
}
.confidence-bar-bg {
    background: rgba(0,0,0,0.08);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    margin-top: 4px;
}
.confidence-bar-fill {
    height: 10px;
    border-radius: 999px;
    transition: width 0.4s ease;
}
.stat-row {
    display: flex;
    gap: 16px;
    margin-top: 12px;
}
.stat-box {
    flex: 1;
    background: rgba(255,255,255,0.6);
    border-radius: 10px;
    padding: 12px 16px;
    text-align: center;
}
.stat-val { font-size: 20px; font-weight: 700; }
.stat-lbl { font-size: 12px; opacity: 0.65; margin-top: 2px; }

.history-item {
    background: white;
    border-radius: 10px;
    border: 1px solid #E5E7EB;
    padding: 12px 16px;
    margin-bottom: 8px;
    font-size: 14px;
}
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 999px;
    font-size: 12px;
    font-weight: 600;
}
.badge-pos { background: #D1FAE5; color: #065F46; }
.badge-neg { background: #FEE2E2; color: #991B1B; }

.sidebar-info {
    background: #EEF2FF;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 13px;
    color: #3730A3;
    margin-bottom: 12px;
}
</style>
""", unsafe_allow_html=True)

# ── Load NLTK & models ─────────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    nltk.download('stopwords', quiet=True)
    sw = set(stopwords.words('english'))
    ps = PorterStemmer()
    return sw, ps

@st.cache_resource
def load_models():
    base = os.path.dirname(os.path.abspath(__file__))

    def load(name):
        return pickle.load(open(os.path.join(base, name), 'rb'))

    cv     = load("countVectorizer.pkl")
    scaler = load("scaler.pkl")
    xgb    = load("model_xgb.pkl")
    rf     = load("model_rf.pkl")
    return cv, scaler, xgb, rf

STOPWORDS, stemmer = load_resources()

try:
    cv, scaler, model_xgb, model_rf = load_models()
    models_ok = True
except Exception as e:
    models_ok = False
    model_error = str(e)

# ── Preprocessing ──────────────────────────────────────────────────────────────
def preprocess(text):
    review = re.sub('[^a-zA-Z]', ' ', text)
    review = review.lower().split()
    review = [stemmer.stem(w) for w in review if w not in STOPWORDS]
    return ' '.join(review)

def predict(text, model):
    cleaned    = preprocess(text)
    vectorized = cv.transform([cleaned]).toarray()
    scaled     = scaler.transform(vectorized)
    pred       = model.predict(scaled)[0]
    proba      = model.predict_proba(scaled)[0]
    return int(pred), proba

# ── Session state ──────────────────────────────────────────────────────────────
if "history" not in st.session_state:
    st.session_state.history = []

# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="header-box">
  <h1>🎙️ Alexa Review Sentiment Analyzer</h1>
  <p>Amazon Alexa Reviews &nbsp;•&nbsp; NLP + XGBoost / Random Forest &nbsp;•&nbsp; CodeAlpha Internship</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Choose Model",
        ["XGBoost", "Random Forest"],
        help="XGBoost usually gives slightly better accuracy."
    )

    st.markdown("---")
    st.markdown("### 📊 About")
    st.markdown("""
<div class="sidebar-info">
<b>Dataset:</b> Amazon Alexa Reviews<br>
<b>Labels:</b> 0 = Negative &nbsp; 1 = Positive<br>
<b>Vectorizer:</b> CountVectorizer (2500 features)<br>
<b>Preprocessing:</b> Stemming + Stopword removal
</div>
""", unsafe_allow_html=True)

    st.markdown("### 💡 Sample Reviews")
    samples = [
        "I love this product, it works perfectly!",
        "Terrible quality, broke after one day.",
        "Amazing sound quality and easy setup.",
        "Does not connect to wifi, very frustrating.",
    ]
    for s in samples:
        if st.button(s[:45] + "...", key=s, use_container_width=True):
            st.session_state["prefill"] = s

# ── Main input ─────────────────────────────────────────────────────────────────
if not models_ok:
    st.error(f"❌ Could not load models. Make sure all `.pkl` files are in the same folder as `app.py`.\n\n`{model_error}`")
    st.stop()

prefill = st.session_state.pop("prefill", "")
review_text = st.text_area(
    "Enter an Alexa product review:",
    value=prefill,
    height=130,
    placeholder="e.g. I love how easy it is to set up and use every day!",
)

col1, col2 = st.columns([1, 3])
with col1:
    analyze_btn = st.button("🔍 Analyze", use_container_width=True, type="primary")
with col2:
    clear_btn = st.button("🗑️ Clear History", use_container_width=True)

if clear_btn:
    st.session_state.history = []

# ── Prediction ─────────────────────────────────────────────────────────────────
if analyze_btn:
    if not review_text.strip():
        st.warning("⚠️ Please enter a review first.")
    else:
        active_model = model_xgb if model_choice == "XGBoost" else model_rf

        with st.spinner("Analyzing..."):
            pred, proba = predict(review_text, active_model)

        neg_conf = round(proba[0] * 100, 1)
        pos_conf = round(proba[1] * 100, 1)
        label    = "Positive 😊" if pred == 1 else "Negative 😞"
        conf     = pos_conf if pred == 1 else neg_conf
        css_cls  = "result-positive" if pred == 1 else "result-negative"
        bar_col  = "#2A9D5C" if pred == 1 else "#E05C4B"
        emoji    = "✅" if pred == 1 else "❌"

        st.markdown(f"""
<div class="{css_cls}">
  <div class="result-label">{emoji} {label}</div>
  <div class="result-sub">Model: {model_choice} &nbsp;•&nbsp; Confidence: {conf}%</div>

  <div style="font-size:13px; margin-bottom:4px; opacity:0.7;">Positive probability</div>
  <div class="confidence-bar-bg">
    <div class="confidence-bar-fill" style="width:{pos_conf}%; background:{bar_col};"></div>
  </div>

  <div class="stat-row">
    <div class="stat-box">
      <div class="stat-val" style="color:#2A9D5C;">{pos_conf}%</div>
      <div class="stat-lbl">Positive</div>
    </div>
    <div class="stat-box">
      <div class="stat-val" style="color:#E05C4B;">{neg_conf}%</div>
      <div class="stat-lbl">Negative</div>
    </div>
    <div class="stat-box">
      <div class="stat-val">{model_choice[:3]}</div>
      <div class="stat-lbl">Model used</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

        # Save to history
        st.session_state.history.insert(0, {
            "text":  review_text[:80] + ("..." if len(review_text) > 80 else ""),
            "label": label,
            "conf":  conf,
            "pred":  pred,
            "model": model_choice,
        })

# ── History ────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("---")
    st.markdown("### 🕓 Recent Predictions")
    for item in st.session_state.history[:8]:
        badge_cls = "badge-pos" if item["pred"] == 1 else "badge-neg"
        badge_txt = "Positive" if item["pred"] == 1 else "Negative"
        st.markdown(f"""
<div class="history-item">
  <span class="badge {badge_cls}">{badge_txt}</span>
  &nbsp; <b>{item['conf']}%</b> &nbsp;
  <span style="opacity:0.5;font-size:12px;">[{item['model']}]</span><br>
  <span style="color:#374151;">{item['text']}</span>
</div>
""", unsafe_allow_html=True)
