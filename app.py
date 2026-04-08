"""
=============================================================================
ระบบปัญญาประดิษฐ์สำหรับประเมินความเสี่ยงและทำนายการฟื้นตัวหลังผ่าตัดกระดูกสันหลัง
Development and Evaluation of an AI-Based Risk Assessment System for
Predicting Postoperative Recovery Quality in Spinal Surgery Patients
=============================================================================
วิสัญญีพยาบาล โรงพยาบาลแพร่ | 
=============================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve
)
import xgboost as xgb
from imblearn.over_sampling import SMOTE
import shap

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="AI ประเมินการฟื้นตัวหลังผ่าตัดกระดูกสันหลัง",
    page_icon="🦴",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main-title {
        background: linear-gradient(135deg, #1e3a5f 0%, #2e6da4 100%);
        color: white;
        padding: 24px 32px;
        border-radius: 12px;
        margin-bottom: 24px;
        text-align: center;
    }
    .main-title h1 { color: white; font-size: 1.6rem; margin: 0; line-height: 1.5; }
    .main-title p { color: #cce0ff; margin: 6px 0 0 0; font-size: 0.95rem; }

    .metric-card {
        background: white;
        border: 1px solid #e0e8f0;
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.06);
    }
    .metric-card .val { font-size: 2rem; font-weight: 700; }
    .metric-card .lbl { font-size: 0.85rem; color: #666; margin-top: 4px; }

    .risk-high   { background: #fff0f0; border-left: 5px solid #e53935; border-radius: 8px; padding: 16px 20px; }
    .risk-mid    { background: #fffde7; border-left: 5px solid #f9a825; border-radius: 8px; padding: 16px 20px; }
    .risk-low    { background: #f1f8f1; border-left: 5px solid #43a047; border-radius: 8px; padding: 16px 20px; }

    .section-header {
        font-size: 1.15rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #2e6da4;
        padding-bottom: 6px; margin: 20px 0 14px 0;
    }
    .badge-blue  { background:#e3f2fd; color:#1565c0; border-radius:20px; padding:4px 12px; font-size:0.82rem; font-weight:600; }
    .badge-green { background:#e8f5e9; color:#2e7d32; border-radius:20px; padding:4px 12px; font-size:0.82rem; font-weight:600; }

    div[data-testid="stSidebarNav"] { padding-top: 10px; }
    .stButton>button { border-radius: 8px; font-weight: 600; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# SYNTHETIC DATA + MODEL TRAINING (cached)
# ─────────────────────────────────────────────
FEATURE_COLS = [
    'Age', 'Sex', 'BMI', 'Frailty_TFRAIL',
    'Nutrition_NAF', 'ASA_Class', 'Comorbidities',
    'Pain_NRS_Preop', 'Surgery_Type'
]
FEATURE_LABELS_TH = {
    'Age':            'อายุ (ปี)',
    'Sex':            'เพศ',
    'BMI':            'ดัชนีมวลกาย (BMI)',
    'Frailty_TFRAIL': 'ภาวะความเปราะบาง (T-FRAIL)',
    'Nutrition_NAF':  'ภาวะโภชนาการ (NAF)',
    'ASA_Class':      'ภาวะสุขภาพ (ASA)',
    'Comorbidities':  'จำนวนโรคร่วม',
    'Pain_NRS_Preop': 'ความปวดก่อนผ่าตัด (NRS)',
    'Surgery_Type':   'ชนิดการผ่าตัด'
}

@st.cache_resource(show_spinner="กำลังเตรียมโมเดล ML กรุณารอสักครู่...")
def build_models():
    """Generate synthetic data, preprocess, train 4 models, return artifacts."""
    np.random.seed(42)
    n = 150

    age           = np.random.normal(56, 14, n).clip(20, 85).astype(int)
    sex           = np.random.choice([0, 1], n, p=[0.48, 0.52])
    bmi           = np.round(np.random.normal(24.5, 4.2, n).clip(16, 42), 1)
    frailty       = np.random.choice([0,1,2,3,4,5], n, p=[0.28,0.26,0.22,0.14,0.07,0.03])
    nutrition     = np.random.choice([1,2,3], n, p=[0.52,0.30,0.18])
    asa           = np.random.choice([1,2,3,4], n, p=[0.12,0.42,0.36,0.10])
    comorbidity   = np.random.choice([0,1,2,3,4], n, p=[0.25,0.30,0.25,0.15,0.05])
    pain_preop    = np.random.randint(0, 11, n)
    surgery_type  = np.random.choice([0,1,2,3], n, p=[0.35,0.30,0.25,0.10])

    risk = (0.30*(frailty/5) + 0.22*((asa-1)/3) + 0.18*((nutrition-1)/2) +
            0.12*(pain_preop/10) + 0.10*(comorbidity/4) +
            0.05*((age-20)/65) + 0.03*(surgery_type/3))
    qor40 = (195 - 65*risk + np.random.normal(0, 12, n)).clip(40, 200).round().astype(int)
    poor  = (qor40 < 170).astype(int)

    df = pd.DataFrame({
        'Age': age, 'Sex': sex, 'BMI': bmi,
        'Frailty_TFRAIL': frailty, 'Nutrition_NAF': nutrition,
        'ASA_Class': asa, 'Comorbidities': comorbidity,
        'Pain_NRS_Preop': pain_preop, 'Surgery_Type': surgery_type,
        'QoR40_Score': qor40, 'Poor_Recovery': poor
    })

    X = df[FEATURE_COLS]
    y = df['Poor_Recovery']

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Impute + SMOTE
    imp = SimpleImputer(strategy='mean')
    X_tr_i = pd.DataFrame(imp.fit_transform(X_tr), columns=FEATURE_COLS)
    X_te_i = pd.DataFrame(imp.transform(X_te),     columns=FEATURE_COLS)

    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_tr_i, y_tr)

    scaler = StandardScaler()
    X_res_s = scaler.fit_transform(X_res)
    X_te_s  = scaler.transform(X_te_i)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # ── Logistic Regression ──
    lr = LogisticRegression(C=1, max_iter=1000, random_state=42)
    lr.fit(X_res_s, y_res)

    # ── Random Forest ──
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)
    rf.fit(X_res, y_res)

    # ── XGBoost ──
    xg = xgb.XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.1,
                            eval_metric='logloss', random_state=42, verbosity=0)
    xg.fit(X_res, y_res)

    # ── SVM ──
    sv = SVC(kernel='rbf', C=10, gamma='scale', probability=True, random_state=42)
    sv.fit(X_res_s, y_res)

    # ── CV scores ──
    def cv_score(mdl, Xd):
        sc = cross_validate(mdl, Xd, y_res, cv=cv,
                            scoring=['accuracy','recall','precision','f1','roc_auc'])
        return {k: (v.mean(), v.std()) for k, v in
                {m: sc[f'test_{m}'] for m in ['accuracy','recall','precision','f1','roc_auc']}.items()}

    cv_scores = {
        'Logistic Regression': cv_score(LogisticRegression(C=1,max_iter=1000,random_state=42), X_res_s),
        'Random Forest':       cv_score(RandomForestClassifier(n_estimators=200,max_depth=5,random_state=42), X_res),
        'XGBoost':             cv_score(xgb.XGBClassifier(n_estimators=200,max_depth=5,learning_rate=0.1,
                                         eval_metric='logloss',random_state=42,verbosity=0), X_res),
        'SVM':                 cv_score(SVC(kernel='rbf',C=10,gamma='scale',probability=True,random_state=42), X_res_s),
    }

    # ── Test-set metrics ──
    def test_metrics(mdl, Xte):
        yp  = mdl.predict(Xte)
        ypr = mdl.predict_proba(Xte)[:,1]
        cm  = confusion_matrix(y_te, yp)
        tn, fp, fn, tp_ = cm.ravel()
        fpr, tpr, _ = roc_curve(y_te, ypr)
        return {
            'Accuracy':    accuracy_score(y_te, yp),
            'Sensitivity': recall_score(y_te, yp, zero_division=0),
            'Specificity': tn/(tn+fp) if (tn+fp)>0 else 0,
            'Precision':   precision_score(y_te, yp, zero_division=0),
            'F1':          f1_score(y_te, yp, zero_division=0),
            'AUC':         roc_auc_score(y_te, ypr),
            'cm':          cm,
            'fpr':         fpr,
            'tpr':         tpr,
            'y_prob':      ypr,
        }

    results = {
        'Logistic Regression': test_metrics(lr, X_te_s),
        'Random Forest':       test_metrics(rf, X_te_i),
        'XGBoost':             test_metrics(xg, X_te_i),
        'SVM':                 test_metrics(sv, X_te_s),
    }

    # ── SHAP ใช้ Random Forest (TreeExplainer ที่เสถียรที่สุด) ──
    explainer = shap.TreeExplainer(rf)
    sv_raw    = explainer.shap_values(X_te_i)
    # RF อาจคืน list หรือ 3D array → ใช้ class 1 (Poor Recovery)
    if isinstance(sv_raw, list):
        shap_vals = sv_raw[1]
    elif hasattr(sv_raw, 'ndim') and sv_raw.ndim == 3:
        shap_vals = sv_raw[:, :, 1]
    else:
        shap_vals = sv_raw

    return {
        'df': df, 'X_te_i': X_te_i, 'X_te_s': X_te_s,
        'y_te': y_te, 'models': {'lr': lr, 'rf': rf, 'xg': xg, 'sv': sv},
        'scaler': scaler, 'imputer': imp,
        'results': results, 'cv_scores': cv_scores,
        'shap_vals': shap_vals, 'explainer': explainer,
    }


# ─────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ─────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding:10px 0 16px 0;'>
    <div style='font-size:2.5rem;'>🦴</div>
    <div style='font-weight:700; color:#1e3a5f; font-size:0.95rem; line-height:1.4;'>
        AI ประเมินความเสี่ยง<br>ผ่าตัดกระดูกสันหลัง
    </div>
    <div style='font-size:0.75rem; color:#888; margin-top:4px;'>โรงพยาบาลแพร่ | v1.0</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.divider()

page = st.sidebar.radio(
    "เลือกหน้า",
    ["🏠 โครงการวิจัย", "🔍 ประเมินความเสี่ยง", "📊 เปรียบเทียบโมเดล", "🧠 SHAP Analysis"],
    label_visibility="collapsed"
)
st.sidebar.divider()
st.sidebar.info("⚠️ ข้อมูลในระบบนี้เป็น **Synthetic Data** สำหรับสาธิต\nข้อมูลจริงจะถูกนำเข้าเมื่อเก็บข้อมูลครบ", icon=None)

# Load models
artifacts = build_models()
results   = artifacts['results']
cv_scores = artifacts['cv_scores']

# ─────────────────────────────────────────────
# PAGE 1: โครงการวิจัย
# ─────────────────────────────────────────────
if page == "🏠 โครงการวิจัย":
    st.markdown("""
    <div class='main-title'>
        <h1>🦴 ระบบปัญญาประดิษฐ์สำหรับประเมินความเสี่ยงและทำนายการฟื้นตัวหลังผ่าตัดกระดูกสันหลัง</h1>
        <p>Development and Evaluation of an AI-Based Risk Assessment System for Predicting Postoperative Recovery Quality in Spinal Surgery Patients</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""<div class='metric-card'>
            <div class='val' style='color:#1565c0;'>4</div>
            <div class='lbl'>ML Algorithms<br>(LR / RF / XGB / SVM)</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class='metric-card'>
            <div class='val' style='color:#2e7d32;'>9</div>
            <div class='lbl'>Input Features<br>(ตัวแปรนำเข้า)</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class='metric-card'>
            <div class='val' style='color:#c62828;'>150</div>
            <div class='lbl'>เป้าหมายกลุ่มตัวอย่าง<br>(Target Sample Size)</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_a, col_b = st.columns([1.1, 0.9])

    with col_a:
        st.markdown("<div class='section-header'>📋 ข้อมูลโครงการ</div>", unsafe_allow_html=True)
        st.markdown("""
| รายการ | รายละเอียด |
|--------|-----------|
| **ผู้วิจัย** | นายศุภชัย ใจอ้าย |
| **สถาบัน** | โรงพยาบาลแพร่ |
| **ที่ปรึกษา** | - |
| **รูปแบบวิจัย** | Research & Development (R&D) |
| **สถานที่** | โรงพยาบาลแพร่ จังหวัดแพร่ |
| **ระยะเวลา** | 18 เดือน (พ.ศ. 2568-2569) |
| **Outcome** | คะแนน QoR-40 (Poor recovery < 170) |
        """)

        st.markdown("<div class='section-header'>🎯 วัตถุประสงค์</div>", unsafe_allow_html=True)
        st.markdown("""
1. พัฒนาระบบปัญญาประดิษฐ์สำหรับประเมินความเสี่ยงและทำนายการฟื้นตัวหลังผ่าตัดกระดูกสันหลัง
2. ศึกษาผลของการใช้ระบบปัญญาประดิษฐ์ โดยเปรียบเทียบคะแนน QoR-40 ที่ 24 และ 48 ชั่วโมงหลังผ่าตัด
3. ประเมินระดับการยอมรับระบบปัญญาประดิษฐ์ของวิสัญญีพยาบาล (UTAUT framework)
        """)

    with col_b:
        st.markdown("<div class='section-header'>📥 Input Features (9 ตัวแปร)</div>", unsafe_allow_html=True)
        features_info = pd.DataFrame([
            ["อายุ", "Age", "ปี (20-85)"],
            ["เพศ", "Sex", "0=หญิง, 1=ชาย"],
            ["ดัชนีมวลกาย", "BMI", "kg/m²"],
            ["ภาวะความเปราะบาง", "T-FRAIL", "0=Robust → 5=Frail"],
            ["ภาวะโภชนาการ", "NAF", "1=ต่ำ, 2=ปานกลาง, 3=สูง"],
            ["ภาวะสุขภาพ", "ASA", "Class 1-4"],
            ["โรคร่วม", "Comorbidities", "จำนวนโรค"],
            ["ความปวดก่อนผ่าตัด", "NRS", "0-10"],
            ["ชนิดการผ่าตัด", "Surgery Type", "0-3"],
        ], columns=["ชื่อ", "ตัวย่อ", "ค่า"])
        st.dataframe(features_info, use_container_width=True, hide_index=True)

        st.markdown("<div class='section-header'>📤 Outcome Variable</div>", unsafe_allow_html=True)
        st.markdown("""
<div style='background:#f0f4f9; border-radius:8px; padding:14px;'>
    <b>QoR-40 Score</b> (Quality of Recovery)<br>
    <span style='color:#666;'>คะแนน 40-200 คะแนน</span><br><br>
    🔴 <b>Poor Recovery</b> = QoR-40 < 170<br>
    🟢 <b>Good Recovery</b> = QoR-40 ≥ 170
</div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>⚙️ ML Pipeline (ขั้นตอนการสร้างโมเดล)</div>", unsafe_allow_html=True)
    cols = st.columns(5)
    steps = [
        ("1️⃣", "Data\nPreprocessing", "จัดการ Missing Data,\nOne-hot encoding,\nSMOTE"),
        ("2️⃣", "Feature\nSelection", "ตัดเลือกปัจจัย\nที่มีผลต่อ QoR-40"),
        ("3️⃣", "Model\nTraining", "LR, RF, XGB, SVM\n+ 5-fold CV"),
        ("4️⃣", "Evaluation", "Accuracy, Sensitivity\nSpecificity, AUC-ROC"),
        ("5️⃣", "SHAP\nAnalysis", "Explainable AI\nรายบุคคล"),
    ]
    for c, (ico, title, desc) in zip(cols, steps):
        with c:
            st.markdown(f"""
<div style='background:#f8faff; border:1px solid #d0e0ff; border-radius:10px; padding:14px; text-align:center; height:130px;'>
    <div style='font-size:1.4rem;'>{ico}</div>
    <div style='font-weight:700; font-size:0.85rem; margin:4px 0;'>{title}</div>
    <div style='font-size:0.78rem; color:#555;'>{desc}</div>
</div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE 2: ประเมินความเสี่ยง
# ─────────────────────────────────────────────
elif page == "🔍 ประเมินความเสี่ยง":
    st.markdown("""
    <div class='main-title'>
        <h1>🔍 ประเมินความเสี่ยงและทำนายการฟื้นตัวหลังผ่าตัดกระดูกสันหลัง</h1>
        <p>กรอกข้อมูลผู้ป่วยก่อนผ่าตัดเพื่อทำนายการฟื้นตัวหลังผ่าตัด (QoR-40)</p>
    </div>
    """, unsafe_allow_html=True)

    col_inp, col_res = st.columns([1, 1.1])

    with col_inp:
        st.markdown("<div class='section-header'>📋 ข้อมูลผู้ป่วยก่อนผ่าตัด</div>", unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            age_val = st.number_input("อายุ (ปี)", 20, 90, 60)
            sex_val = st.selectbox("เพศ", [0, 1], format_func=lambda x: "หญิง" if x == 0 else "ชาย")
        with c2:
            bmi_val = st.number_input("BMI (kg/m²)", 15.0, 45.0, 24.0, step=0.1)
            frailty_val = st.selectbox(
                "T-FRAIL Score (ภาวะความเปราะบาง)",
                [0,1,2,3,4,5],
                format_func=lambda x: f"{x} - {'Robust (ปกติ)' if x==0 else 'Pre-frail (เริ่มเปราะบาง)' if x<=2 else 'Frail (เปราะบาง)'}"
            )

        c3, c4 = st.columns(2)
        with c3:
            nutrition_val = st.selectbox(
                "NAF Score (ภาวะโภชนาการ)",
                [1, 2, 3],
                format_func=lambda x: {1:"1 - ความเสี่ยงต่ำ", 2:"2 - ความเสี่ยงปานกลาง", 3:"3 - ความเสี่ยงสูง"}[x]
            )
            asa_val = st.selectbox(
                "ASA Class (ภาวะสุขภาพ)",
                [1, 2, 3, 4],
                index=1,
                format_func=lambda x: {1:"ASA I - สุขภาพดี", 2:"ASA II - โรคเล็กน้อย",
                                        3:"ASA III - โรคปานกลาง", 4:"ASA IV - โรครุนแรง"}[x]
            )
        with c4:
            comorbidity_val = st.slider("จำนวนโรคร่วม", 0, 5, 1)
            pain_val = st.slider("ความปวดก่อนผ่าตัด NRS (0-10)", 0, 10, 4)

        surgery_val = st.selectbox(
            "ชนิดการผ่าตัดกระดูกสันหลัง",
            [0, 1, 2, 3],
            format_func=lambda x: {
                0:"0 - Decompression (คลายการกดทับ)",
                1:"1 - Single-level Fusion (เชื่อมข้อ 1 ระดับ)",
                2:"2 - Multi-level Fusion (เชื่อมข้อ ≥2 ระดับ)",
                3:"3 - อื่นๆ"
            }[x]
        )

        predict_btn = st.button("🔍 ทำนายผลการฟื้นตัว", use_container_width=True, type="primary")

    with col_res:
        st.markdown("<div class='section-header'>📊 ผลการทำนาย</div>", unsafe_allow_html=True)

        if predict_btn:
            input_data = pd.DataFrame([[
                age_val, sex_val, bmi_val, frailty_val,
                nutrition_val, asa_val, comorbidity_val,
                pain_val, surgery_val
            ]], columns=FEATURE_COLS)

            inp_imp    = pd.DataFrame(artifacts['imputer'].transform(input_data), columns=FEATURE_COLS)
            inp_scaled = artifacts['scaler'].transform(inp_imp)

            mdls = artifacts['models']
            probs = {
                'Logistic Regression': mdls['lr'].predict_proba(inp_scaled)[0,1],
                'Random Forest':       mdls['rf'].predict_proba(inp_imp)[0,1],
                'XGBoost':             mdls['xg'].predict_proba(inp_imp)[0,1],
                'SVM':                 mdls['sv'].predict_proba(inp_scaled)[0,1],
            }
            ensemble_prob = np.mean(list(probs.values()))

            # Risk level
            if ensemble_prob >= 0.70:
                risk_class = "high"; risk_label = "🔴 ความเสี่ยงสูง (High Risk)"; color = "#e53935"
            elif ensemble_prob >= 0.45:
                risk_class = "mid";  risk_label = "🟡 ความเสี่ยงปานกลาง (Moderate Risk)"; color = "#f9a825"
            else:
                risk_class = "low";  risk_label = "🟢 ความเสี่ยงต่ำ (Low Risk)"; color = "#43a047"

            # Big result box
            st.markdown(f"""
<div class='risk-{risk_class}'>
    <div style='font-size:1.3rem; font-weight:700;'>{risk_label}</div>
    <div style='font-size:2.2rem; font-weight:800; color:{color}; margin:8px 0;'>
        {ensemble_prob*100:.1f}%
    </div>
    <div style='color:#555; font-size:0.9rem;'>ความน่าจะเป็นการฟื้นตัวล่าช้า (QoR-40 &lt; 170)</div>
</div>
            """, unsafe_allow_html=True)

            # Predicted QoR-40 estimate
            qor_est = int(200 - ensemble_prob * 80)
            st.markdown(f"""
<div style='background:#f5f5f5; border-radius:8px; padding:12px; margin-top:12px;'>
    <b>คะแนน QoR-40 ที่คาดการณ์:</b> ~{qor_est} คะแนน
    <span style='color:#888; font-size:0.85rem;'> (เกณฑ์: &lt;170 = ฟื้นตัวล่าช้า)</span>
</div>
            """, unsafe_allow_html=True)

            # Individual model probabilities
            st.markdown("<br><b>ความน่าจะเป็นจากแต่ละโมเดล:</b>", unsafe_allow_html=True)
            prob_df = pd.DataFrame(list(probs.items()), columns=['โมเดล', 'ความน่าจะเป็น'])

            fig_bar, ax = plt.subplots(figsize=(5, 2.8))
            colors_bar = ['#2196F3','#FF9800','#4CAF50','#9C27B0']
            bars = ax.barh(prob_df['โมเดล'], prob_df['ความน่าจะเป็น']*100,
                           color=colors_bar, edgecolor='white', height=0.55)
            ax.axvline(50, color='red', linestyle='--', linewidth=1.2, alpha=0.7, label='50% threshold')
            for bar, val in zip(bars, prob_df['ความน่าจะเป็น']*100):
                ax.text(val+1.5, bar.get_y()+bar.get_height()/2,
                        f'{val:.1f}%', va='center', fontsize=9, fontweight='bold')
            ax.set_xlim(0, 110)
            ax.set_xlabel('ความน่าจะเป็น (%)', fontsize=9)
            ax.set_title('Model Predictions', fontsize=10, fontweight='bold')
            ax.legend(fontsize=8)
            fig_bar.tight_layout()
            st.pyplot(fig_bar)
            plt.close()

            # Clinical recommendations
            st.markdown("<div class='section-header'>💊 คำแนะนำทางคลินิก</div>", unsafe_allow_html=True)
            if risk_class == "high":
                st.error("""
**แผนการดูแลก่อนผ่าตัด (ความเสี่ยงสูง):**
- 📞 ส่งปรึกษาแพทย์เฉพาะทาง (โภชนาการ / อายุรกรรม)
- 🏋️ ประเมินและแก้ไขภาวะความเปราะบางก่อนผ่าตัด
- 🏥 เตรียมหอผู้ป่วยวิกฤต (ICU) สำรองไว้
- 📋 วางแผนควบคุมความปวดหลังผ่าตัดแบบผสมผสาน
- 👁️ ติดตามอาการภาวะสับสนเฉียบพลันหลังผ่าตัด
                """)
            elif risk_class == "mid":
                st.warning("""
**แผนการดูแลก่อนผ่าตัด (ความเสี่ยงปานกลาง):**
- 📋 วางแผนการดูแลแบบ ERAS protocol
- 💊 เตรียมแผนควบคุมความปวดล่วงหน้า
- 📊 ติดตามใกล้ชิดช่วง 24-48 ชม. หลังผ่าตัด
- 🥗 ปรับโภชนาการและ Exercise prehabilitation
                """)
            else:
                st.success("""
**แผนการดูแลก่อนผ่าตัด (ความเสี่ยงต่ำ):**
- ✅ ดำเนินการตามแนวทางมาตรฐานของโรงพยาบาล
- 📝 ติดตามตามปกติหลังผ่าตัด
- 💪 Prehabilitation ตามมาตรฐาน
                """)

            # SHAP for this patient (using RF explainer)
            try:
                explainer_pt = artifacts['explainer']
                sv_raw_pt    = explainer_pt.shap_values(inp_imp)
                if isinstance(sv_raw_pt, list):
                    shap_vals_pt = sv_raw_pt[1]
                elif hasattr(sv_raw_pt, 'ndim') and sv_raw_pt.ndim == 3:
                    shap_vals_pt = sv_raw_pt[:, :, 1]
                else:
                    shap_vals_pt = sv_raw_pt
                st.markdown("<div class='section-header'>🧠 SHAP: ปัจจัยที่ส่งผลสำหรับผู้ป่วยรายนี้</div>", unsafe_allow_html=True)
                labels = [FEATURE_LABELS_TH[f] for f in FEATURE_COLS]
                sv_pt  = shap_vals_pt[0] if shap_vals_pt.ndim > 1 else shap_vals_pt
                fig_shap, ax_s = plt.subplots(figsize=(5.5, 3.5))
                colors_s = ['#e53935' if v > 0 else '#43a047' for v in sv_pt]
                order = np.argsort(np.abs(sv_pt))[::-1]
                ax_s.barh([labels[i] for i in order[:7]], sv_pt[order[:7]],
                          color=[colors_s[i] for i in order[:7]], edgecolor='white')
                ax_s.axvline(0, color='black', linewidth=0.8)
                ax_s.set_xlabel('SHAP Value (ทิศทางและน้ำหนักอิทธิพล)', fontsize=8)
                ax_s.set_title('SHAP: ปัจจัยที่ส่งผลต่อการทำนายรายนี้\n(แดง=เพิ่มความเสี่ยง, เขียว=ลดความเสี่ยง)', fontsize=9)
                ax_s.tick_params(labelsize=8)
                fig_shap.tight_layout()
                st.pyplot(fig_shap)
                plt.close()
            except Exception:
                pass
        else:
            st.info("👈 กรอกข้อมูลผู้ป่วยด้านซ้าย แล้วกด **ทำนายผลการฟื้นตัว**")


# ─────────────────────────────────────────────
# PAGE 3: เปรียบเทียบโมเดล
# ─────────────────────────────────────────────
elif page == "📊 เปรียบเทียบโมเดล":
    st.markdown("""
    <div class='main-title'>
        <h1>📊 เปรียบเทียบประสิทธิภาพโมเดล ML</h1>
        <p>ผลการทดสอบ 4 อัลกอริทึม บน Synthetic Data (n=150) | 5-Fold Cross-Validation</p>
    </div>
    """, unsafe_allow_html=True)

    # Summary metrics table
    st.markdown("<div class='section-header'>📋 ตารางเปรียบเทียบประสิทธิภาพ (Test Set)</div>", unsafe_allow_html=True)
    metrics_data = {
        name: {
            'Accuracy':    f"{v['Accuracy']:.4f}",
            'Sensitivity': f"{v['Sensitivity']:.4f}",
            'Specificity': f"{v['Specificity']:.4f}",
            'Precision':   f"{v['Precision']:.4f}",
            'F1-Score':    f"{v['F1']:.4f}",
            'AUC-ROC':     f"{v['AUC']:.4f}",
        }
        for name, v in results.items()
    }
    df_metrics = pd.DataFrame(metrics_data).T
    best_model = max(results, key=lambda k: results[k]['AUC'])
    st.dataframe(
        df_metrics.style.highlight_max(color='#d5f5e3', axis=0),
        use_container_width=True
    )
    st.success(f"🏆 **Best Model: {best_model}** (AUC-ROC = {results[best_model]['AUC']:.4f})")

    # CV scores
    st.markdown("<div class='section-header'>🔄 5-Fold Cross-Validation (Training Set + SMOTE)</div>", unsafe_allow_html=True)
    cv_data = {
        name: {
            'Accuracy (CV)':    f"{cv_scores[name]['accuracy'][0]:.4f} ± {cv_scores[name]['accuracy'][1]:.4f}",
            'Sensitivity (CV)': f"{cv_scores[name]['recall'][0]:.4f} ± {cv_scores[name]['recall'][1]:.4f}",
            'AUC-ROC (CV)':     f"{cv_scores[name]['roc_auc'][0]:.4f} ± {cv_scores[name]['roc_auc'][1]:.4f}",
        }
        for name in cv_scores
    }
    st.dataframe(pd.DataFrame(cv_data).T, use_container_width=True)

    # ROC curves + Bar chart
    col_roc, col_bar = st.columns(2)

    with col_roc:
        st.markdown("<div class='section-header'>📈 ROC Curves</div>", unsafe_allow_html=True)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
        colors_roc = ['#1565c0','#f57c00','#2e7d32','#7b1fa2']
        for (name, res), clr in zip(results.items(), colors_roc):
            ax_roc.plot(res['fpr'], res['tpr'], color=clr, linewidth=2.2,
                       label=f"{name[:10]} (AUC={res['AUC']:.3f})")
        ax_roc.plot([0,1],[0,1],'k--', linewidth=1, alpha=0.5, label='Random')
        ax_roc.fill_between([0,1],[0,1], alpha=0.05, color='gray')
        ax_roc.set_xlabel('1 - Specificity (FPR)', fontsize=10)
        ax_roc.set_ylabel('Sensitivity (TPR)', fontsize=10)
        ax_roc.set_title('ROC Curves', fontsize=11, fontweight='bold')
        ax_roc.legend(fontsize=8.5, loc='lower right')
        ax_roc.grid(True, alpha=0.3)
        fig_roc.tight_layout()
        st.pyplot(fig_roc)
        plt.close()

    with col_bar:
        st.markdown("<div class='section-header'>📊 Bar Chart เปรียบเทียบ</div>", unsafe_allow_html=True)
        metrics_plot = ['Accuracy', 'Sensitivity', 'Specificity', 'F1', 'AUC']
        x = np.arange(len(metrics_plot))
        width = 0.2
        fig_b, ax_b = plt.subplots(figsize=(6, 5))
        colors_b = ['#1565c0','#f57c00','#2e7d32','#7b1fa2']
        for i, (name, res) in enumerate(results.items()):
            vals = [res[m] for m in metrics_plot]
            ax_b.bar(x + i*width, vals, width, label=name[:8], color=colors_b[i], alpha=0.85)
        ax_b.set_xticks(x + width*1.5)
        ax_b.set_xticklabels(metrics_plot, fontsize=9)
        ax_b.set_ylabel('Score', fontsize=10)
        ax_b.set_ylim(0, 1.12)
        ax_b.set_title('Performance Comparison', fontsize=11, fontweight='bold')
        ax_b.axhline(0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='0.8 threshold')
        ax_b.legend(fontsize=8)
        ax_b.grid(axis='y', alpha=0.3)
        fig_b.tight_layout()
        st.pyplot(fig_b)
        plt.close()

    # Confusion matrices
    st.markdown("<div class='section-header'>🔲 Confusion Matrices</div>", unsafe_allow_html=True)
    cm_cols = st.columns(4)
    for col, (name, res) in zip(cm_cols, results.items()):
        with col:
            fig_cm, ax_cm = plt.subplots(figsize=(3, 2.8))
            sns.heatmap(res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax_cm,
                        xticklabels=['Good','Poor'], yticklabels=['Good','Poor'])
            ax_cm.set_title(name[:9], fontsize=9, fontweight='bold')
            ax_cm.set_xlabel('Predicted', fontsize=8)
            ax_cm.set_ylabel('Actual', fontsize=8)
            ax_cm.tick_params(labelsize=7)
            fig_cm.tight_layout()
            st.pyplot(fig_cm)
            plt.close()

    # Data distribution
    st.markdown("<div class='section-header'>📊 การกระจายข้อมูล Synthetic Dataset</div>", unsafe_allow_html=True)
    df_viz = artifacts['df']
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        fig_h, ax_h = plt.subplots(figsize=(4.5, 3))
        ax_h.hist(df_viz['QoR40_Score'], bins=20, color='steelblue', edgecolor='white', alpha=0.85)
        ax_h.axvline(170, color='red', linestyle='--', linewidth=2, label='Poor recovery threshold')
        ax_h.set_xlabel('QoR-40 Score', fontsize=10)
        ax_h.set_ylabel('Frequency', fontsize=10)
        ax_h.set_title('Distribution of QoR-40', fontsize=11, fontweight='bold')
        ax_h.legend(fontsize=8)
        fig_h.tight_layout()
        st.pyplot(fig_h)
        plt.close()
    with col_d2:
        fig_f, ax_f = plt.subplots(figsize=(4.5, 3))
        poor_by_frailty = df_viz.groupby('Frailty_TFRAIL')['Poor_Recovery'].mean()*100
        ax_f.bar(poor_by_frailty.index, poor_by_frailty.values, color='coral', edgecolor='white')
        ax_f.set_xlabel('T-FRAIL Score', fontsize=10)
        ax_f.set_ylabel('Poor Recovery Rate (%)', fontsize=10)
        ax_f.set_title('Poor Recovery Rate by Frailty', fontsize=11, fontweight='bold')
        fig_f.tight_layout()
        st.pyplot(fig_f)
        plt.close()
    with col_d3:
        fig_a, ax_a = plt.subplots(figsize=(4.5, 3))
        poor_by_asa = df_viz.groupby('ASA_Class')['Poor_Recovery'].mean()*100
        ax_a.bar([f'ASA {i}' for i in poor_by_asa.index],
                 poor_by_asa.values, color='mediumpurple', edgecolor='white')
        ax_a.set_xlabel('ASA Class', fontsize=10)
        ax_a.set_ylabel('Poor Recovery Rate (%)', fontsize=10)
        ax_a.set_title('Poor Recovery Rate by ASA', fontsize=11, fontweight='bold')
        fig_a.tight_layout()
        st.pyplot(fig_a)
        plt.close()


# ─────────────────────────────────────────────
# PAGE 4: SHAP ANALYSIS
# ─────────────────────────────────────────────
elif page == "🧠 SHAP Analysis":
    st.markdown("""
    <div class='main-title'>
        <h1>🧠 SHAP Explainable AI Analysis</h1>
        <p>SHapley Additive exPlanations — อธิบายน้ำหนักความสำคัญของปัจจัยในการทำนายรายบุคคล</p>
    </div>
    """, unsafe_allow_html=True)

    st.info("**SHAP Analysis** ช่วยให้วิสัญญีพยาบาลเข้าใจว่าปัจจัยใดมีผลต่อการทำนายมากที่สุด สำหรับผู้ป่วยแต่ละราย ซึ่งช่วยสนับสนุนการตัดสินใจทางคลินิกได้อย่างมีประสิทธิภาพ")

    shap_vals = artifacts['shap_vals']   # already class-1 slice from build_models()
    X_te_i    = artifacts['X_te_i']
    labels    = [FEATURE_LABELS_TH[f] for f in FEATURE_COLS]

    col_s1, col_s2 = st.columns(2)

    with col_s1:
        st.markdown("<div class='section-header'>📊 SHAP Feature Importance (Bar)</div>", unsafe_allow_html=True)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        order_s   = np.argsort(mean_shap)
        fig_imp, ax_imp = plt.subplots(figsize=(5.5, 4.5))
        ax_imp.barh([labels[i] for i in order_s], mean_shap[order_s],
                    color='#2196F3', edgecolor='white', alpha=0.85)
        ax_imp.set_xlabel('Mean |SHAP Value|', fontsize=10)
        ax_imp.set_title('SHAP Feature Importance\n(XGBoost Model)', fontsize=11, fontweight='bold')
        ax_imp.grid(axis='x', alpha=0.3)
        fig_imp.tight_layout()
        st.pyplot(fig_imp)
        plt.close()

        st.markdown("""
<div style='background:#f0f4ff; border-radius:8px; padding:12px; font-size:0.88rem;'>
    <b>การตีความ:</b><br>
    ค่า SHAP สูง = ปัจจัยนั้นมีอิทธิพลต่อการทำนายมาก<br>
    เช่น ถ้า T-FRAIL มี SHAP value สูงสุด = ภาวะเปราะบาง<br>
    เป็นตัวพยากรณ์ QoR-40 ที่สำคัญที่สุด
</div>
        """, unsafe_allow_html=True)

    with col_s2:
        st.markdown("<div class='section-header'>🌐 SHAP Beeswarm Plot</div>", unsafe_allow_html=True)
        try:
            fig_bee, ax_bee = plt.subplots(figsize=(6, 5))
            shap.summary_plot(shap_vals, X_te_i, feature_names=labels,
                              show=False, plot_size=None, color_bar=True)
            ax_curr = plt.gca()
            ax_curr.set_title('SHAP Beeswarm\n(ทิศทางและขนาดอิทธิพลของแต่ละปัจจัย)',
                              fontsize=10, fontweight='bold')
            fig_bee = plt.gcf()
            fig_bee.set_size_inches(6, 5)
            plt.tight_layout()
            st.pyplot(fig_bee)
            plt.close('all')
        except Exception as e:
            # Fallback: manual beeswarm
            fig_bee2, ax_b2 = plt.subplots(figsize=(6, 5))
            for i, feat in enumerate(labels):
                jitter = np.random.uniform(-0.2, 0.2, len(shap_vals))
                sc = ax_b2.scatter(shap_vals[:,i], np.ones(len(shap_vals))*i + jitter,
                                   c=X_te_i.iloc[:,i], cmap='RdBu_r', alpha=0.6, s=20)
            ax_b2.set_yticks(range(len(labels)))
            ax_b2.set_yticklabels(labels, fontsize=8)
            ax_b2.axvline(0, color='black', linewidth=1)
            ax_b2.set_xlabel('SHAP Value', fontsize=10)
            ax_b2.set_title('SHAP Beeswarm (Manual)', fontsize=11, fontweight='bold')
            plt.colorbar(sc, ax=ax_b2, label='Feature Value')
            fig_bee2.tight_layout()
            st.pyplot(fig_bee2)
            plt.close()

    # SHAP for selected patient
    st.markdown("<div class='section-header'>👤 SHAP รายบุคคล (Individual Patient)</div>", unsafe_allow_html=True)
    pt_idx = st.slider("เลือกผู้ป่วย (index จาก Test Set)", 0, len(X_te_i)-1, 0)

    sv_pt = shap_vals[pt_idx]
    xp_pt = X_te_i.iloc[pt_idx]
    pred_label = "🔴 ฟื้นตัวล่าช้า" if sv_pt.sum() > 0 else "🟢 ฟื้นตัวดี"

    c_pt1, c_pt2 = st.columns([1, 1.2])
    with c_pt1:
        st.markdown(f"**ผู้ป่วย index {pt_idx}** | การทำนาย: {pred_label}")
        pt_table = pd.DataFrame({
            'ตัวแปร': labels,
            'ค่าที่กรอก': xp_pt.values,
            'SHAP Value': sv_pt.round(4)
        })
        pt_table['ผล'] = pt_table['SHAP Value'].apply(lambda x: '🔴 เพิ่มความเสี่ยง' if x > 0 else '🟢 ลดความเสี่ยง')
        st.dataframe(pt_table, use_container_width=True, hide_index=True)

    with c_pt2:
        fig_pt, ax_pt = plt.subplots(figsize=(5.5, 4))
        colors_pt = ['#e53935' if v > 0 else '#43a047' for v in sv_pt]
        ord_pt    = np.argsort(np.abs(sv_pt))
        ax_pt.barh([labels[i] for i in ord_pt],
                   sv_pt[ord_pt],
                   color=[colors_pt[i] for i in ord_pt], edgecolor='white', alpha=0.9)
        ax_pt.axvline(0, color='black', linewidth=1.2)
        ax_pt.set_xlabel('SHAP Value', fontsize=10)
        ax_pt.set_title(f'SHAP: ผู้ป่วยที่ {pt_idx}\n(แดง=เพิ่มความเสี่ยง, เขียว=ลดความเสี่ยง)',
                        fontsize=10, fontweight='bold')
        ax_pt.tick_params(labelsize=8.5)
        ax_pt.grid(axis='x', alpha=0.3)
        fig_pt.tight_layout()
        st.pyplot(fig_pt)
        plt.close()

    # Methodology note
    st.markdown("<div class='section-header'>📚 ทฤษฎีพื้นฐาน SHAP</div>", unsafe_allow_html=True)
    st.markdown("""
<div style='background:#f8f9fa; border-radius:8px; padding:16px; font-size:0.9rem; line-height:1.6;'>
    <b>SHAP (SHapley Additive exPlanations)</b> คือเทคนิค Explainable AI ที่ใช้หลักการจาก Game Theory
    โดยคำนวณ "Shapley Value" ของแต่ละตัวแปรในการทำนาย<br><br>

    <b>ประโยชน์ในงานวิจัยนี้:</b><br>
    • บอกว่าปัจจัยใดมีน้ำหนักมากที่สุดในการทำนาย QoR-40 สำหรับผู้ป่วยแต่ละราย<br>
    • ทิศทาง (+/-) บอกว่าปัจจัยนั้นเพิ่มหรือลดความเสี่ยงการฟื้นตัวล่าช้า<br>
    • ช่วยให้วิสัญญีพยาบาลไม่มองระบบ AI เป็น "กล่องดำ" (black box)<br>
    • เพิ่มความเชื่อมั่นในการนำผลไปใช้วางแผนการดูแลผู้ป่วย (Transparency)
</div>
    """, unsafe_allow_html=True)

# Footer
st.sidebar.markdown("""
<div style='text-align:center; font-size:0.75rem; color:#aaa; margin-top:20px;'>
    ระบบ AI นี้พัฒนาเพื่อการวิจัย<br>
    ไม่ใช่การวินิจฉัยทางการแพทย์<br><br>
    <b>โรงพยาบาลแพร่ | 2568-2569</b>
</div>
""", unsafe_allow_html=True)
