import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import joblib
import os
import datetime
import warnings

warnings.filterwarnings('ignore')

st.set_page_config(
    page_title="Predição de Burnout Estudantil",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:opsz,wght@9..40,300;9..40,400;9..40,500;9..40,600;9..40,700&display=swap');

:root {
  --bg-main: #faf8f4; --bg-card: #ffffff; --bg-sidebar: #f2ede4;
  --border: rgba(160,120,50,0.1); --border-strong: rgba(160,120,50,0.2);
  --text-primary: #1c1916; --text-secondary: #6b5e50; --text-muted: #a0907e;
  --accent: #b86b0a; --accent-glow: rgba(184,107,10,0.08);
  --header-bg: linear-gradient(135deg,#fefcf8 0%,#f5f0e8 100%);
  --header-glow: rgba(184,107,10,0.06); --header-border: rgba(160,120,50,0.16);
  --badge-bg: rgba(184,107,10,0.07); --badge-border: rgba(184,107,10,0.18);
  --pill-bg: rgba(0,0,0,0.03); --pill-border: rgba(0,0,0,0.08);
  --pill-color: #a0907e;
}

html, body, [class*="css"], .stMarkdown, p, label, h1, h2, h3, h4, button {
    font-family: 'DM Sans', -apple-system, sans-serif !important;
}

#MainMenu, footer { visibility: hidden; }
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 1.5rem 2.5rem 3rem; }

[data-testid="stSidebar"] {
    background: var(--bg-sidebar) !important;
    border-right: 1px solid var(--border-strong) !important;
}
[data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important;
}

[data-testid="metric-container"] {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    padding: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="metric-container"]:hover { border-color: var(--border-strong) !important; }
[data-testid="metric-container"] label {
    font-size: 0.69rem !important; font-weight: 600 !important;
    text-transform: uppercase !important; letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"] { font-size: 1.3rem !important; font-weight: 600 !important; }

.result-box-alto {
    background: linear-gradient(135deg, rgba(239,68,68,0.1) 0%, rgba(239,68,68,0.03) 100%);
    border: 1px solid rgba(239,68,68,0.28); border-left: 4px solid #ef4444;
    padding: 24px 28px; border-radius: 12px; box-shadow: 0 0 40px rgba(239,68,68,0.05); margin: 12px 0;
}
.result-box-medio {
    background: linear-gradient(135deg, rgba(249,115,22,0.1) 0%, rgba(249,115,22,0.03) 100%);
    border: 1px solid rgba(249,115,22,0.28); border-left: 4px solid #f97316;
    padding: 24px 28px; border-radius: 12px; box-shadow: 0 0 40px rgba(249,115,22,0.05); margin: 12px 0;
}
.result-box-baixo {
    background: linear-gradient(135deg, rgba(34,197,94,0.1) 0%, rgba(34,197,94,0.03) 100%);
    border: 1px solid rgba(34,197,94,0.28); border-left: 4px solid #22c55e;
    padding: 24px 28px; border-radius: 12px; box-shadow: 0 0 40px rgba(34,197,94,0.05); margin: 12px 0;
}

.stTabs [data-baseweb="tab-list"] {
    gap: 2px; background: var(--bg-card); border-radius: 12px;
    padding: 4px; border: 1px solid var(--border); margin-bottom: 20px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px; padding: 9px 24px; font-weight: 500; font-size: 0.87rem;
    border: none !important; background: transparent !important; transition: color 0.15s;
}
.stTabs [aria-selected="true"] {
    background: var(--accent-glow) !important; color: var(--accent) !important;
    font-weight: 600 !important; border: 1px solid rgba(184,107,10,0.2) !important;
}

[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #b86b0a 0%, #974f06 100%) !important;
    color: #ffffff !important; border: none !important; border-radius: 10px !important;
    font-weight: 700 !important; font-size: 0.92rem !important; letter-spacing: 0.02em;
    box-shadow: 0 4px 20px rgba(184,107,10,0.22) !important; transition: opacity 0.15s !important;
}
[data-testid="stButton"] > button[kind="primary"]:hover { opacity: 0.88 !important; }

[data-testid="stDownloadButton"] > button {
    border-radius: 9px !important; font-weight: 500 !important; font-size: 0.85rem !important;
}
[data-testid="stDownloadButton"] > button:hover {
    border-color: var(--accent) !important; color: var(--accent) !important;
}

[data-testid="stMarkdownContainer"] h3 {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-weight: 400 !important; font-size: 1.35rem !important;
}

hr { border-color: var(--border) !important; margin: 2rem 0 !important; }

[data-testid="stDataFrame"] {
    border-radius: 12px; overflow: hidden; border: 1px solid var(--border) !important;
}
.stAlert { border-radius: 12px !important; }
</style>
""", unsafe_allow_html=True)

FEATURES_ENC = [
    'age', 'gender_enc', 'academic_year', 'study_hours_per_day',
    'exam_pressure', 'academic_performance', 'stress_level',
    'anxiety_score', 'depression_score', 'sleep_hours',
    'physical_activity', 'social_support', 'screen_time',
    'internet_usage', 'financial_stress', 'family_expectation',
    'dropout_risk'
]

FEATURE_LABELS = {
    'age': 'Idade', 'gender': 'Gênero', 'academic_year': 'Ano Acadêmico',
    'study_hours_per_day': 'Horas de Estudo/dia', 'exam_pressure': 'Pressão de Provas',
    'academic_performance': 'Desempenho Acadêmico', 'stress_level': 'Nível de Estresse',
    'anxiety_score': 'Ansiedade', 'depression_score': 'Depressão',
    'sleep_hours': 'Horas de Sono', 'physical_activity': 'Atividade Física',
    'social_support': 'Suporte Social', 'screen_time': 'Tempo de Tela',
    'internet_usage': 'Uso de Internet', 'financial_stress': 'Estresse Financeiro',
    'family_expectation': 'Expectativa Familiar',
    'dropout_risk': 'Risco de Evasão'
}

label_map = {0: 'Baixo', 1: 'Médio', 2: 'Alto'}
badge_map = {'Baixo': 'BAIXO', 'Médio': 'MÉDIO', 'Alto': 'ALTO'}
color_map = {'Baixo': '#22c55e', 'Médio': '#f97316', 'Alto': '#ef4444'}
css_map   = {'Baixo': 'result-box-baixo', 'Médio': 'result-box-medio', 'Alto': 'result-box-alto'}

CHART_FONT = dict(family='DM Sans, -apple-system, sans-serif', color='#6b5e50', size=13)
CHART_BASE = dict(font=CHART_FONT, paper_bgcolor='#ffffff', plot_bgcolor='#faf8f4')
_GRID      = 'rgba(0,0,0,0.05)'
_TICK      = '#a0907e'
_MEDIAN_C  = '#3b82f6'

DATASET_MEDIANS = {
    'age': 23.5, 'academic_year': 2.5, 'study_hours_per_day': 5.0,
    'exam_pressure': 5.5, 'academic_performance': 65.0, 'stress_level': 4.0,
    'anxiety_score': 2.5, 'depression_score': 2.0, 'sleep_hours': 7.0,
    'physical_activity': 3.5, 'social_support': 4.5, 'screen_time': 5.0,
    'internet_usage': 5.0, 'financial_stress': 4.0, 'family_expectation': 4.5,
    'dropout_risk': 1.5, 'gender': 0
}

def generate_dashboard_html(nivel, confianca, inputs, probas, fig_gauge, fig_radar, fig_comp):
    _bar_colors  = ['#22c55e', '#f97316', '#ef4444']
    _bar_labels  = ['Baixo', 'Médio', 'Alto']
    _bar_values  = [float(v) * 100 for v in probas]
    _bar_max     = max(_bar_values) * 1.15
    _bar_items   = ""
    for lbl, val, color in zip(_bar_labels, _bar_values, _bar_colors):
        bar_h_pct = val / _bar_max * 100
        _bar_items += f"""
        <div style="flex:1;display:flex;flex-direction:column;align-items:center;gap:0;height:100%;">
          <div style="flex:1;display:flex;flex-direction:column;justify-content:flex-end;width:100%;align-items:center;">
            <span style="font-size:13px;font-weight:600;color:#ffffff;background:{color};
                         padding:3px 10px;border-radius:4px;margin-bottom:6px;">{val:.1f}%</span>
            <div style="width:55%;background:{color};height:{bar_h_pct:.1f}%;
                        border-radius:5px 5px 0 0;min-height:4px;"></div>
          </div>
          <span style="font-size:13px;color:#6b5e50;font-family:'DM Sans',sans-serif;
                       padding:8px 0 0;">{lbl}</span>
        </div>"""
    h_bar = f"""
    <div style="background:#faf8f4;border-radius:8px;padding:16px 24px 12px;
                font-family:'DM Sans',sans-serif;">
      <div style="font-size:11px;font-weight:600;color:#a0907e;letter-spacing:.08em;
                  text-transform:uppercase;margin-bottom:12px;">Probabilidade (%)</div>
      <div style="display:flex;gap:12px;height:240px;align-items:stretch;">
        {_bar_items}
      </div>
    </div>"""

    static_cfg = dict(staticPlot=True, displayModeBar=False)
    cfg = dict(full_html=False, include_plotlyjs=False, config=static_cfg)
    h_gauge = fig_gauge.to_html(**cfg)
    h_radar = fig_radar.to_html(**cfg)
    h_comp  = fig_comp.to_html(**cfg)

    metrics = [
        ("Nível de Burnout",  nivel),
        ("Estresse",          f"{inputs['stress_level']:.1f} / 10"),
        ("Ansiedade",         f"{inputs['anxiety_score']:.1f} / 10"),
        ("Depressão",         f"{inputs['depression_score']:.1f} / 10"),
        ("Sono (h/dia)",      f"{inputs['sleep_hours']:.1f} h"),
        ("Confiança Modelo",  f"{confianca:.1f} %"),
        ("Atividade Física",  f"{inputs['physical_activity']:.1f} / 10"),
        ("Suporte Social",    f"{inputs['social_support']:.1f} / 10"),
    ]
    metric_rows = "".join(
        f"<tr><td>{k}</td><td><strong>{v}</strong></td></tr>" for k, v in metrics
    )

    result_color = color_map[nivel]
    timestamp    = datetime.datetime.now().strftime('%d/%m/%Y %H:%M:%S')

    html = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Dashboard Burnout — {timestamp}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link href="https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@400;500;600&display=swap" rel="stylesheet">
<script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: 'DM Sans', sans-serif; background: #faf8f4; color: #1c1916; }}
  .page {{ max-width: 1200px; margin: 0 auto; padding: 36px 28px; }}
  header {{ text-align: center; margin-bottom: 36px; }}
  header h1 {{ font-family: 'DM Serif Display', serif; font-size: 2.2rem; color: #1c1916; font-weight: 400; }}
  header p  {{ color: #a0907e; margin-top: 8px; font-size: 0.88rem; }}
  .result-banner {{
    border-left: 6px solid {result_color};
    background: #ffffff;
    padding: 20px 24px;
    border-radius: 10px;
    margin-bottom: 28px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
  }}
  .result-banner h2 {{ font-family: 'DM Serif Display', serif; font-size: 1.9rem; color: {result_color}; font-weight: 400; }}
  .result-banner p  {{ color: #6b5e50; margin-top: 6px; font-size: 0.93rem; }}
  .grid-2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 24px; }}
  .card {{
    background: #ffffff; border-radius: 10px; padding: 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    border: 1px solid #f0e8dc;
  }}
  .card h3 {{
    font-family: 'DM Sans', sans-serif; font-size: 0.75rem; font-weight: 600;
    color: #a0907e; margin-bottom: 14px; text-transform: uppercase; letter-spacing: .08em;
  }}
  table {{ width: 100%; border-collapse: collapse; }}
  td {{ padding: 9px 10px; border-bottom: 1px solid #f2ede4; font-size: 0.9rem; color: #1c1916; }}
  tr:last-child td {{ border-bottom: none; }}
  td:first-child {{ color: #a0907e; }}
  .section-title {{
    font-family: 'DM Serif Display', serif; font-size: 1.15rem; font-weight: 400;
    margin: 28px 0 14px; color: #1c1916;
  }}
  .model-section-header {{
    background: #f2ede4;
    border-left: 5px solid #b86b0a;
    border-radius: 8px;
    padding: 20px 24px;
    margin: 36px 0 20px;
  }}
  .model-section-header h2 {{ font-family: 'DM Serif Display', serif; font-size: 1.2rem; color: #1c1916; font-weight: 400; margin-bottom: 8px; }}
  .model-section-header p  {{ color: #6b5e50; font-size: 0.9rem; line-height: 1.6; }}
  footer {{ text-align: center; color: #c4b8ae; font-size: 0.8rem; margin-top: 44px; padding-top: 20px; border-top: 1px solid #f0e8dc; }}
</style>
</head>
<body>
<div class="page">

  <header>
    <h1>Relatório de Predição de Burnout Estudantil</h1>
    <p>Gerado em {timestamp} &nbsp;·&nbsp; Random Forest Classifier &nbsp;·&nbsp; Dataset: 1M registros</p>
  </header>

  <div class="result-banner">
    <h2>Burnout: {nivel.upper()}</h2>
    <p>Confiança do modelo: <strong>{confianca:.1f}%</strong></p>
  </div>

  <div class="grid-2">
    <div class="card">
      <h3>Gauge de Confiança</h3>
      {h_gauge}
    </div>
    <div class="card">
      <h3>Métricas do Indivíduo</h3>
      <table>{metric_rows}</table>
    </div>
  </div>

  <div class="card" style="margin-bottom:24px">
    <h3>Probabilidades por Classe</h3>
    {h_bar}
  </div>

  <p class="section-title">Perfil do Indivíduo vs Perfil Médio Geral — Top 10 Variáveis</p>
  <div class="grid-2" style="margin-bottom:24px">
    <div class="card">{h_radar}</div>
    <div class="card">{h_comp}</div>
  </div>

  <footer>
    Burnout Prediction App &nbsp;·&nbsp; Scikit-learn · Plotly · Streamlit<br>
    <em>Para fins educacionais e de pesquisa. Não substitui avaliação profissional.</em>
  </footer>

</div>
</body>
</html>"""
    return html


@st.cache_resource
def load_artifacts():
    model_path = 'burnout_rf_model.pkl'

    if os.path.exists(model_path):
        model       = joblib.load('burnout_rf_model.pkl')
        le_gender   = joblib.load('gender_encoder.pkl')
        feat_imp_df = joblib.load('feature_importance.pkl')
        thresholds  = joblib.load('burnout_thresholds.pkl')
        return model, le_gender, feat_imp_df, thresholds, None

    try:
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder

        CSV_PATH = 'student_mental_health_burnout_1M.csv'
        if not os.path.exists(CSV_PATH):
            return None, None, None, None, f"Arquivo '{CSV_PATH}' não encontrado."

        df = pd.read_csv(CSV_PATH)

        p33 = df['burnout_score'].quantile(0.33)
        p66 = df['burnout_score'].quantile(0.66)

        df['burnout_class'] = df['burnout_score'].apply(
            lambda s: 0 if s <= p33 else (1 if s <= p66 else 2)
        )

        le = LabelEncoder()
        df['gender_enc'] = le.fit_transform(df['gender'])

        X = df[FEATURES_ENC]
        y = df['burnout_class']
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

        clf = RandomForestClassifier(n_estimators=150, max_depth=18,
                                     class_weight='balanced', n_jobs=-1, random_state=42)
        clf.fit(X_tr, y_tr)

        feat_names = list(FEATURE_LABELS.keys())
        fidf = pd.DataFrame({'feature': feat_names, 'importance': clf.feature_importances_})
        fidf = fidf.sort_values('importance', ascending=False)

        joblib.dump(clf,  'burnout_rf_model.pkl')
        joblib.dump(le,   'gender_encoder.pkl')
        joblib.dump(fidf, 'feature_importance.pkl')
        joblib.dump({'p33': p33, 'p66': p66}, 'burnout_thresholds.pkl')

        return clf, le, fidf, {'p33': p33, 'p66': p66}, None

    except Exception as e:
        return None, None, None, None, str(e)


st.markdown("""
<div style="
    background: var(--header-bg);
    border: 1px solid var(--header-border);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
">
  <div style="
    position: absolute; top: -60px; right: -60px; width: 260px; height: 260px;
    background: radial-gradient(circle, var(--header-glow) 0%, transparent 65%);
    border-radius: 50%; pointer-events: none;
  "></div>
  <div style="
    display: inline-block;
    background: var(--badge-bg); border: 1px solid var(--badge-border);
    color: var(--accent); font-size: 0.7rem; font-weight: 600;
    padding: 4px 13px; border-radius: 20px; letter-spacing: 0.1em;
    text-transform: uppercase; margin-bottom: 14px; font-family: 'DM Sans', sans-serif;
  ">Análise Preditiva · Saúde Mental Estudantil</div>
  <h1 style="
    color: var(--text-primary); margin: 0 0 10px;
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 2.1rem; font-weight: 400; letter-spacing: -0.01em; line-height: 1.2;
  ">Predição de Burnout Estudantil</h1>
  <p style="color: var(--text-muted); font-size: 0.85rem; margin: 0 0 18px; line-height: 1.6; font-family: 'DM Sans', sans-serif;">
    Preencha os dados abaixo e receba uma análise de risco de burnout baseada em
    Random Forest treinado com 1 milhão de registros acadêmicos.
  </p>
  <div style="display:flex; gap:8px; flex-wrap:wrap">
    <span style="background:var(--pill-bg); color:var(--pill-color); font-size:0.73rem;
                 font-weight:500; padding:4px 12px; border-radius:20px;
                 border:1px solid var(--pill-border); font-family:'DM Sans',sans-serif;">
      Random Forest Classifier
    </span>
    <span style="background:var(--pill-bg); color:var(--pill-color); font-size:0.73rem;
                 font-weight:500; padding:4px 12px; border-radius:20px;
                 border:1px solid var(--pill-border); font-family:'DM Sans',sans-serif;">
      1M registros
    </span>
    <span style="background:var(--pill-bg); color:var(--pill-color); font-size:0.73rem;
                 font-weight:500; padding:4px 12px; border-radius:20px;
                 border:1px solid var(--pill-border); font-family:'DM Sans',sans-serif;">
      3 Classes: Baixo · Médio · Alto
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

with st.spinner("Carregando modelo..."):
    model, le_gender, feat_imp_df, thresholds, error_msg = load_artifacts()

if error_msg:
    st.error(error_msg)
    st.stop()

if model is None:
    st.error("Não foi possível carregar ou treinar o modelo.")
    st.stop()

if 'prediction' not in st.session_state:
    st.session_state.prediction = None
if 'history' not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    if st.session_state.prediction is not None:
        st.divider()
        st.header("Dados do Indivíduo")
        st.markdown("Ajuste os campos e clique em **Prever Burnout**.")

        st.subheader("Dados Pessoais")
        age           = st.slider("Idade", 17, 40, 22)
        gender        = st.selectbox("Gênero", ["Masculino", "Feminino", "Outro"])
        academic_year = st.selectbox("Ano Acadêmico", [1, 2, 3, 4])

        st.subheader("Acadêmico")
        study_hours   = st.slider("Horas de Estudo / dia", 0.0, 16.0, 6.0, 0.5)
        exam_pressure = st.slider("Pressão de Provas (0–10)", 0.0, 10.0, 5.0, 0.5,
                                  help="0 = nenhuma pressão · 10 = pressão extrema")
        acad_perf     = st.slider("Desempenho Acadêmico (0–100)", 0.0, 100.0, 65.0, 1.0)

        st.subheader("Saúde Mental")
        stress     = st.slider("Nível de Estresse (0–10)", 0.0, 10.0, 5.0, 0.5,
                               help="0 = sem estresse · 10 = estresse extremo")
        anxiety    = st.slider("Score de Ansiedade (0–10)", 0.0, 10.0, 4.0, 0.5,
                               help="0 = sem ansiedade · 10 = ansiedade severa")
        depression = st.slider("Score de Depressão (0–10)", 0.0, 10.0, 3.0, 0.5,
                               help="0 = sem sintomas · 10 = sintomas severos")

        st.subheader("Hábitos e Estilo de Vida")
        sleep    = st.slider("Horas de Sono / dia", 2.0, 12.0, 7.0, 0.5)
        physical = st.slider("Atividade Física (0–10)", 0.0, 10.0, 4.0, 0.5,
                             help="0 = sedentário · 10 = muito ativo")
        social   = st.slider("Suporte Social (0–10)", 0.0, 10.0, 5.0, 0.5,
                             help="0 = sem suporte · 10 = forte rede de apoio")
        screen   = st.slider("Tempo de Tela / dia (h)", 0.0, 16.0, 5.0, 0.5)
        internet = st.slider("Uso de Internet / dia (h)", 0.0, 16.0, 5.0, 0.5)

        st.subheader("Fatores Externos")
        fin_stress = st.slider("Estresse Financeiro (0–10)", 0.0, 10.0, 4.0, 0.5,
                               help="0 = sem preocupações · 10 = crise financeira severa")
        fam_exp    = st.slider("Expectativa Familiar (0–10)", 0.0, 10.0, 5.0, 0.5,
                               help="0 = sem pressão · 10 = expectativa muito alta")
        dropout    = st.slider("Risco de Evasão (0–10)", 0.0, 10.0, 2.0, 0.5,
                               help="0 = sem risco · 10 = muito propenso a evadir")

        predict_btn = st.button("Prever Burnout", type="primary", width='stretch')

if st.session_state.prediction is None:
    st.markdown("### Dados do Indivíduo")
    st.caption("Preencha os campos abaixo e clique em **Prever Burnout** para ver o resultado.")
    st.markdown("")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Dados Pessoais**")
        age           = st.slider("Idade", 17, 40, 22)
        gender        = st.selectbox("Gênero", ["Masculino", "Feminino", "Outro"])
        academic_year = st.selectbox("Ano Acadêmico", [1, 2, 3, 4])

        st.markdown("**Acadêmico**")
        study_hours   = st.slider("Horas de Estudo / dia", 0.0, 16.0, 6.0, 0.5)
        exam_pressure = st.slider("Pressão de Provas (0–10)", 0.0, 10.0, 5.0, 0.5,
                                  help="0 = nenhuma pressão · 10 = pressão extrema")
        acad_perf     = st.slider("Desempenho Acadêmico (0–100)", 0.0, 100.0, 65.0, 1.0)

    with col2:
        st.markdown("**Saúde Mental**")
        stress     = st.slider("Nível de Estresse (0–10)", 0.0, 10.0, 5.0, 0.5,
                               help="0 = sem estresse · 10 = estresse extremo")
        anxiety    = st.slider("Score de Ansiedade (0–10)", 0.0, 10.0, 4.0, 0.5,
                               help="0 = sem ansiedade · 10 = ansiedade severa")
        depression = st.slider("Score de Depressão (0–10)", 0.0, 10.0, 3.0, 0.5,
                               help="0 = sem sintomas · 10 = sintomas severos")

    with col3:
        st.markdown("**Hábitos e Estilo de Vida**")
        sleep    = st.slider("Horas de Sono / dia", 2.0, 12.0, 7.0, 0.5)
        physical = st.slider("Atividade Física (0–10)", 0.0, 10.0, 4.0, 0.5,
                             help="0 = sedentário · 10 = muito ativo")
        social   = st.slider("Suporte Social (0–10)", 0.0, 10.0, 5.0, 0.5,
                             help="0 = sem suporte · 10 = forte rede de apoio")
        screen   = st.slider("Tempo de Tela / dia (h)", 0.0, 16.0, 5.0, 0.5)
        internet = st.slider("Uso de Internet / dia (h)", 0.0, 16.0, 5.0, 0.5)

        st.markdown("**Fatores Externos**")
        fin_stress = st.slider("Estresse Financeiro (0–10)", 0.0, 10.0, 4.0, 0.5,
                               help="0 = sem preocupações · 10 = crise financeira severa")
        fam_exp    = st.slider("Expectativa Familiar (0–10)", 0.0, 10.0, 5.0, 0.5,
                               help="0 = sem pressão · 10 = expectativa muito alta")
        dropout    = st.slider("Risco de Evasão (0–10)", 0.0, 10.0, 2.0, 0.5,
                               help="0 = sem risco · 10 = muito propenso a evadir")

    st.markdown("")
    predict_btn = st.button("Prever Burnout", type="primary", width='stretch')
    st.divider()

_GENDER_PT_TO_EN = {"Masculino": "Male", "Feminino": "Female", "Outro": "Other"}

if locals().get('predict_btn', False):
    gender_enc = le_gender.transform([_GENDER_PT_TO_EN[gender]])[0]
    input_data = {
        'age': age, 'gender_enc': gender_enc, 'academic_year': academic_year,
        'study_hours_per_day': study_hours, 'exam_pressure': exam_pressure,
        'academic_performance': acad_perf, 'stress_level': stress,
        'anxiety_score': anxiety, 'depression_score': depression,
        'sleep_hours': sleep, 'physical_activity': physical,
        'social_support': social, 'screen_time': screen,
        'internet_usage': internet, 'financial_stress': fin_stress,
        'family_expectation': fam_exp, 'dropout_risk': dropout
    }

    X_new  = pd.DataFrame([input_data])[FEATURES_ENC]
    pred   = model.predict(X_new)[0]
    probas = model.predict_proba(X_new)[0]
    nivel  = label_map[pred]

    human_inputs = {
        'age': age, 'gender': gender, 'academic_year': academic_year,
        'study_hours_per_day': study_hours, 'exam_pressure': exam_pressure,
        'academic_performance': acad_perf, 'stress_level': stress,
        'anxiety_score': anxiety, 'depression_score': depression,
        'sleep_hours': sleep, 'physical_activity': physical,
        'social_support': social, 'screen_time': screen,
        'internet_usage': internet, 'financial_stress': fin_stress,
        'family_expectation': fam_exp, 'dropout_risk': dropout
    }

    st.session_state.history.append({
        'ts':          datetime.datetime.now().strftime('%H:%M:%S'),
        'nivel':       nivel,
        'confianca':   round(probas[pred] * 100, 1),
        'stress':      stress,
        'anxiety':     anxiety,
        'depression':  depression,
        'sleep':       sleep,
        'physical':    physical,
        'social':      social,
        'exam_pressure': exam_pressure,
        'study_hours': study_hours,
        'fin_stress':  fin_stress,
    })
    if len(st.session_state.history) > 10:
        st.session_state.history = st.session_state.history[-10:]

    st.session_state.prediction = {
        'nivel': nivel, 'pred': pred, 'probas': probas, 'inputs': human_inputs
    }
    st.rerun()

if st.session_state.prediction is not None:
    p         = st.session_state.prediction
    nivel     = p['nivel']
    pred      = p['pred']
    probas    = p['probas']
    inputs    = p['inputs']
    confianca = probas[pred] * 100

    top10_feats = feat_imp_df.head(10)['feature'].tolist()

    _gauge_steps = [
        {"range": [0, 40],   "color": _GRID},
        {"range": [40, 70],  "color": _GRID},
        {"range": [70, 100], "color": _GRID},
    ]
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confianca,
        title={"text": f"Confiança — {nivel}", "font": {"size": 13, "color": _TICK}},
        number={"suffix": "%", "font": {"size": 28, "color": CHART_FONT['color']}},
        gauge={
            "axis": {"range": [0, 100], "tickcolor": _TICK, "tickfont": {"color": _TICK}},
            "bar": {"color": color_map[nivel]},
            "bgcolor": CHART_BASE['paper_bgcolor'],
            "bordercolor": _GRID,
            "steps": _gauge_steps,
        }
    ))
    fig_gauge.update_layout(height=260, **CHART_BASE)

    bar_labels = ['Baixo', 'Médio', 'Alto']
    bar_values = probas * 100
    fig_bar = go.Figure(go.Bar(
        x=bar_labels,
        y=bar_values,
        marker_color=['#22c55e', '#f97316', '#ef4444'],
        marker_line_width=0,
        width=0.4,
    ))
    for lbl, val in zip(bar_labels, bar_values):
        fig_bar.add_annotation(
            x=lbl, y=val / 2,
            text=f"<b>{val:.1f}%</b>",
            showarrow=False,
            font=dict(color='#ffffff', size=13, family='DM Sans, sans-serif'),
            xref='x', yref='y',
        )
    fig_bar.update_layout(
        yaxis_title='Probabilidade (%)',
        yaxis_range=[0, 105],
        height=320,
        xaxis=dict(tickfont=dict(color=_TICK)),
        yaxis=dict(tickfont=dict(color=_TICK), gridcolor=_GRID),
        margin=dict(l=40, r=40, t=20, b=20),
        **CHART_BASE
    )

    top_labels  = [FEATURE_LABELS.get(f, f) for f in top10_feats]
    individuo_v = [inputs.get(f, 0) for f in top10_feats]
    mediana_v   = [DATASET_MEDIANS.get(f, 0) for f in top10_feats]
    _lvl_rgb    = ",".join(str(int(color_map[nivel].lstrip("#")[i:i+2], 16)) for i in (0, 2, 4))

    fig_radar = go.Figure()
    fig_radar.add_trace(go.Scatterpolar(
        r=mediana_v + [mediana_v[0]],
        theta=top_labels + [top_labels[0]],
        fill='toself', name='Mediana Geral',
        line_color=_MEDIAN_C, opacity=0.55,
        fillcolor='rgba(96,165,250,0.1)'
    ))
    fig_radar.add_trace(go.Scatterpolar(
        r=individuo_v + [individuo_v[0]],
        theta=top_labels + [top_labels[0]],
        fill='toself', name=f'Indivíduo ({nivel})',
        line_color=color_map[nivel], opacity=0.75,
        fillcolor=f'rgba({_lvl_rgb},0.12)'
    ))
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, gridcolor=_GRID, tickfont=dict(color=_TICK)),
            angularaxis=dict(tickfont=dict(color=_TICK)),
            bgcolor=CHART_BASE['paper_bgcolor'],
        ),
        height=420,
        title=dict(text='Radar: Top 10 Variáveis', font=dict(color=_TICK, size=13)),
        legend=dict(orientation='h', yanchor='bottom', y=-0.22, font=dict(color=_TICK)),
        **CHART_BASE
    )

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Mediana Geral', x=top_labels, y=mediana_v,
        marker_color=_MEDIAN_C, opacity=0.8, marker_line_width=0,
    ))
    fig_comp.add_trace(go.Bar(
        name=f'Indivíduo ({nivel})', x=top_labels, y=individuo_v,
        marker_color=color_map[nivel], opacity=0.9, marker_line_width=0,
    ))
    fig_comp.update_layout(
        barmode='group', height=420,
        xaxis=dict(tickangle=-30, tickfont=dict(color=_TICK), gridcolor=_GRID),
        yaxis=dict(tickfont=dict(color=_TICK), gridcolor=_GRID),
        title=dict(text='Barras: Top 10 Variáveis', font=dict(color=_TICK, size=13)),
        legend=dict(font=dict(color=_TICK)),
        margin=dict(t=50, b=60),
        **CHART_BASE
    )

    feat_imp_display = feat_imp_df.copy()
    feat_imp_display['feature_label'] = feat_imp_display['feature'].map(
        lambda x: FEATURE_LABELS.get(x, x)
    )
    _imp_scale = [[0, '#fef3e2'], [0.4, '#f5c07a'], [1, '#b86b0a']]
    fig_imp = px.bar(
        feat_imp_display,
        x='importance', y='feature_label',
        orientation='h',
        color='importance',
        color_continuous_scale=_imp_scale,
        labels={'importance': 'Importância (Gini)', 'feature_label': 'Variável'},
        title='Importância Global das Features — Random Forest'
    )
    fig_imp.update_layout(
        height=550,
        yaxis={'categoryorder': 'total ascending', 'tickfont': dict(color=_TICK), 'gridcolor': _GRID},
        xaxis={'tickfont': dict(color=_TICK), 'gridcolor': _GRID},
        title=dict(font=dict(color=_TICK, size=13)),
        coloraxis_showscale=False,
        **CHART_BASE
    )

    tab1, tab2, tab3 = st.tabs([
        "Resultado",
        "Comparação de Perfil",
        "Histórico",
    ])

    with tab1:
        col1, col3 = st.columns([2, 1.4])

        with col1:
            st.subheader("Resultado da Predição")
            st.markdown(f"""
            <div class="{css_map[nivel]}">
                <h2 style="margin:0; font-size:1.5rem; color:var(--text-primary); font-family:'DM Serif Display',Georgia,serif; font-weight:400;">
                  Burnout: <span style="color:{color_map[nivel]}; font-weight:400;">{nivel.upper()}</span>
                </h2>
                <p style="margin:6px 0 0 0; color:var(--text-secondary); font-family:'DM Sans',sans-serif; font-size:0.88rem;">
                  Confiança do modelo: <b style="color:var(--text-primary);">{confianca:.1f}%</b>
                </p>
            </div>
            """, unsafe_allow_html=True)

            st.plotly_chart(fig_gauge, width='stretch')

            dashboard_html = generate_dashboard_html(
                nivel, confianca, inputs, probas,
                fig_gauge, fig_radar, fig_comp
            )
            st.download_button(
                label="Baixar Dashboard (HTML)",
                data=dashboard_html.encode('utf-8'),
                file_name=f"burnout_dashboard_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                mime='text/html',
                width='stretch'
            )

        with col3:
            st.subheader("Métricas do Indivíduo")
            st.metric("Nível de Burnout",  nivel)
            st.metric("Estresse",          f"{inputs['stress_level']:.1f}/10")
            st.metric("Ansiedade",         f"{inputs['anxiety_score']:.1f}/10")
            st.metric("Depressão",         f"{inputs['depression_score']:.1f}/10")
            st.metric("Sono (h/dia)",      f"{inputs['sleep_hours']:.1f}h")
            st.metric("Confiança Modelo",  f"{confianca:.1f}%")

        st.subheader("Probabilidades por Classe")
        st.plotly_chart(fig_bar, width='stretch')

    with tab2:
        st.subheader("Perfil do Indivíduo vs Perfil Médio Geral")
        st.caption("Comparação entre seus valores e a mediana do dataset nas 10 variáveis mais relevantes para o modelo.")

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            st.plotly_chart(fig_radar, width='stretch')
        with col_r2:
            st.plotly_chart(fig_comp, width='stretch')

        st.divider()
        st.subheader("Importância das Variáveis")
        st.caption(
            "Importância global das features calculada pelo modelo Random Forest treinado em 1 milhão de registros. "
            "Reflete o comportamento geral do modelo — não é específica de nenhum resultado individual."
        )

        col_a, col_b = st.columns([2, 1])
        with col_a:
            st.plotly_chart(fig_imp, width='stretch')
        with col_b:
            st.markdown("**Ranking Completo**")
            rank_df = feat_imp_display[['feature_label', 'importance']].copy()
            rank_df.columns = ['Variável', 'Importância']
            rank_df['Importância'] = rank_df['Importância'].map(lambda x: f"{x:.4f}")
            rank_df = rank_df.reset_index(drop=True)
            rank_df.index = rank_df.index + 1
            st.dataframe(rank_df, width='stretch', height=520)

    with tab3:
        hist = st.session_state.history

        if not hist:
            st.markdown("""
            <div style="
                text-align:center; padding:80px 40px;
                background:var(--bg-card); border-radius:16px;
                border:1px dashed var(--border-strong); margin:8px 0 24px;
            ">
                <div style="
                    width:52px;height:52px;border-radius:50%;
                    border:2px dashed var(--border-strong);
                    display:inline-flex;align-items:center;justify-content:center;
                    margin-bottom:20px;opacity:0.5;
                ">
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" stroke-width="1.5">
                        <circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>
                    </svg>
                </div>
                <h3 style="
                    font-family:'DM Serif Display',Georgia,serif; font-weight:400;
                    font-size:1.25rem; color:var(--text-primary); margin:0 0 10px;
                ">Nenhuma predição ainda</h3>
                <p style="
                    color:var(--text-muted); font-size:0.85rem; margin:0 auto;
                    max-width:340px; line-height:1.7; font-family:'DM Sans',sans-serif;
                ">
                    Preencha os dados do indivíduo e clique em
                    <strong style="color:var(--accent);">Prever Burnout</strong>
                    para iniciar o histórico da sessão.<br>
                    <span style="font-size:0.78rem; opacity:0.7;">
                        Até 10 predições são armazenadas por sessão.
                    </span>
                </p>
            </div>
            """, unsafe_allow_html=True)

        else:
            n = len(hist)
            level_counts = {}
            for h in hist:
                level_counts[h['nivel']] = level_counts.get(h['nivel'], 0) + 1

            avg_stress = sum(h['stress'] for h in hist) / n
            avg_sleep  = sum(h['sleep']  for h in hist) / n
            avg_anxiety = sum(h['anxiety'] for h in hist) / n

            level_pills = ""
            for lv, clr in [('Alto', '#ef4444'), ('Médio', '#f97316'), ('Baixo', '#22c55e')]:
                cnt = level_counts.get(lv, 0)
                if cnt:
                    level_pills += f"""
                    <span style="
                        background:{clr}18;color:{clr};border:1px solid {clr}38;
                        border-radius:20px;padding:3px 13px;font-size:0.72rem;font-weight:700;
                        font-family:'DM Sans',sans-serif;letter-spacing:.04em;
                    ">{lv} &times; {cnt}</span>"""

            summary = f"""
            <div style="
                background:#ffffff;border:1px solid rgba(160,120,50,0.1);
                border-radius:14px;padding:18px 26px;margin-bottom:20px;
                display:flex;align-items:center;gap:20px;flex-wrap:wrap;
            ">
                <div>
                    <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                                letter-spacing:.09em;color:#a0907e;
                                font-family:'DM Sans',sans-serif;margin-bottom:3px;">Esta sessão</div>
                    <div style="font-family:'DM Serif Display',serif;font-size:1.5rem;
                                color:#1c1916;font-weight:400;line-height:1;">
                        {n}<span style="font-size:0.85rem;color:#a0907e;
                        font-family:'DM Sans',sans-serif;margin-left:5px;">
                        prediç{'ões' if n!=1 else 'ão'}</span>
                    </div>
                </div>
                <div style="width:1px;height:38px;background:rgba(160,120,50,0.1);flex-shrink:0;"></div>
                <div style="display:flex;gap:7px;flex-wrap:wrap;align-items:center;">{level_pills}</div>
                <div style="width:1px;height:38px;background:rgba(160,120,50,0.1);flex-shrink:0;"></div>
                <div style="display:flex;gap:28px;margin-left:auto;flex-wrap:wrap;">
                    <div>
                        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:.09em;color:#a0907e;
                                    font-family:'DM Sans',sans-serif;margin-bottom:3px;">Estresse médio</div>
                        <div style="font-size:1rem;font-weight:700;color:#1c1916;
                                    font-family:'DM Sans',sans-serif;">{avg_stress:.1f}<span style="font-size:0.72rem;color:#a0907e;font-weight:400;"> /10</span></div>
                    </div>
                    <div>
                        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:.09em;color:#a0907e;
                                    font-family:'DM Sans',sans-serif;margin-bottom:3px;">Ansiedade média</div>
                        <div style="font-size:1rem;font-weight:700;color:#1c1916;
                                    font-family:'DM Sans',sans-serif;">{avg_anxiety:.1f}<span style="font-size:0.72rem;color:#a0907e;font-weight:400;"> /10</span></div>
                    </div>
                    <div>
                        <div style="font-size:0.65rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:.09em;color:#a0907e;
                                    font-family:'DM Sans',sans-serif;margin-bottom:3px;">Sono médio</div>
                        <div style="font-size:1rem;font-weight:700;color:#1c1916;
                                    font-family:'DM Sans',sans-serif;">{avg_sleep:.1f}<span style="font-size:0.72rem;color:#a0907e;font-weight:400;"> h</span></div>
                    </div>
                </div>
            </div>"""

            cards = ""
            reversed_pairs = list(reversed(list(enumerate(hist))))
            for display_i, (orig_idx, entry) in enumerate(reversed_pairs):
                entry_num = orig_idx + 1
                lv    = entry['nivel']
                clr   = color_map[lv]
                conf  = entry['confianca']
                prev  = hist[orig_idx - 1] if orig_idx > 0 else None
                is_last = display_i == len(reversed_pairs) - 1

                metrics_def = [
                    ('Estresse',    entry['stress'],      '/10'),
                    ('Ansiedade',   entry['anxiety'],     '/10'),
                    ('Depressão',   entry['depression'],  '/10'),
                    ('Sono',        entry['sleep'],       'h'),
                    ('Ativ. Física',entry['physical'],    '/10'),
                    ('Suporte',     entry['social'],      '/10'),
                ]
                metrics_html = ""
                for mlabel, mval, munit in metrics_def:
                    metrics_html += f"""
                    <div style="text-align:center;">
                        <div style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                                    letter-spacing:.07em;color:#a0907e;
                                    font-family:'DM Sans',sans-serif;margin-bottom:5px;">{mlabel}</div>
                        <div style="font-size:1.05rem;font-weight:700;color:#1c1916;
                                    font-family:'DM Sans',sans-serif;line-height:1;">
                            {mval:.1f}<span style="font-size:0.68rem;color:#a0907e;
                            font-weight:400;margin-left:1px;">{munit}</span>
                        </div>
                    </div>"""

                conf_bar = f"""
                <div style="margin:14px 0 16px;">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:6px;">
                        <span style="font-size:0.63rem;font-weight:700;text-transform:uppercase;
                                     letter-spacing:.09em;color:#a0907e;
                                     font-family:'DM Sans',sans-serif;">Confiança do modelo</span>
                        <span style="font-size:0.85rem;font-weight:700;color:{clr};
                                     font-family:'DM Sans',sans-serif;">{conf:.1f}%</span>
                    </div>
                    <div style="height:5px;background:rgba(160,120,50,0.1);border-radius:5px;overflow:hidden;">
                        <div style="height:100%;width:{conf:.1f}%;background:{clr};
                                    border-radius:5px;"></div>
                    </div>
                </div>"""

                delta_html = ""
                if prev:
                    delta_defs = [
                        ('stress',    'Estresse',     False),
                        ('anxiety',   'Ansiedade',    False),
                        ('depression','Depressão',    False),
                        ('sleep',     'Sono',         True),
                        ('physical',  'Ativ. Física', True),
                        ('social',    'Suporte',      True),
                        ('exam_pressure', 'Pr. Provas', False),
                        ('fin_stress', 'Est. Fin.',   False),
                    ]
                    chips = ""
                    for key, dlabel, more_is_good in delta_defs:
                        if key in entry and key in prev:
                            diff = entry[key] - prev[key]
                            if abs(diff) >= 0.4:
                                sign = "+" if diff > 0 else ""
                                improved = (diff > 0 and more_is_good) or (diff < 0 and not more_is_good)
                                d_clr = '#22c55e' if improved else '#ef4444'
                                arrow = '↓' if diff < 0 else '↑'
                                chips += f"""
                                <span style="
                                    background:{d_clr}12;color:{d_clr};
                                    border:1px solid {d_clr}30;border-radius:6px;
                                    padding:2px 9px;font-size:0.7rem;font-weight:700;
                                    font-family:'DM Sans',sans-serif;white-space:nowrap;
                                ">{dlabel} {arrow} {sign}{diff:.1f}</span>"""

                    if prev['nivel'] != lv:
                        prev_clr = color_map[prev['nivel']]
                        chips = f"""
                        <span style="
                            background:#f2ede4;color:#a0907e;
                            border:1px solid rgba(160,120,50,0.1);border-radius:6px;
                            padding:2px 9px;font-size:0.7rem;font-weight:600;
                            font-family:'DM Sans',sans-serif;
                        "><span style='color:{prev_clr};font-weight:700;'>{prev['nivel']}</span>
                          &nbsp;→&nbsp;
                          <span style='color:{clr};font-weight:700;'>{lv}</span></span>
                        """ + chips

                    if chips:
                        delta_html = f"""
                        <div style="
                            margin-top:14px;padding-top:13px;
                            border-top:1px solid rgba(160,120,50,0.1);
                            display:flex;gap:6px;flex-wrap:wrap;align-items:center;
                        ">
                            <span style="font-size:0.62rem;font-weight:700;text-transform:uppercase;
                                         letter-spacing:.07em;color:#a0907e;
                                         font-family:'DM Sans',sans-serif;margin-right:2px;
                                         flex-shrink:0;">Δ vs anterior</span>
                            {chips}
                        </div>"""

                connector = "" if is_last else """
                <div style="display:flex;justify-content:flex-start;
                            padding-left:26px;height:18px;">
                    <div style="width:1px;height:100%;background:rgba(160,120,50,0.1);"></div>
                </div>"""

                cards += f"""
                <div style="
                    background:#ffffff;border:1px solid rgba(160,120,50,0.1);
                    border-left:4px solid {clr};border-radius:12px;
                    padding:20px 24px;
                ">
                    <div style="display:flex;justify-content:space-between;
                                align-items:center;margin-bottom:2px;">
                        <div style="display:flex;align-items:center;gap:10px;">
                            <span style="
                                background:#f2ede4;color:#a0907e;
                                font-size:0.66rem;font-weight:700;padding:3px 10px;
                                border-radius:20px;font-family:'DM Sans',sans-serif;
                                letter-spacing:.06em;text-transform:uppercase;
                            ">#{entry_num}</span>
                            <span style="font-size:0.78rem;color:#a0907e;
                                         font-family:'DM Sans',sans-serif;">
                                {entry['ts']}
                            </span>
                        </div>
                        <span style="
                            background:{clr}18;color:{clr};border:1px solid {clr}40;
                            font-size:0.72rem;font-weight:700;padding:4px 15px;
                            border-radius:20px;font-family:'DM Sans',sans-serif;
                            letter-spacing:.08em;text-transform:uppercase;
                        ">{lv}</span>
                    </div>
                    {conf_bar}
                    <div style="display:grid;grid-template-columns:repeat(6,1fr);gap:16px;">
                        {metrics_html}
                    </div>
                    {delta_html}
                </div>
                {connector}"""

            _html = (summary + f'<div style="display:flex;flex-direction:column;">{cards}</div>').replace('\n', ' ')
            st.markdown(_html, unsafe_allow_html=True)

            st.markdown("")
            if st.button("Limpar Histórico", key="clear_hist"):
                st.session_state.history = []
                st.rerun()

st.divider()
st.markdown("""
<div style="text-align:center; padding:8px 0 20px">
  <p style="color:var(--text-muted); font-size:0.75rem; margin:0; letter-spacing:0.04em; text-transform:uppercase; font-family:'DM Sans',sans-serif;">
    Burnout Prediction App &nbsp;·&nbsp; Random Forest &nbsp;·&nbsp; Scikit-learn &nbsp;·&nbsp; Plotly &nbsp;·&nbsp; Streamlit
  </p>
  <p style="color:var(--border-strong); font-size:0.7rem; margin:5px 0 0; font-style:italic; font-family:'DM Sans',sans-serif;">
    Para fins educacionais e de pesquisa. Não substitui avaliação profissional.
  </p>
</div>
""", unsafe_allow_html=True)
