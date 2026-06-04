import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import sympy as sp
import streamlit as st

# IMPORTAÇÃO DA LÓGICA DO SEU NOVO MOTOR
from src.integration.engine import (
    METODOS, ORDEM_TEORICA, parse_function, safe_eval_curve,
    compute_reference_quad, series_convergencia, estimate_observed_order
)

# ----------------------------
# 1) ESTILO (CSS)
# ----------------------------
st.markdown(
    """
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.08);
  --muted: rgba(229,231,235,0.60);
  --muted2: rgba(229,231,235,0.40);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}
div[data-testid="stMetric"]{
  background: linear-gradient(180deg, rgba(255,255,255,0.045), rgba(255,255,255,0.018));
  border: 1px solid rgba(255,255,255,0.06);
  border-radius: 14px;
  padding: 14px;
}
.hr { border: none; border-top: 1px solid var(--border); margin: 0.75rem 0 1.0rem 0; }
.small-muted { color: var(--muted); font-size: 0.92rem; }
.badge {
  display:inline-block; padding: 0.18rem 0.55rem; border-radius: 999px;
  background: rgba(255,255,255,0.06); border: 1px solid rgba(255,255,255,0.08);
  color: rgba(229,231,235,0.80); font-size: 0.82rem;
}
.footer { text-align:center; color: var(--muted2); margin-top: 14px; font-size: 0.85rem; }
.function-display { text-align: center; padding: 1.5rem 0; }
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------
# 2) WRAPPERS DE CACHE PARA O STREAMLIT
# ----------------------------
@st.cache_resource(show_spinner=False)
def cached_parse(expr_str: str):
    return parse_function(expr_str)

@st.cache_data(show_spinner=False)
def cached_eval_curve(expr_str: str, a: float, b: float):
    return safe_eval_curve(expr_str, a, b)

@st.cache_data(show_spinner=False)
def cached_ref_quad(expr_str: str, a: float, b: float):
    return compute_reference_quad(expr_str, a, b)

@st.cache_data(show_spinner=False)
def cached_convergencia(expr_str: str, a: float, b: float, nome_metodo: str, n_max: int, step: int):
    return series_convergencia(expr_str, a, b, nome_metodo, n_max, step)


# ----------------------------
# 3) FUNÇÕES DE PLOTAGEM (FRONT-END)
# ----------------------------
def make_main_plot(expr_str: str, expr: sp.Expr, f_num, a: float, b: float, n: int, show_rectangles: bool) -> go.Figure:
    h = (b - a) / n
    x_curve, y_curve = cached_eval_curve(expr_str, a, b)

    fig = go.Figure()
    mask = (x_curve >= a) & (x_curve <= b)
    
    fig.add_trace(go.Scatter(
        x=x_curve[mask], y=y_curve[mask], fill="tozeroy", name="Área (visual)",
        fillcolor="rgba(255, 75, 75, 0.10)", line=dict(color="rgba(255,255,255,0)"), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve, mode="lines", line=dict(color="rgba(255,75,75,0.18)", width=10),
        hoverinfo="skip", showlegend=False,
    ))
    fig.add_trace(go.Scatter(
        x=x_curve, y=y_curve, mode="lines", name="f(x)",
        line=dict(color="#FF4B4B", width=3), hovertemplate="x=%{x:.6f}<br>f(x)=%{y:.6f}<extra></extra>",
    ))

    if show_rectangles:
        x_left = np.linspace(a, b - h, n)
        y_left = np.array(f_num(x_left), dtype=float)
        y_left[~np.isfinite(y_left)] = np.nan
        fig.add_trace(go.Bar(
            x=x_left, y=y_left, width=h, name="Barras da Partição",
            marker=dict(color="#1E90FF", opacity=0.55, line=dict(color="rgba(255,255,255,0.35)", width=0.5)),
            hovertemplate="x=%{x:.6f}<br>altura=%{y:.6f}<extra></extra>",
        ))

    fig.add_vline(x=a, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")
    fig.add_vline(x=b, line_width=1, line_dash="dot", line_color="rgba(229,231,235,0.35)")
    fig.update_layout(template="plotly_dark", hovermode="x unified", margin=dict(l=0, r=0, t=20, b=0), legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    return fig

def make_convergence_plot(expr_str: str, a: float, b: float, nome_metodo: str, n_max: int, step: int, loglog: bool):
    ns, errs = cached_convergencia(expr_str, a, b, nome_metodo, n_max, step)
    if ns.size == 0:
        return None, None, ns, errs

    eps = 1e-300
    errs_plot = np.array(errs, dtype=float)
    if loglog: errs_plot = np.maximum(errs_plot, eps)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ns, y=errs_plot, mode="lines", name=f"Erro ({nome_metodo})", hovertemplate="n=%{x}<br>erro=%{y:.3e}<extra></extra>"))
    fig.update_layout(template="plotly_dark", margin=dict(l=0, r=0, t=50, b=0), title="Erro Absoluto vs n", xaxis_title="n (partições)", yaxis_title="Erro Absoluto")
    if loglog: fig.update_layout(xaxis_type="log", yaxis_type="log")
    
    p_obs = estimate_observed_order(ns, np.maximum(errs, eps), a, b)
    return fig, p_obs, ns, errs


# ----------------------------
# 4) ESTRUTURA PRINCIPAL (UI)
# ----------------------------
st.title("Integrais Definidas")
st.caption("Métodos Numéricos e Teoria Explicada")
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# SIDEBAR
st.sidebar.header("Controles")
exemplos = {"Suave": "x**2 * sin(x)", "Oscilatória": "sin(50*x) / (1 + x**2)", "Mod": "Abs(x)", "Exponencial": "exp(-x**2)"}
exemplo_escolhido = st.sidebar.selectbox("Exemplos rápidos", list(exemplos.keys()), index=3)

expr_str = st.sidebar.text_input("f(x) (Sintaxe SymPy)", value=exemplos[exemplo_escolhido])
colA, colB = st.sidebar.columns(2)
a = colA.number_input("a", value=-2.0, format="%.6f")
b = colB.number_input("b", value=2.0, format="%.6f")

if a == b:
    st.error("a e b não podem ser iguais.")
    st.stop()
if a > b:
    st.sidebar.warning("Invertendo limites pois a > b.")
    a, b = b, a

nome_metodo = st.sidebar.selectbox("Método principal", list(METODOS.keys()), index=2)
n = st.sidebar.slider("Quantidade de partições", 10, 4000, 400, step=10)
show_rectangles = st.sidebar.checkbox("Mostrar barras de partição", value=True)

st.sidebar.markdown("---")
show_conv = st.sidebar.checkbox("Mostrar diagnósticos de convergência", value=True)
n_max = st.sidebar.slider("N máx de convergência", 200, 12000, 4000, step=100)
step = st.sidebar.slider("Passo de convergência", 10, 400, 40, step=10)
loglog = st.sidebar.checkbox("Visão Log-log", value=True)

# PARSING E REFERÊNCIA
try:
    expr, f_num = cached_parse(expr_str)
    test = f_num(np.array([a, (a + b) / 2, b], dtype=float))
except Exception as e:
    st.error(f"Função inválida. Falha no parsing/avaliação: {e}")
    st.stop()

ref_val, ref_err = cached_ref_quad(expr_str, a, b)

# CÁLCULOS
h = (b - a) / n
rows = []
for nome, fn in METODOS.items():
    t0 = time.time()
    val = fn(f_num, a, b, n)
    t1 = time.time()
    err = abs(ref_val - val) if ref_val is not None else np.nan
    rows.append([nome, val, err, t1 - t0, ORDEM_TEORICA.get(nome, np.nan)])

df = pd.DataFrame(rows, columns=["Método", "Aproximação", "Erro Abs (vs quad)", "Tempo (s)", "Ordem Teórica p"]).sort_values(by=["Erro Abs (vs quad)"])
primary_val = float(df[df["Método"] == nome_metodo]["Aproximação"].iloc[0])

# EXIBIÇÃO
st.markdown("<div class='function-display'>", unsafe_allow_html=True)
st.latex(rf"\huge f(x) = {sp.latex(expr)}")
st.markdown("</div>", unsafe_allow_html=True)

m1, m2, m3, m4, m5 = st.columns([1.2, 1.1, 1.1, 1.0, 1.0])
m1.metric("Aprox. Principal", f"{primary_val:.8f}", f"dx = {h:.6g}")
if ref_val is not None:
    m2.metric("SciPy quad", f"{ref_val:.8f}", f"erro: {ref_err:.2e}")
    m3.metric("Erro Absoluto", f"{abs(ref_val - primary_val):.4e}", delta_color="inverse")
else:
    m2.metric("SciPy quad", "n/a")
    m3.metric("Erro Absoluto", "n/a")
m4.metric("n partições", f"{n}")
m5.metric("Ordem p", f"{ORDEM_TEORICA.get(nome_metodo,'—')}")

st.dataframe(df, use_container_width=True, hide_index=True)
st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ABAS DE GRÁFICOS
tab_engine, tab_diag = st.tabs(["Visão do Motor", "Diagnósticos"])
with tab_engine:
    st.plotly_chart(make_main_plot(expr_str, expr, f_num, a, b, n, show_rectangles), use_container_width=True)

with tab_diag:
    if show_conv and ref_val is not None:
        fig_c, p_obs, ns, errs = make_convergence_plot(expr_str, a, b, nome_metodo, n_max, step, loglog)
        if fig_c:
            st.plotly_chart(fig_c, use_container_width=True)
            c1, c2, c3 = st.columns(3)
            c1.metric("Ordem observada", f"{p_obs:.3f}" if p_obs else "n/a")
            c2.metric("Ordem teórica", f"{ORDEM_TEORICA.get(nome_metodo)}")
            st.dataframe(pd.DataFrame({"n": ns, "erro_abs": errs}), use_container_width=True, hide_index=True)
    else:
        st.info("Requer uma referência quad válida.")

st.markdown("<div class='footer'>MathWork Labs • Integração Numérica</div>", unsafe_allow_html=True)
