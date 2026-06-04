import streamlit as st

st.set_page_config(
    page_title="MathWork Labs - Painel de Otimização",
    page_icon="🔬",
    layout="wide"
)

st.markdown("""
<style>
:root {
  --bg: #0e1117;
  --border: rgba(255,255,255,0.1);
  --muted: rgba(229,231,235,0.70);
  --muted2: rgba(229,231,235,0.40);
  --accent: #FF4B4B;
  --accent2: #1E90FF;
}

.stApp { background-color: var(--bg); }

.hero-section {
    padding: 4rem 2rem;
    background: radial-gradient(circle at top left, rgba(255,75,75,0.1), transparent),
                radial-gradient(circle at bottom right, rgba(30,144,255,0.1), transparent);
    border-radius: 24px;
    border: 1px solid var(--border);
    margin-bottom: 3rem;
    text-align: center;
}

.title-text {
    font-size: 4rem;
    font-weight: 800;
    letter-spacing: -2px;
    margin-bottom: 0.5rem;
    color: #FFFFFF;
}

.feature-card {
    background: rgba(255,255,255,0.03);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 24px;
    height: 100%;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.feature-card:hover {
    border-color: var(--accent);
    transform: translateY(-8px);
    background: rgba(255,255,255,0.05);
    box-shadow: 0 10px 30px rgba(0,0,0,0.4);
}

.card-icon { font-size: 2rem; margin-bottom: 15px; }
.card-title { color: #FFFFFF; font-size: 1.3rem; font-weight: 700; margin-bottom: 12px; }

.info-box {
    background: rgba(30,144,255,0.05);
    border-left: 4px solid var(--accent2);
    padding: 20px;
    border-radius: 0 12px 12px 0;
}

.contact-card {
    background: linear-gradient(145deg, rgba(255,255,255,0.05), rgba(255,255,255,0.01));
    padding: 25px;
    border-radius: 16px;
    border: 1px solid var(--border);
}

.hr {
    border: none;
    border-top: 1px solid var(--border);
    margin: 3rem 0;
}

.footer { 
    text-align: center; 
    color: var(--muted2); 
    margin-top: 5rem; 
    padding-bottom: 3rem;
    font-size: 0.9rem; 
}

code { color: var(--accent) !important; }
.badge {
    display: inline-block;
    padding: 2px 10px;
    border-radius: 5px;
    background: rgba(255,255,255,0.1);
    font-size: 0.75rem;
    margin-bottom: 10px;
}
</style>
""", unsafe_allow_html=True)

# HERO
st.markdown("""
    <div class="hero-section">
        <div class="badge">CORE ENGINE v1.1.0 - LIVE</div>
        <h1 class="title-text">MATHWORK LABS</h1>
        <p style="color: var(--muted); font-size: 1.3rem; max-width: 800px; margin: 0 auto; line-height: 1.6;">
            Infraestrutura de computação numérica avançada e Pesquisa Operacional focada em 
            modelagem matemática estruturada, algoritmos de otimização e análise de convergência.
        </p>
    </div>
""", unsafe_allow_html=True)

# CARDS
st.markdown("### 🛠️ Módulos Ativos de Engenharia")
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
        <div class="feature-card">
            <div class="card-icon">📐</div>
            <div class="card-title">Análise de Integrais</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Quadratura numérica via Riemann, Simpson e Trapézios. Abordagem desacoplada com diagnósticos de erro assintótico em tempo real.
            </p>
        </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
        <div class="feature-card">
            <div class="card-icon">🧬</div>
            <div class="card-title">Algoritmos Genéticos</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Modelos heurísticos aplicados à Pesquisa Operacional e análise de risco, utilizando operadores customizados de seleção, cruzamento e mutação.
            </p>
        </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
        <div class="feature-card">
            <div class="card-icon">📊</div>
            <div class="card-title">Análise de Erro Log-Log</div>
            <p style="color: var(--muted); font-size: 0.95rem;">
                Verificação rigorosa da ordem de convergência observada (slope) para validação empírica de estabilidade de algoritmos.
            </p>
        </div>
    """, unsafe_allow_html=True)

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# ROADMAP E SINTAXE
col_road, col_syntax = st.columns([1, 1], gap="large")

with col_road:
    st.markdown("### 🚀 Roadmap de Desenvolvimento")
    st.markdown("""
        - ✅ **v1.0:** Motor de Integração Numérica Detalhado
        - ✅ **v1.1:** Estrutura Monorepo e Módulo de Algoritmos Genéticos
        - 🔄 **v1.2:** Simulações Estocásticas e Métodos de Monte Carlo
        - 📅 **v1.3:** Otimização Linear e Algoritmo Simplex
        - 📅 **v1.4:** Sistemas Dinâmicos e Equações Diferenciais Aplicadas
    """)

with col_syntax:
    st.markdown("### ⌨️ Escopo de Computação Científica")
    st.markdown("O ecossistema utiliza processamento simbólico SymPy com conversão automatizada para matrizes NumPy de alta performance:")
    st.code("""
# Potência matricial: x**2 (evite o operador ^)
# Constantes analíticas: pi, E
# Funções nativas: exp(x), log(x), sin(x), cos(x)
# Operações modulares: Abs(x), sqrt(x)
    """, language="python")

st.markdown("<div class='hr'></div>", unsafe_allow_html=True)

# SOBRE O PROJETO
inf_left, inf_right = st.columns([1.5, 1])

with inf_left:
    st.markdown("### 🔍 Diretriz de Desenvolvimento")
    st.write("""
        O **MathWork Labs** funciona como um laboratório de Pesquisa e Desenvolvimento focado em descolar a matemática avançada
        de exercícios puramente abstratos, aplicando-a diretamente em problemas estruturais de engenharia, logística e tomada de decisão.
        O código prioriza arquiteturas limpas, separando rigorosamente motores matemáticos puros de interfaces visuais.
    """)
    st.markdown("""
        <div class="info-box">
            <strong>Garantia de Rigor:</strong> Os métodos numéricos e heurísticos expostos nesta suíte são validados contra ferramentas consolidadas da indústria (como SciPy) para assegurar estabilidade e convergência matemática.
        </div>
    """, unsafe_allow_html=True)

with inf_right:
    st.markdown("### ✉️ Engenharia e Contato")
    st.markdown("""
        <div class="contact-card">
            <p style="margin-bottom: 10px; font-size: 1.1rem;"><strong>Fellipe Almässy</strong></p>
            <p style="margin-bottom: 8px; font-size: 0.95rem; color: var(--muted);">📧 <a href="mailto:fealmassy@gmail.com" style="color: var(--accent2); text-decoration:none;">fealmassy@gmail.com</a></p>
            <p style="margin-bottom: 8px; font-size: 0.95rem; color: var(--muted);">📱 +55 (11) 91258-3939</p>
            <p style="margin-bottom: 0px; font-size: 0.85rem; color: var(--muted2);">📍 São Paulo - SP, Brasil</p>
        </div>
    """, unsafe_allow_html=True)

# RODAPÉ
st.markdown("""
    <div class='footer'>
        <strong>MathWork Labs v1.1.0</strong> — Otimização Computacional<br>
        Fellipe Almässy • 2026
    </div>
""", unsafe_allow_html=True)

st.sidebar.title("Navegação")
st.sidebar.info("Acesse os módulos ativos no menu lateral para iniciar as simulações técnicas.")
st.sidebar.markdown("---")
st.sidebar.caption("Ambiente: SymPy 1.12 | NumPy 1.26")
