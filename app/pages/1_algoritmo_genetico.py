import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Força o Python a enxergar a pasta raiz 'src' a partir da subpasta de páginas
raiz = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if raiz not in sys.path:
    sys.path.append(raiz)

from src.genetic_algorithm.core import (
    processar_base_dados, criar_cromossomos, calcular_fitness,
    fitness_percentual, selecionar_pais_roleta, cruzar_pais,
    mutar, atualizar_populacao, prever_novo_cliente
)

st.title("🧬 Otimização Heurística: Algoritmo Genético")
st.caption("Modelagem preditiva de risco de crédito baseada em evolução biológica simulada.")

# Estilização básica inline para consistência visual
st.markdown("""
<style>
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 10px;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR: PARÂMETROS DO MODELO
st.sidebar.header("Configurações do Algoritmo")
qtd_cromossomos = st.sidebar.slider("Tamanho da População", 6, 50, 10, step=2)
geracoes = st.sidebar.slider("Número Máximo de Gerações", 10, 500, 100, step=10)
fitness_alvo = st.sidebar.slider("Fitness Mínimo Alvo", 0.50, 1.00, 0.92, step=0.01)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Entrada de Dados")
arquivo_excel = st.sidebar.file_uploader("Carregar Base de Clientes (.xlsx)", type=["xlsx"])

# CONTEÚDO PRINCIPAL
if arquivo_excel is not None:
    try:
        df_original = pd.read_excel(arquivo_excel)
        array_dados, array_gabarito, qtd_features, qtd_genes = processar_base_dados(df_original)
        
        st.success(f"Base carregada com sucesso! Clientes mapeados: {array_dados.shape[0]} | Variáveis analisadas (Features): {qtd_features}")
        
        # Abas de Execução e Visualização dos Dados Brutos
        tab_treino, tab_dados = st.tabs(["Treinamento do Algoritmo", "Visualização da Base"])
        
        with tab_dados:
            st.dataframe(df_original, use_container_width=True)
            
        with tab_treino:
            if st.button("Iniciar Evolução Populacional"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # Inicialização da População
                populacao = criar_cromossomos(qtd_cromossomos, qtd_genes)
                melhor_cromossomo = None
                melhor_fitness = 0.0
                historico_fitness = []

                # LOOP DE EVOLUÇÃO (Gerações)
                for g in range(geracoes):
                    fitnesses = calcular_fitness(populacao, array_dados, array_gabarito)
                    percentuais = fitness_percentual(fitnesses)

                    idx_melhor = int(np.argmax(fitnesses))
                    fitness_atual = float(fitnesses[idx_melhor])

                    if fitness_atual > melhor_fitness:
                        melhor_fitness = fitness_atual
                        melhor_cromossomo = populacao[idx_melhor].copy()

                    historico_fitness.append(melhor_fitness)
                    
                    # Atualiza a UI a cada geração
                    progress = (g + 1) / geracoes
                    progress_bar.progress(progress)
                    status_text.text(f"Geração {g+1}/{geracoes} | Melhor Fitness Atual: {melhor_fitness:.4f}")

                    if melhor_fitness >= fitness_alvo:
                        status_text.text(f"🎯 Fitness alvo de {fitness_alvo} atingido na geração {g+1}!")
                        break

                    # Ciclo Evolutivo
                    pai, mae = selecionar_pais_roleta(populacao, percentuais)
                    filho1, ... = cruzar_pais(pai, mae)
                    filho1, ... = mutar(filho1, ...)
                    populacao = atualizar_populacao(
                        populacao, fitnesses, filho1, ..., array_dados, array_gabarito
                    )

                # Salva o melhor modelo na sessão do Streamlit para usar na predição abaixo
                st.session_state["melhor_modelo"] = melhor_cromossomo
                st.session_state["qtd_features"] = qtd_features
                
                # Módulos de Resultados Graficos
                st.markdown("### Resultados Técnicos")
                c1, c2 = st.columns(2)
                c1.metric("Melhor Fitness Alcançado", f"{melhor_fitness:.5f}")
                c2.metric("Gerações Computadas", f"{len(historico_fitness)}")
                
                st.markdown("#### Curva de Convergência Heurística")
                st.line_chart(historico_fitness)
        
        # SESSÃO DE PREVISÃO INDIVIDUAL (INFERÊNCIA)
        if "melhor_modelo" in st.session_state:
            st.markdown("<hr>", unsafe_allow_html=True)
            st.markdown("### 🔮 Predição de Risco para Novo Cliente")
            st.write("Insira os parâmetros numéricos do novo cliente para calcular a tendência de crédito:")
            
            # Cria colunas dinâmicas para preenchimento de inputs com base no número de features
            col_inputs = st.columns(4)
            dados_novo_cliente = []
            
            for idx in range(st.session_state["qtd_features"]):
                nome_coluna = df_original.columns[idx + 1]
                valor_medio = float(np.mean(array_dados[:, idx]))
                
                with col_inputs[idx % 4]:
                    val = st.number_input(f"{nome_coluna}", value=valor_medio, format="%.4f", key=f"feat_{idx}")
                    dados_novo_cliente.append(val)
            
            if st.button("Executar Análise de Crédito"):
                modelo = st.session_state["melhor_modelo"]
                entrada_array = np.array(dados_novo_cliente)
                
                predicao = prever_novo_cliente(modelo, entrada_array)
                
                if predicao == 1:
                    st.success("Análise Concluída: **Cliente Classificado como ADIMPLENTE (Baixo Risco)**")
                else:
                    st.error("Análise Concluída: **Cliente Classificado como INADIMPLENTE (Alto Risco)**")
                    
    except Exception as e:
        st.error(f"Erro ao processar arquivo Excel. Verifique a formatação interna da planilha. Detalhes: {e}")
else:
    st.info("Aguardando upload da base de dados '.xlsx' na barra lateral para iniciar os ciclos computacionais.")
