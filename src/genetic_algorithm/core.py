import numpy as np
import pandas as pd
from typing import Tuple

def processar_base_dados(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Processa o DataFrame vindo do Excel, separando features e gabarito.
    """
    dataframe_dados_clientes = df.iloc[:, 1:-1]  # ignora primeira coluna (índice) e última (gabarito)
    dataframe_gabarito       = df.iloc[:, -1]    # sempre a última coluna

    array_dados_clientes = dataframe_dados_clientes.values
    array_gabarito       = dataframe_gabarito.values

    qtd_features = array_dados_clientes.shape[1]
    qtd_genes    = qtd_features + 1  # features + bias

    return array_dados_clientes, array_gabarito, qtd_features, qtd_genes

def criar_cromossomos(qtd_cromossomos: int = 6, qtd_genes: int = 19) -> np.ndarray:
    return -1 + 2 * np.random.rand(qtd_cromossomos, qtd_genes)

def calcular_fitness(cromossomos: np.ndarray, array_dados_clientes: np.ndarray, array_gabarito: np.ndarray) -> np.ndarray:
    total_adimplentes = np.sum(array_gabarito == 1)
    total_inadimplentes = np.sum(array_gabarito == 0)
    
    # Evitar divisão por zero se a base de dados for inválida
    if total_adimplentes == 0: total_adimplentes = 1
    if total_inadimplentes == 0: total_inadimplentes = 1
    
    lista_hipotese = []

    for linha in cromossomos:
        bias = linha[0]
        genes = linha[1:]

        q = np.dot(array_dados_clientes, genes) + bias
        vetor_hipotese = np.where(q >= 0, 1, 0)

        acertos_adimplentes = np.sum((vetor_hipotese == 1) & (array_gabarito == 1))
        acertos_inadimplentes = np.sum((vetor_hipotese == 0) & (array_gabarito == 0))

        percentual_adimplente = acertos_adimplentes / total_adimplentes
        percentual_inadimplente = acertos_inadimplentes / total_inadimplentes

        fitness = percentual_adimplente * percentual_inadimplente
        lista_hipotese.append(fitness)

    return np.array(lista_hipotese)

def fitness_percentual(vetor_fitnesses: np.ndarray) -> np.ndarray:
    soma = np.sum(vetor_fitnesses)
    if soma == 0:
        return np.ones(len(vetor_fitnesses)) / len(vetor_fitnesses)
    return vetor_fitnesses / soma

def selecionar_pais_roleta(cromossomos: np.ndarray, percentual_fitnesses: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    roleta_acumulada = np.cumsum(percentual_fitnesses)
    indice_pai = min(int(np.searchsorted(roleta_acumulada, np.random.rand())), len(cromossomos) - 1)
    indice_mae = min(int(np.searchsorted(roleta_acumulada, np.random.rand())), len(cromossomos) - 1)
    return cromossomos[indice_pai], cromossomos[indice_mae]

def cruzar_pais(pai: np.ndarray, mae: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    c1 = np.random.randint(1, len(pai))
    c2 = np.random.randint(1, len(pai))
    c3 = np.random.randint(1, len(pai))

    filho1 = np.concatenate([pai[:c1], mae[c1:]])
    filho2 = np.concatenate([pai[:c2], mae[c2:]])
    filho3 = np.concatenate([pai[:c3], mae[c3:]])

    return filho1, filho2, filho3

def mutar(filho1: np.ndarray, filho2: np.ndarray, filho3: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    for filho in [filho1, filho2, filho3]:
        indice = np.random.randint(0, len(filho))
        filho[indice] = -1 + 2 * np.random.rand()
    return filho1, filho2, filho3

def atualizar_populacao(cromossomos: np.ndarray, vetor_fitnesses: np.ndarray, filho1: np.ndarray, filho2: np.ndarray, filho3: np.ndarray, array_dados_clientes: np.ndarray, array_gabarito: np.ndarray) -> np.ndarray:
    filhos = np.array([filho1, filho2, filho3])
    fitnesses_filhos = calcular_fitness(filhos, array_dados_clientes, array_gabarito)

    indices_melhores_filhos = np.argsort(fitnesses_filhos)[-2:]
    indices_piores          = np.argsort(vetor_fitnesses)[:2]

    nova_populacao = cromossomos.copy()
    for i in range(2):
        idx_pior  = int(indices_piores[i])
        idx_filho = int(indices_melhores_filhos[i])
        nova_populacao[idx_pior] = filhos[idx_filho]

    return nova_populacao

def prever_novo_cliente(melhor_cromossomo: np.ndarray, dados_novo_cliente: np.ndarray) -> int:
    """
    Realiza a inferência para um cliente inédito utilizando os pesos otimizados.
    """
    bias = melhor_cromossomo[0]
    genes = melhor_cromossomo[1:]
    q = np.dot(dados_novo_cliente, genes) + bias
    return 1 if q >= 0 else 0
