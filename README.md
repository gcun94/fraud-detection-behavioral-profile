# Detecção de Fraude com Perfil Comportamental de Clientes

> Pipeline integrada de Machine Learning que combina segmentação comportamental não supervisionada com detecção de fraude supervisionada em transações de cartão de crédito.

\---

## Problema

Instituições financeiras perdem bilhões anualmente com fraudes em cartão de crédito. Modelos tradicionais de detecção tratam todos os clientes com o mesmo padrão de normalidade, ignorando que uma transação suspeita para um perfil pode ser completamente normal para outro.

**Exemplo:** Uma compra de R$ 5.000 às 3h da manhã no exterior é um sinal crítico de fraude para um cliente com histórico de compras locais e de baixo valor. Para um executivo com histórico de viagens internacionais frequentes, é uma transação comum.

Sem contexto comportamental, o modelo não consegue distinguir esses casos — gerando tanto falsos positivos (bloqueios indevidos) quanto falsos negativos (fraudes não detectadas).

\---

## Solução

Pipeline de dois módulos integrados:

**Módulo 1 — Segmentação Comportamental (Não Supervisionado)**
Agrupa clientes por padrão de comportamento financeiro usando K-Means. O cluster atribuído a cada cliente vira uma feature do modelo de fraude.

**Módulo 2 — Detecção de Fraude (Supervisionado)**
Modelo de classificação binária que usa o perfil comportamental do cliente como contexto para avaliar se uma transação é fraudulenta.

**Hipótese testável:** um modelo de fraude que conhece o perfil do cliente performa melhor do que um modelo que trata todos os clientes igualmente.

\---

## Dataset

|Atributo|Detalhe|
|-|-|
|**Fonte**|[Credit Card Transactions Fraud Detection — Kaggle](https://www.kaggle.com/datasets/kartik2112/fraud-detection)|
|**Volume**|\~1,8 milhão de transações|
|**Período**|Janeiro 2019 — Dezembro 2020|
|**Licença**|Creative Commons Zero (CC0)|
|**Target**|`is\_fraud` (0 = legítima, 1 = fraude)|

**Variáveis disponíveis:** timestamp, valor da transação, categoria do comerciante, localização do cliente e comerciante, gênero, profissão, idade.

> ⚠️ O dataset não está incluído neste repositório. Baixe diretamente no Kaggle e coloque os arquivos em `data/raw/`.

\---

## Estrutura do Projeto

```
fraud-detection-behavioral-profile/
│
├── data/
│   ├── raw/                          ← dados originais (não versionados)
│   └── processed/                    ← dados após limpeza e transformações
│
├── notebooks/
│   ├── 01\_eda.ipynb                  ← Análise Exploratória
│   ├── 02\_preprocessing.ipynb        ← Limpeza e Feature Engineering
│   ├── 03\_clustering.ipynb           ← Módulo 1: Segmentação Comportamental
│   └── 04\_fraud\_detection.ipynb      ← Módulo 2: Detecção de Fraude
│
├── src/
│   └── utils.py                      ← funções reutilizáveis entre notebooks
│
├── outputs/
│   ├── figures/                      ← gráficos e visualizações exportados
│   └── models/                       ← modelos serializados (.pkl)
│
├── requirements.txt
├── .gitignore
└── README.md
```

\---

## Metodologia

### 1\. Análise Exploratória (EDA)

* Distribuição do target (desbalanceamento de classes)
* Comportamento temporal das transações
* Análise por categoria, valor e localização
* Identificação de missing values e outliers

### 2\. Pré-processamento e Feature Engineering

* Tratamento de valores ausentes e outliers
* Agregações por cliente: ticket médio, frequência, variação geográfica, horários atípicos
* Encoding de variáveis categóricas
* Normalização para o clustering

### 3\. Módulo 1 — Segmentação Comportamental

* **Algoritmo:** K-Means
* **Validação do número de clusters:** Elbow Method + Silhouette Score
* **Visualização:** PCA 2D dos clusters
* **Interpretação:** perfil de cada grupo (comportamento esperado)

### 4\. Módulo 2 — Detecção de Fraude

* **Baseline:** Regressão Logística (sem perfil comportamental)
* **Modelos comparados:** Random Forest · XGBoost
* **Tratamento do desbalanceamento:** SMOTE
* **Métricas:** AUC-ROC · Precision-Recall · F1-Score · Matriz de Confusão
* **Interpretabilidade:** SHAP values

### 5\. Comparação e Validação da Hipótese

* Modelo sem cluster vs. modelo com cluster como feature
* Análise do impacto do perfil comportamental na performance
* Threshold tuning orientado ao custo do erro de negócio

\---

## Resultados

> Tabela atualizada após execução completa do projeto.

|Modelo|AUC-ROC|F1-Score|Precision|Recall|
|-|-|-|-|-|
|Baseline — Regressão Logística|—|—|—|—|
|Random Forest (sem cluster)|—|—|—|—|
|XGBoost (sem cluster)|—|—|—|—|
|**XGBoost (com perfil comportamental)**|**—**|**—**|**—**|**—**|

\---

## Como Reproduzir

### Pré-requisitos

* Python 3.9+
* Git instalado
* Conta no Kaggle

### Passo a passo

```bash
# 1. Clonar o repositório
git clone https://github.com/seuusuario/fraud-detection-behavioral-profile.git
cd fraud-detection-behavioral-profile

# 2. Criar e ativar ambiente virtual (Windows)
python -m venv venv
venv\\Scripts\\activate

# 3. Instalar dependências
pip install -r requirements.txt

# 4. Baixar o dataset
# Acesse: https://www.kaggle.com/datasets/kartik2112/fraud-detection
# Faça o download e coloque os arquivos CSV em data/raw/

# 5. Executar os notebooks na ordem
jupyter notebook
```

Execute os notebooks na sequência: `01 → 02 → 03 → 04`

\---

## Tecnologias

[![Python](https://img.shields.io/badge/Python-3.9+-blue)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-2.0+-green)](https://pandas.pydata.org/)
[![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-orange)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-red)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-purple)](https://shap.readthedocs.io/)

* **Manipulação de dados:** Pandas · NumPy
* **Visualização:** Matplotlib · Seaborn
* **Machine Learning:** Scikit-learn · XGBoost · Imbalanced-learn
* **Interpretabilidade:** SHAP
* **Ambiente:** Jupyter Notebook

\---

## Requirements

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
xgboost>=2.0.0
imbalanced-learn>=0.11.0
shap>=0.43.0
jupyter>=1.0.0
```

\---

## Autor

**Gustavo Simões da Cunha**
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue)](https://www.linkedin.com/in/gustavo-s-cunha-516102181/)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black)](https://github.com/gcun94)
\---

## Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

\---

*Projeto desenvolvido como parte de portfólio acadêmico e profissional em Ciência de Dados.*

