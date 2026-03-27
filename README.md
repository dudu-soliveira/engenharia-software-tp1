# BurnCheck: Saúde Mental e Vida Acadêmica


**Objetivo Central**
O projeto tem como objetivo investigar a correlação entre o cotidiano acadêmico e a saúde mental dos estudantes. A meta principal é desenvolver uma **aplicação interativa baseada em um questionário** que, alimentada por um modelo preditivo, seja capaz de avaliar previamente o risco de *burnout* de um aluno com base em seus hábitos e rotina.

**Funcionalidades Principais**
* **Tratamento e Exploração de Dados:** Limpeza da base de dados e realização de uma Análise Exploratória (EDA) para entender o impacto de variáveis como carga horária, sono e integração social nos níveis de estresse e depressão.
* **Aplicação de Questionário (Interface do Usuário):** Uma interface em que o estudante preenche informações sobre seu cotidiano. O sistema utiliza o modelo de classificação treinado para processar essas respostas e retornar o nível de risco de *burnout* desse estudante.
* **Dashboard de Visualização Personalizado:** Ao final do questionário, o sistema gera um dashboard dinâmico que permite ao usuário visualizar seus próprios dados em comparação com as médias do dataset (ex: horas de sono vs. média de outros alunos), facilitando a compreensão do seu estado atual.
* **Feedback de Vulnerabilidade:** A aplicação demonstrará a importância das variáveis, revelando quais fatores específicos da rotina do usuário são os maiores preditores do seu risco mental, incentivando a prevenção e mudanças de hábito.

---

## Base de dados
Student Mental Health and Burnout Dataset: https://www.kaggle.com/datasets/sehaj1104/student-mental-health-and-burnout-datase

### 📚 Sobre o Dataset (*Student Mental Health and Burnout*)

O **Conjunto de Dados de Saúde Mental e Burnout de Estudantes** (150 mil linhas, 20 atributos) é uma base de dados sintética de larga escala, projetada para analisar e prever os níveis de *burnout* estudantil com base em fatores acadêmicos, psicológicos e de estilo de vida.

### 📊 Atributos do Dataset (*Features*)

O conjunto de dados inclui os seguintes grupos de variáveis:

* **Dados Demográficos:** idade, gênero, curso, ano letivo.
* **Indicadores Acadêmicos:** média geral (CGPA/CR), frequência (presença), horas de estudo.
* **Pontuações Psicológicas:** ansiedade, depressão, nível de estresse.
* **Fatores de Estilo de Vida:** horas de sono, atividade física, tempo de tela.
* **Indicadores Sociais:** estresse social e financeiro.
  

* **Variável Alvo (*Target*):** `burnout_level` (nível de esgotamento)

---

## Membros da Equipe e Papéis

*   Eduardo de Sousa Oliveira: Elaboração do questionário e da aplicação web
*   Guilherme Bueno de Andrade Motta: Criação do dashboard interativo
*   Gustavo Cabral Gonçalves: Desenvolvimento do modelo preditivo
*   Renato Lucas Valente Lisboa: Desenvolvimento do modelo preditivo

---

## 🛠️ Pilha de Tecnologias (Tech Stack)

### 🐍 Linguagem e Processamento de Dados
* **Python / R**
* **Pandas & NumPy / tidyverse**
* **Scikit-Learn & XGBoost / tidymodels & caret**

### 🖥️ Interface e Visualização
* **Streamlit**
* **Plotly**

### 🏗️ Infraestrutura e Suporte
* **Git & GitHub**
* **Joblib / Pickle**
* **Streamlit Cloud**
