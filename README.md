# Datathon - PÓS TECH: Sistema de Match de Vagas com IA

## 1. Visão Geral do Projeto

### Objetivo
Este projeto visa solucionar um desafio da empresa de recrutamento "Decision", desenvolvendo uma solução de Inteligência Artificial para otimizar o processo de triagem de candidatos. O objetivo é criar um modelo de Machine Learning que prevê a compatibilidade ("match") entre o perfil de um candidato e os requisitos de uma vaga, entregando essa solução através de uma API containerizada.

### Solução Proposta
A solução consiste em uma pipeline completa de Machine Learning, desde a análise exploratória e pré-processamento dos dados até o treinamento, avaliação e deploy de um modelo preditivo.

O modelo final escolhido foi um **RandomForestClassifier (V2)**, que demonstrou a melhor performance, especialmente na métrica de **recall (49%)**, crucial para não perder bons candidatos. A solução é disponibilizada como uma API RESTful construída com FastAPI e empacotada em um contêiner Docker para garantir portabilidade e escalabilidade.

### Stack Tecnológica
* **Linguagem:** Python 3.11
* **Bibliotecas de Dados:** Pandas, Numpy, Scikit-learn
* **API:** FastAPI, Uvicorn
* **Serialização:** Joblib
* **Containerização:** Docker
* **Testes:** Pytest, Pytest-Cov

## 2. Instruções de Deploy

### Pré-requisitos
* Docker Desktop instalado e em execução.

### Passos para Executar
1.  **Clone o Repositório:**
    ```bash
    git clone [URL_DO_SEU_REPOSITORIO]
    cd [NOME_DO_SEU_REPOSITORIO]
    ```

2.  **Construa a Imagem Docker:**
    O `Dockerfile` na raiz do projeto contém todas as instruções necessárias.
    ```bash
    docker build -t match-api:v2 .
    ```

3.  **Execute o Contêiner:**
    Este comando irá iniciar a API na porta 8000.
    ```bash
    docker run -p 8000:8000 match-api:v2
    ```

4.  **Acesse a API:**
    * **Status:** [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
    * **Documentação Interativa (Swagger):** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## 3. Exemplos de Chamadas à API

Você pode usar a documentação interativa ou a ferramenta `cURL` para testar o endpoint `/predict`.

```bash
curl -X 'POST' \
  '[http://127.0.0.1:8000/predict](http://127.0.0.1:8000/predict)' \
  -H 'Content-Type: application/json' \
  -d '{
  "nivel_profissional_candidato": "sênior",
  "nivel_academico_candidato": "ensino superior completo",
  "nivel_ingles_candidato": "avançado",
  "nivel_profissional_vaga": "sênior",
  "nivel_academico_vaga": "ensino superior completo",
  "nivel_ingles_vaga": "fluente",
  "area_atuacao_candidato": "ti - sistemas e ferramentas-",
  "area_atuacao_vaga": "ti - sistemas e ferramentas-",
  "tipo_contratacao_vaga": "clt full"
}'
```

## 4. Pipeline de Machine Learning

A construção do modelo seguiu as seguintes etapas:
1.  **Carga e Junção dos Dados:** Os três arquivos JSON (`Jobs`, `Applicants`, `Prospects`) foram carregados, processados e unidos em um único DataFrame.
2.  **Definição da Variável Alvo:** A coluna `situacao_candidado` foi mapeada para uma variável binária `target` (1 para status de avanço no processo, 0 para os demais).
3.  **Engenharia de Features:** Foram criadas features manuais para comparar diretamente o perfil do candidato e da vaga (ex: `match_nivel_profissional`).
4.  **Pré-processamento:** As features categóricas foram tratadas com `OneHotEncoder` para conversão em formato numérico.
5.  **Seleção de Modelo:** O modelo `RandomForestClassifier` foi escolhido e otimizado com o parâmetro `class_weight='balanced'` para melhorar a identificação da classe minoritária (matches), resultando em um **aumento de recall de 15% para 49%**.