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
Pré-requisitos
Docker Desktop: Instale e execute o Docker Desktop, que inclui o Docker Engine e o Docker Compose.

Passos para Executar
O projeto utiliza o Docker Compose para orquestrar todos os serviços de forma simples, garantindo que a API e qualquer outro serviço necessário subam com um único comando.

Clone o Repositório:

```bash
git clone [URL_DO_SEU_REPOSITORIO]
cd [NOME_DO_SEU_REPOSITORIO]
```

Construa e Suba os Contêineres:
Com o docker-compose.yml na raiz do projeto, este comando irá construir a imagem da API e iniciar o serviço na porta 8000.


```bash
docker compose up --build
```

A flag --build garante que a imagem seja reconstruída com as últimas alterações do seu código antes de iniciar o contêiner.

Acesse a API:

Status: http://127.0.0.1:8000/

Documentação Interativa (Swagger): http://127.0.0.1:8000/docs

Comandos Úteis do Docker Compose
Rodar em segundo plano: Para deixar o terminal livre, use a flag -d (detach).

```bash
docker compose up -d
```
Parar e Remover Contêineres: Para encerrar os serviços e limpar os contêineres e redes, use este comando.

```bash
docker compose down
Visualizar Logs: Para inspecionar a saída de todos os contêineres, use o comando de logs.
```

```bash
docker compose logs
```

## 3. Exemplos de Chamadas à API

### 1. Health Check (/health)
Este endpoint verifica o status da API e se o modelo de machine learning foi carregado corretamente.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/health'
```

Retorno esperado: Se tudo estiver funcionando, você receberá um status 200 OK e um JSON com a mensagem de que a API e o modelo estão operacionais.

### 2. Status da API (/)
Este é o endpoint raiz, que confirma que a sua API está no ar.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/'
```
Retorno esperado: Um JSON com a mensagem de status da API.

### 3. Predição de Match (/predict)
Este é o endpoint principal que recebe os dados de um candidato e de uma vaga e retorna a predição de compatibilidade do modelo. Você deve enviar um corpo JSON com as informações necessárias.

```bash
curl -X 'POST' \
  'http://127.0.0.1:8000/predict' \
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
Retorno esperado: Um JSON com a predição do modelo.

### 4. Métricas do Prometheus (/metrics)
Este endpoint, usado para monitoramento, expõe as métricas de latência, número de requisições e taxa de erro da sua aplicação.

```bash
curl -X 'GET' \
  'http://127.0.0.1:8000/metrics'
```
Retorno esperado: Um texto simples com as métricas da sua API no formato do Prometheus

## 4. Pipeline de Machine Learning

A construção do modelo seguiu as seguintes etapas:
1.  **Carga e Junção dos Dados:** Os três arquivos JSON (`Jobs`, `Applicants`, `Prospects`) foram carregados, processados e unidos em um único DataFrame.
2.  **Definição da Variável Alvo:** A coluna `situacao_candidado` foi mapeada para uma variável binária `target` (1 para status de avanço no processo, 0 para os demais).
3.  **Engenharia de Features:** Foram criadas features manuais para comparar diretamente o perfil do candidato e da vaga (ex: `match_nivel_profissional`).
4.  **Pré-processamento:** As features categóricas foram tratadas com `OneHotEncoder` para conversão em formato numérico.
5.  **Seleção de Modelo:** O modelo `RandomForestClassifier` foi escolhido e otimizado com o parâmetro `class_weight='balanced'` para melhorar a identificação da classe minoritária (matches), resultando em um **aumento de recall de 15% para 49%**.


## 5. Deploy na AWS

O deploy da solução na AWS foi realizado em uma conta free tier, seguindo uma arquitetura de orquestração de contêineres com AWS ECS (Elastic Container Service). O processo envolve o empacotamento do código em uma imagem Docker, o envio para um repositório na nuvem e a criação de serviços AWS para gerenciar a aplicação.

### 5.1. Repositório ECR
O Amazon Elastic Container Registry (ECR) foi utilizado como o repositório privado para armazenar a imagem Docker da API. Os seguintes comandos foram executados via AWS CLI para autenticar o Docker e fazer o push da imagem.

Autenticar o Docker no ECR:

```bash
aws ecr get-login-password --region sa-east-1 | docker login --username AWS --password-stdin [account-number].dkr.ecr.sa-east-1.amazonaws.com
```
Observação: A versão mais recente da AWS CLI é necessária para que o comando funcione corretamente.


Construir a Imagem Docker:

```bash
docker build -t datathon .
```
Adicionar a Tag do Repositório ECR na Imagem:

```bash
docker tag datathon:latest 530997927415.dkr.ecr.sa-east-1.amazonaws.com/datathon:latest
```

Fazer o Push da Imagem para o ECR:

```bash
docker push 530997927415.dkr.ecr.sa-east-1.amazonaws.com/datathon:latest
```

### 5.2. Configuração no ECS
Após a imagem ser enviada para o ECR, a orquestração foi configurada no ECS.

Criação do Cluster ECS: Um cluster ECS foi criado para agrupar as instâncias onde a aplicação seria executada.

Definição da Task Definition: Uma Definição de Tarefa foi criada para servir como um "blueprint" para o contêiner. Ela foi configurada para fazer o pull da imagem a partir do repositório ECR.

Criação do Serviço ECS: Um Serviço foi criado para garantir que a aplicação estivesse sempre em execução. Este Serviço foi configurado para usar a Definição de Tarefa e para manter um número mínimo de tarefas rodando.

### 5.3. Balanceador de Carga (ALB)
Um Application Load Balancer (ALB) foi configurado para gerenciar o tráfego de entrada e direcioná-lo para a API.

Target Group: Um Target Group foi criado, apontando para a porta 8000, que é a porta em que a API é executada dentro do contêiner.

Listener: Um Listener foi configurado na porta 8000 do ALB para receber o tráfego da internet e direcioná-lo para o Target Group.

Dessa forma, a aplicação foi disponibilizada na AWS, acessível através da URL do Application Load Balancer.