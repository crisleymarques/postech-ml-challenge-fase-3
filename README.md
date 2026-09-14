# Postech ML Challenge - Fase 3
## Triagem Automática de Urgência em Laudos Médicos (NLP)

Projeto de pós-graduação (FIAP/POSTECH) que implementa um sistema de classificação
de texto (NLP), destinado à triagem automática de resumos/laudos médicos
em três níveis de urgência: **Normal**, **Atenção** e **Urgente**.
Inclui um pipeline de treinamento, análise exploratória e **API FastAPI para inferência em produção**.

![CI Status](https://github.com/crisleymarques/postech-ml-challenge-fase-3/actions/workflows/ci.yml/badge.svg?branch=classificacao_laudos_fastapi)

## 📖 Hub de Documentação

O projeto é extenso e foi dividido em tópicos para facilitar o entendimento. Escolha a seção que mais se adequa à sua necessidade:

- **[🤖 API e Docker](docs/API_DOCKER.md)**: Como consumir a API FastAPI, schemas de input/output, tratamento de erro, e como subir a API isolada via Docker/Docker Compose.
- **[✈️ Orquestração (Airflow)](docs/AIRFLOW.md)**: Guia completo para rodar o pipeline de retreinamento automatizado com Apache Airflow.
- **[📊 Observabilidade (Prometheus/Grafana)](docs/OBSERVABILITY.md)**: Métricas exportadas, dashboards disponíveis e gerador de tráfego.
- **[⚖️ Benchmark e Otimização](docs/BASELINE_LATENCY.md)**: Baseline oficial de latência e execução do script de benchmark.
- **[🏛️ Estratégia de Deploy (ADR)](docs/ADR_PRODUCTION_STRATEGY.md)**: Decisões arquiteturais sobre inferência em tempo real vs batch.
- **[📚 Análise Exploratória e Dataset](docs/DATASET.md)**: Documentação detalhada sobre a origem do dataset e os mapeamentos de target aplicados (disponível no `notebooks/EDA_Medical_Abstracts.ipynb`).
- **[🛠️ Guia de Contribuição e CI/CD](docs/CONTRIBUTING.md)**: Pipeline do GitHub Actions, testes automáticos, convenção de commits e Definition of Done.

---

## 🚀 Guia Rápido (Getting Started)

Siga os passos abaixo para clonar e rodar o pipeline completo na sua máquina.

### Pré-requisitos
* **Python 3.10+** (recomendado usar [uv](https://docs.astral.sh/uv/) como gerenciador)
* **Docker e Docker Compose** (para serviços de infraestrutura e Airflow)

### 1. Instalação Local e Preparação
```bash
git clone https://github.com/crisleymarques/postech-ml-challenge-fase-3.git
cd postech-ml-challenge-fase-3

# Instalando dependências (utilizando uv)
uv sync

# Ou com pip tradicional:
pip install -r requirements.txt
```

### 2. Treinamento do Modelo (Pipeline ML)
Gere o modelo base (`.joblib`) que será usado pela API.
```bash
python run_pipeline.py --model logistic_regression
```

### 3. Execução da API Localmente
Com o modelo gerado, inicie o servidor local para testes:
```bash
python run_api.py
```
Acesse a documentação interativa (Swagger) em: **http://localhost:8000/docs**

**Exemplo de Requisição (cURL):**
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d "{\"text\": \"Patient with severe chest pain and elevated ST segment on ECG. Urgent cath needed.\"}"
```

### 4. Rodando a Stack Completa de Produção (Docker)
Para testar a infraestrutura completa de inferência + observabilidade (API, Prometheus, Grafana e Gerador de Tráfego), certifique-se de que o modelo foi treinado e execute:
```bash
docker compose --profile observability up --build -d
```
- **API (Swagger)**: http://localhost:8000/docs
- **Grafana (Admin/Admin)**: http://localhost:3000
- **Prometheus**: http://localhost:9090

Para maiores detalhes sobre cada um destes componentes (incluindo o Airflow para retreinamento contínuo), acesse os documentos do nosso **[Hub de Documentação](#-hub-de-documentação)**.
