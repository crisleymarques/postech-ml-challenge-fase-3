# Postech ML Challenge - Fase 3
## Triagem Automática de Urgência em Laudos Médicos (NLP)

Projeto de pós-graduação (FIAP/POSTECH) que implementa um sistema de classificação
de texto (NLP), destinado à triagem automática de resumos/laudos médicos
em três níveis de urgência: **Normal**, **Atenção** e **Urgente**.
Inclui um pipeline de treinamento, análise exploratória e **API FastAPI para inferência em produção**.

![CI Status](https://github.com/crisleymarques/postech-ml-challenge-fase-3/actions/workflows/ci.yml/badge.svg?branch=classificacao_laudos_fastapi)

## 📖 Hub de Documentação

Para atender aos requisitos técnicos e de arquitetura definidos para este Tech Challenge, a documentação detalhada da solução foi estruturada nos seguintes artefatos, organizados pela ordem lógica de desenvolvimento:

- **[📚 Análise Exploratória e Dataset](docs/DATASET.md)**: Documentação detalhada sobre a origem do dataset e os mapeamentos de target aplicados (análise completa em `notebooks/EDA_Medical_Abstracts.ipynb`).
- **[🏛️ Estratégia de Deploy (ADR)](docs/ADR_PRODUCTION_STRATEGY.md)**: Justificativa arquitetural e decisão sobre inferência em tempo real vs processamento em lote.
- **[🤖 API e Docker](docs/API_DOCKER.md)**: Contrato e endpoints da API FastAPI, schemas de input/output, tratamento de erro, e conteinerização via Docker.
- **[⚖️ Benchmark e Otimização de Latência](docs/BASELINE_LATENCY.md)**: Documentação do baseline oficial de latência da API e execução do script de benchmark.
- **[⚡ Otimização ONNX Runtime](docs/INFERENCE_OPTIMIZATION_REPORT.md)**: Relatório técnico de otimização de inferência com ONNX Runtime (CPU) e comparativo com Scikit-Learn.
- **[📊 Observabilidade (Prometheus/Grafana)](docs/OBSERVABILITY.md)**: Configuração de monitoramento contínuo, métricas exportadas, dashboards provisionados e gerador de tráfego.
- **[✈️ Orquestração (Airflow)](docs/AIRFLOW.md)**: Guia de arquitetura e operação do pipeline de retreinamento contínuo e automatizado com Apache Airflow.
- **[🛠️ Guia de Contribuição e CI/CD](docs/CONTRIBUTING.md)**: Pipeline de integração contínua (GitHub Actions), testes automatizados, Definition of Done (DoD) e convenção de commits.

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
```

### 2. Treinamento do Modelo (Pipeline ML)
Gere o modelo base (`.joblib`) que será usado pela API.
```bash
python run_pipeline.py --model logistic_regression
```

> 💡 **Exemplo Rápido (sem API)**: Se você só quiser validar o modelo em um script Python puro, pode carregá-lo assim:
> ```python
> import joblib
> 
> artifact = joblib.load("models/urgency_classifier.joblib")
> pipeline = artifact["model"]
> 
> texto = "Patient with severe chest pain and elevated ST segment on ECG. Urgent cath needed."
> y_pred = pipeline.predict([texto])   # retorna array([2]) -> Urgente
> prob   = pipeline.predict_proba([texto])
> ```

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
