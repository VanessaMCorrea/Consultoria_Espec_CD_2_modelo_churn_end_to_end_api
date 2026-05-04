# Modelo de Previsão de Churn - Telecom

Projeto de **previsão de churn** em uma empresa de telecomunicações utilizando Machine Learning. O modelo foi desenvolvido com foco em MLOps, incluindo pipeline reprodutível, API com FastAPI, deploy do modelo no Hugging Face e pipeline de CI/CD com GitHub Actions.

## 📋 Descrição

Este modelo prediz a probabilidade de um cliente cancelar o serviço (churn) com base em características contratuais, demográficas e de uso dos serviços de telecom.

### Features de Entrada

- Base de dados: https://www.kaggle.com/datasets/palashfendarkar/wa-fnusec-telcocustomerchurn

| Feature            | Tipo  | Descrição                        |
|--------------------|-------|----------------------------------|
| `gender`           | str   | Gênero do cliente                |
| `SeniorCitizen`    | str   | Indica se o cliente é idoso      |
| `Partner`          | str   | Possui parceiro(a)               |
| `Dependents`       | str   | Possui dependentes               |
| `tenure`           | int   | Tempo de contrato (em meses)     |
| `PhoneService`     | str   | Possui serviço telefônico        |
| `MultipleLines`    | str   | Possui múltiplas linhas          |
| `InternetService`  | str   | Tipo de serviço de internet      |
| `OnlineSecurity`   | str   | Segurança online                 |
| `OnlineBackup`     | str   | Backup online                    |
| `DeviceProtection` | str   | Proteção de dispositivo          |
| `TechSupport`      | str   | Suporte técnico                  |
| `StreamingTV`      | str   | Streaming de TV                  |
| `StreamingMovies`  | str   | Streaming de filmes              |
| `Contract`         | str   | Tipo de contrato                 |
| `PaperlessBilling` | str   | Fatura digital                   |
| `PaymentMethod`    | str   | Método de pagamento              |
| `MonthlyCharges`   | float | Valor mensal                     |
| `TotalCharges`     | float | Valor total                      |

---

## 🛠️ Pipeline do Modelo

O modelo foi construído utilizando um **Pipeline do Scikit-learn**, garantindo consistência entre treino e inferência.

### Etapas do Pipeline

1. Remoção da coluna `customerID`
2. Tratamento da coluna `TotalCharges` (conversão numérica com `errors="coerce"`)
3. Padronização de categorias (`No internet service` e `No phone service` → `No`)
4. Imputação de valores faltantes (mediana para numéricos, moda para categóricos)
5. OneHotEncoding com `handle_unknown="ignore"`
6. Modelo **Decision Tree** (critério = entropy, `max_depth=4`, `random_state=42`)
7. Serialização do modelo com `joblib`

### Divisão treino/teste

O treinamento utiliza divisão estratificada 80/20 com `random_state=42`, garantindo reprodutibilidade e distribuição balanceada de classes.

---

## 📊 Métricas

- **Acurácia aproximada**: 76%
- **Modelo**: Decision Tree (entropy, max_depth=4)
- **Principais variáveis** (por importância):
  - `Contract`
  - `OnlineSecurity`
  - `InternetService`
  - `TechSupport`
  - `StreamingMovies`

---

## 🧩 Dependências

```
fastapi
uvicorn
scikit-learn
pandas
joblib
huggingface_hub
python-dotenv
pytest
httpx
```

Versões fixadas em `requirements.txt`.

---

## 🌐 Integração com API

A API foi desenvolvida com **FastAPI** e consome o modelo hospedado no Hugging Face Hub. O modelo é baixado em tempo de execução via `hf_hub_download()` — **nenhum artefato binário é versionado no repositório Git**.

### Endpoint

- **GET** `/` — verifica se a API está no ar
- **GET** `/health` — healthcheck para monitoramento
- **POST** `/predict` — realiza a previsão de churn

### Exemplo de Request

```json
{
  "gender": "Female",
  "SeniorCitizen": "0",
  "Partner": "Yes",
  "Dependents": "No",
  "tenure": 12,
  "PhoneService": "Yes",
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaperlessBilling": "Yes",
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 89.5,
  "TotalCharges": 1074.0
}
```

### Exemplo de Resposta

```json
{
  "churn_predito": 1,
  "churn_label": "Yes",
  "probabilidade_churn": 0.6404
}
```

---

## 🤗 Publicação do Modelo no Hugging Face

Modelo publicado em: `VanessaMCorrea/churn-telecom-model`

O arquivo `churn_pipeline.joblib` é baixado automaticamente pela API e pelo pipeline de CI/CD. As variáveis de ambiente controlam qual repositório e arquivo usar:

| Variável       | Valor padrão                          |
|----------------|---------------------------------------|
| `HF_REPO_ID`   | `VanessaMCorrea/churn-telecom-model`  |
| `HF_FILENAME`  | `churn_pipeline.joblib`               |
| `HF_TOKEN`     | configurado via secret no GitHub      |

---

## ⚙️ CI/CD com GitHub Actions

O pipeline de CI está em `.github/workflows/ci.yml` e é composto por **3 jobs encadeados**:

```
qualidade ──needs──▶ testes ──needs──▶ api_modelo
                                        (somente push → main)
```

### Job 1 — `qualidade`
- Roda em todo `push` e `pull_request` para `main`
- Verifica sintaxe Python com `python -m compileall`
- Usa cache de pip via `actions/setup-python`

### Job 2 — `testes`
- Depende de `qualidade` (`needs: qualidade`)
- Em `pull_request`: roda apenas testes marcados com `health` — feedback rápido
- Em `push`: roda toda a suite de testes
- Autentica no Hugging Face com `HF_TOKEN` via secret

### Job 3 — `api_modelo`
- Depende de `testes` (`needs: testes`)
- **Condição**: só executa em `push` para `main` — nunca em PRs
- Sobe a API com `uvicorn` em background
- Verifica `/health` e `/docs` com `curl --fail`

### Configurando o secret `HF_TOKEN`

No GitHub: **Settings → Secrets and variables → Actions → New repository secret**

- Nome: `HF_TOKEN`
- Valor: token gerado em [hf.co/settings/tokens](https://huggingface.co/settings/tokens)

---

## 🧪 Testes Automatizados

Os testes estão organizados em `tests/` com os seguintes arquivos:

| Arquivo         | Conteúdo                                              |
|-----------------|-------------------------------------------------------|
| `conftest.py`   | Fixtures `client` (TestClient) e `valid_payload`      |
| `test_api.py`   | Testes dos endpoints `/health`, `/` e `/predict`      |
| `test_model.py` | Testes do objeto de modelo carregado                  |
| `pytest.ini`    | Marcadores registrados: `health`, `predict`, `model`  |

### Executando localmente

```bash
# Criar ambiente virtual limpo
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# Instalar dependências
pip install -r requirements.txt

# Rodar todos os testes
pytest tests/ -v

# Rodar apenas testes rápidos (healthcheck)
pytest tests/ -m health -v

# Rodar apenas testes de predição
pytest tests/ -m predict -v
```

### Estratégia de testes

Os testes verificam **comportamento relativo**, não estado absoluto:

```python
# ✅ correto — não depende do valor exato do modelo
assert data["churn_predito"] in [0, 1]
assert 0 <= data["probabilidade_churn"] <= 1

# ❌ frágil — quebra se o modelo for retreinado
assert data["churn_predito"] == 1
```

Testes parametrizados cobrem múltiplos perfis de cliente com um único bloco de código:

```python
@pytest.mark.parametrize("tenure,perfil", [
    (1,  "cliente novo"),
    (12, "cliente intermediário"),
    (60, "cliente antigo"),
])
def test_predict_accepts_different_tenure_values(client, valid_payload, tenure, perfil):
    ...
```

---

## 🗂️ Estrutura do Projeto

```
.
├── .github/
│   └── workflows/
│       └── ci.yml              # Pipeline CI/CD com 3 jobs
├── app/
│   ├── __init__.py
│   ├── main.py                 # Aplicação FastAPI
│   ├── predictor.py            # Download do modelo e lógica de predição
│   └── schemas.py              # Schema de entrada (Pydantic)
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
├── scripts/
│   └── train_model.py          # Treinamento e serialização do modelo
├── tests/
│   ├── conftest.py             # Fixtures compartilhadas
│   ├── test_api.py             # Testes dos endpoints
│   └── test_model.py           # Testes do modelo
├── .gitignore
├── Dockerfile
├── pytest.ini                  # Configuração e marcadores do pytest
├── requirements.txt
└── README.md
```

---

## 🐳 Docker

```bash
# Build da imagem
docker build -t churn-api .

# Subir o container (passando as variáveis de ambiente)
docker run -p 8000:8000 \
  -e HF_TOKEN=seu_token \
  -e HF_REPO_ID=VanessaMCorrea/churn-telecom-model \
  -e HF_FILENAME=churn_pipeline.joblib \
  churn-api
```

A API ficará disponível em `http://localhost:8000`. A documentação interativa (Swagger) em `http://localhost:8000/docs`.

---

## ⚠️ Limitações e Considerações Éticas

- O dataset utilizado é público e pode não refletir perfeitamente a realidade de uma empresa real.
- Pode ocorrer queda de performance em dados novos (data drift) — monitoramento contínuo é recomendado.
- Modelos de árvore de decisão simplificam comportamentos complexos de clientes.
- Não recomendado para uso em produção sem validação adicional e análise de fairness.
- Decisões baseadas neste modelo não devem ser automatizadas sem revisão humana.

---

## 👩‍💻 Autora

Vanessa Correa  
PUC-SP — Ciência de Dados e Inteligência Artificial  
Projeto acadêmico com foco em MLOps.