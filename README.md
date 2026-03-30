# Modelo de Previsão de Churn - Telecom

Projeto de **previsão de churn** em uma empresa de telecomunicações utilizando Machine Learning. O modelo foi desenvolvido com foco em MLOps, incluindo pipeline reprodutível, API com FastAPI e deploy do modelo no Hugging Face.

## 📋 Descrição

Este modelo prediz a probabilidade de um cliente cancelar o serviço (churn) com base em características contratuais, demográficas e de uso dos serviços de telecom.

### Features de Entrada

| Feature            | Tipo    | Descrição                          |
|--------------------|---------|------------------------------------|
| `gender`           | str     | Gênero do cliente                  |
| `SeniorCitizen`    | str     | Indica se o cliente é idoso        |
| `Partner`          | str     | Possui parceiro(a)                 |
| `Dependents`       | str     | Possui dependentes                 |
| `tenure`           | int     | Tempo de contrato (em meses)       |
| `PhoneService`     | str     | Possui serviço telefônico          |
| `MultipleLines`    | str     | Possui múltiplas linhas            |
| `InternetService`  | str     | Tipo de serviço de internet        |
| `OnlineSecurity`   | str     | Segurança online                   |
| `OnlineBackup`     | str     | Backup online                      |
| `DeviceProtection` | str     | Proteção de dispositivo            |
| `TechSupport`      | str     | Suporte técnico                    |
| `StreamingTV`      | str     | Streaming de TV                    |
| `StreamingMovies`  | str     | Streaming de filmes                |
| `Contract`         | str     | Tipo de contrato                   |
| `PaperlessBilling` | str     | Fatura digital                     |
| `PaymentMethod`    | str     | Método de pagamento                |
| `MonthlyCharges`   | float   | Valor mensal                       |
| `TotalCharges`     | float   | Valor total                        |

## 🛠️ Pipeline do Modelo

O modelo foi construído utilizando um **Pipeline do Scikit-learn**, garantindo consistência entre treino e inferência.

### Etapas do Pipeline:

1. Remoção da coluna `customerID`
2. Tratamento da coluna `TotalCharges`
3. Padronização de categorias
4. Imputação de valores faltantes
5. OneHotEncoding
6. Modelo **Decision Tree** (critério = entropy, `max_depth=4`)
7. Serialização do modelo com `joblib`

## 📊 Métricas

- **Acurácia aproximada**: 76%
- **Modelo**: Decision Tree (entropy, max_depth=4)
- **Principais variáveis** (por importância):
  - `Contract`
  - `OnlineSecurity`
  - `InternetService`
  - `TechSupport`
  - `StreamingMovies`

## 🧩 Dependências

- scikit-learn
- pandas
- joblib
- huggingface_hub
- FastAPI
- Uvicorn

## 🌐 Integração com API

A API foi desenvolvida com **FastAPI** e consome o modelo hospedado no Hugging Face.

### Endpoint

- **POST** `/predict`

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

Exemplo de Resposta
JSON{
  "churn_predito": 1,
  "churn_label": "Yes",
  "probabilidade_churn": 0.6404
}

🤗 Hugging Face
Modelo publicado em:
VanessaMCorrea/churn-telecom-model
⚠️ Limitações e Considerações Éticas

O dataset utilizado é público e pode não refletir perfeitamente a realidade da empresa.
Pode ocorrer queda de performance em dados novos (data drift).
Modelos de árvore de decisão simplificam comportamentos complexos de clientes.
Não recomendado para uso em produção sem validação adicional e monitoramento contínuo.

👩‍💻 Autora
Vanessa Correa
PUC-SP — Ciência de Dados e Inteligência Artificial
Projeto acadêmico com foco em MLOps.