import pytest


@pytest.mark.health
def test_health_status(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


@pytest.mark.health
def test_root_status(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


@pytest.mark.predict
def test_predict_returns_expected_structure(client, valid_payload):
    response = client.post("/predict", json=valid_payload)

    assert response.status_code == 200

    data = response.json()

    assert "churn_predito" in data
    assert "churn_label" in data
    assert "probabilidade_churn" in data

    assert data["churn_predito"] in [0, 1]
    assert data["churn_label"] in ["Yes", "No"]
    assert 0 <= data["probabilidade_churn"] <= 1


@pytest.mark.predict
@pytest.mark.parametrize(
    "tenure,dias_esperado",
    [
        (1, "cliente novo"),
        (12, "cliente intermediario"),
        (60, "cliente antigo"),
    ],
)
def test_predict_accepts_different_tenure_values(client, valid_payload, tenure, dias_esperado):
    payload = valid_payload.copy()
    payload["tenure"] = tenure

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["churn_predito"] in [0, 1]