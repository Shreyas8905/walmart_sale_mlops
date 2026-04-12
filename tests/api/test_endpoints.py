"""API endpoint tests."""

from __future__ import annotations


def _payload() -> dict:
	return {
		"Store": 1,
		"Date": "05-02-2010",
		"Holiday_Flag": 0,
		"Temperature": 42.31,
		"Fuel_Price": 2.572,
		"CPI": 211.0963582,
		"Unemployment": 8.106,
	}


def test_health_endpoint(app_client):
	response = app_client.get("/api/v1/health")
	assert response.status_code == 200


def test_model_info_endpoint(app_client):
	response = app_client.get("/api/v1/model/info")
	assert response.status_code == 200
	payload = response.json()
	assert "model_name" in payload
	assert "version" in payload


def test_predict_valid_payload(app_client):
	response = app_client.post("/api/v1/predict", json=_payload())
	assert response.status_code == 200
	body = response.json()
	assert "predicted_weekly_sales" in body


def test_predict_missing_field(app_client):
	payload = _payload()
	payload.pop("CPI")
	response = app_client.post("/api/v1/predict", json=payload)
	assert response.status_code == 422


def test_predict_wrong_type(app_client):
	payload = _payload()
	payload["Temperature"] = "hot"
	response = app_client.post("/api/v1/predict", json=payload)
	assert response.status_code == 422


def test_predict_batch_valid(app_client):
	response = app_client.post("/api/v1/predict/batch", json={"records": [_payload() for _ in range(5)]})
	assert response.status_code == 200
	body = response.json()
	assert body["total_records"] == 5


def test_predict_batch_empty_records(app_client):
	response = app_client.post("/api/v1/predict/batch", json={"records": []})
	assert response.status_code == 422


def test_predict_batch_too_many_records(app_client):
	response = app_client.post("/api/v1/predict/batch", json={"records": [_payload() for _ in range(1001)]})
	assert response.status_code == 422
