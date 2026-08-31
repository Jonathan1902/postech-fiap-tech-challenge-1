def test_health_returns_200(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_health_schema(client):
    data = client.get("/health").json()
    assert data["status"] == "ok"
    assert "model_version" in data
    assert 0.0 < data["threshold"] < 1.0
    assert data["uptime_s"] >= 0
