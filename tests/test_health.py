from app.orchestration.tasks import healthcheck

def test_healthcheck():
    result = healthcheck()
    assert "status" in result
    assert result["status"] == "ok"
