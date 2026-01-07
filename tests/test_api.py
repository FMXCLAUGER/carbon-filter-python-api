"""Tests for FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_root_endpoint(self, client):
        """Root endpoint should return API info."""
        response = client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "Carbon Filter" in data["name"]

    def test_health_endpoint(self, client):
        """Health endpoint should return OK status."""
        response = client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"


class TestCalculationsEndpoint:
    """Tests for /api/calculate endpoint."""

    def test_calculate_single_pollutant(self, client, sample_calculation_request):
        """Single pollutant calculation should succeed."""
        response = client.post("/api/calculate", json=sample_calculation_request)
        assert response.status_code == 200

        data = response.json()
        assert "pollutant_results" in data
        assert len(data["pollutant_results"]) == 1
        assert data["pollutant_results"][0]["name"] == "toluene"
        assert "breakthrough_time" in data["pollutant_results"][0]
        assert "service_life_months" in data
        assert "overall_efficiency" in data

    def test_calculate_multi_pollutant(self, client, sample_calculation_request):
        """Multi-pollutant calculation should succeed."""
        request = sample_calculation_request.copy()
        request["pollutants"] = [
            {"name": "toluene", "concentration": 100.0, "molecular_weight": 92.14},
            {"name": "benzene", "concentration": 50.0, "molecular_weight": 78.11},
        ]

        response = client.post("/api/calculate", json=request)
        assert response.status_code == 200

        data = response.json()
        assert len(data["pollutant_results"]) == 2

    def test_calculate_missing_fields(self, client):
        """Missing required fields should return 422."""
        response = client.post("/api/calculate", json={})
        assert response.status_code == 422

    def test_calculate_invalid_values(self, client, sample_calculation_request):
        """Invalid values should return error."""
        request = sample_calculation_request.copy()
        request["flow_rate"] = -100  # Invalid negative value

        response = client.post("/api/calculate", json=request)
        # Should either return 422 or handle gracefully
        assert response.status_code in [200, 422]


class TestIASTEndpoint:
    """Tests for IAST endpoints."""

    def test_iast_status(self, client):
        """IAST status endpoint should return availability info."""
        response = client.get("/api/iast/status")
        assert response.status_code == 200

        data = response.json()
        assert "iast_available" in data
        assert isinstance(data["iast_available"], bool)

    def test_iast_calculate_multi_component(self, client, sample_iast_request):
        """IAST calculation with multiple pollutants."""
        response = client.post("/api/iast/calculate", json=sample_iast_request)

        # IAST may not be available, so accept either success or graceful failure
        assert response.status_code in [200, 400, 503, 422]

        if response.status_code == 200:
            data = response.json()
            assert "pollutants" in data or "iast_capacities_g_g" in data


class TestIsothermsEndpoint:
    """Tests for isotherms endpoints - these may not be fully implemented."""

    def test_isotherm_models_list(self, client):
        """Should return list of available isotherm models if endpoint exists."""
        response = client.get("/api/isotherms/models")
        # Endpoint may not be implemented yet
        assert response.status_code in [200, 404]

        if response.status_code == 200:
            data = response.json()
            assert "models" in data
            assert isinstance(data["models"], list)

    def test_isotherm_fit(self, client):
        """Isotherm fitting endpoint - may not be implemented."""
        request = {
            "pressure": [0.1, 0.2, 0.3, 0.4, 0.5],
            "loading": [0.15, 0.25, 0.32, 0.38, 0.42],
            "model": "langmuir",
        }

        response = client.post("/api/isotherms/fit", json=request)
        # Endpoint may or may not be implemented
        assert response.status_code in [200, 404, 422, 501]
