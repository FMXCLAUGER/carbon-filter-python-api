import pytest
from fastapi.testclient import TestClient
from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    return TestClient(app)


@pytest.fixture
def sample_calculation_request():
    """Sample calculation request data."""
    return {
        "pollutants": [
            {
                "name": "toluene",
                "concentration": 100.0,
                "molecular_weight": 92.14
            }
        ],
        "flow_rate": 1000.0,
        "temperature": 25.0,
        "humidity": 50.0,
        "pressure": 101325.0,
        "safety_factor": 1.5,
        "design_life_months": 12,
        "breakthrough_model": "WHEELER_JONAS",
        "carbon": {
            "name": "Norit GAC 1240",
            "surface_area": 1050.0,
            "bulk_density": 450.0,
            "bed_voidage": 0.4,
            "particle_diameter": 3.0
        }
    }


@pytest.fixture
def sample_iast_request():
    """Sample IAST request data for multi-component calculation."""
    return {
        "pollutants": [
            {
                "name": "toluene",
                "concentration": 100.0,
                "molecular_weight": 92.14
            },
            {
                "name": "benzene",
                "concentration": 50.0,
                "molecular_weight": 78.11
            }
        ],
        "temperature": 25.0
    }
