"""
pyGAPS integration service for isotherm modeling.

Uses pyGAPS library for accurate adsorption isotherm calculations:
- Model isotherms (Langmuir, Freundlich, Toth, DR, DA, etc.)
- Predict adsorption capacity at operating conditions
- Temperature corrections via Clausius-Clapeyron
"""

from typing import Optional
import numpy as np

try:
    import pygaps
    PYGAPS_AVAILABLE = True
except ImportError:
    PYGAPS_AVAILABLE = False


SUPPORTED_MODELS = [
    "Langmuir",
    "Freundlich",
    "Toth",
    "DSLangmuir",
    "DR",
    "DA",
    "BET",
    "Henry",
]


# Isotherm parameters calibrated for trace gas concentrations (10-1000 ppm)
# K values adjusted to match industrial activated carbon capacities
# n_m in mmol/g, K in 1/bar, t is heterogeneity parameter
# Target: at 100 mg/m³ inlet, capacity should match typical industrial values
DEFAULT_ISOTHERM_PARAMS = {
    "toluene": {
        "model": "Toth",
        "params": {"n_m": 4.8, "K": 50000.0, "t": 0.48},
        "temp_ref": 298.15,
    },
    "benzene": {
        "model": "Toth",
        "params": {"n_m": 4.2, "K": 55000.0, "t": 0.45},
        "temp_ref": 298.15,
    },
    "xylene": {
        "model": "Toth",
        "params": {"n_m": 5.2, "K": 60000.0, "t": 0.50},
        "temp_ref": 298.15,
    },
    "ethylbenzene": {
        "model": "Toth",
        "params": {"n_m": 5.0, "K": 58000.0, "t": 0.49},
        "temp_ref": 298.15,
    },
    "acetone": {
        "model": "Langmuir",
        "params": {"n_m": 3.2, "K": 30000.0},
        "temp_ref": 298.15,
    },
    "methanol": {
        "model": "Langmuir",
        "params": {"n_m": 2.5, "K": 25000.0},
        "temp_ref": 298.15,
    },
    "ethanol": {
        "model": "Langmuir",
        "params": {"n_m": 3.0, "K": 28000.0},
        "temp_ref": 298.15,
    },
    "formaldehyde": {
        "model": "Langmuir",
        "params": {"n_m": 1.5, "K": 15000.0},
        "temp_ref": 298.15,
    },
    "h2s": {
        "model": "Toth",
        "params": {"n_m": 3.5, "K": 80000.0, "t": 0.42},
        "temp_ref": 298.15,
    },
    "ammonia": {
        "model": "Langmuir",
        "params": {"n_m": 2.0, "K": 35000.0},
        "temp_ref": 298.15,
    },
    "chlorine": {
        "model": "Langmuir",
        "params": {"n_m": 4.5, "K": 65000.0},
        "temp_ref": 298.15,
    },
    "so2": {
        "model": "Langmuir",
        "params": {"n_m": 2.8, "K": 40000.0},
        "temp_ref": 298.15,
    },
    "styrene": {
        "model": "Toth",
        "params": {"n_m": 4.6, "K": 52000.0, "t": 0.47},
        "temp_ref": 298.15,
    },
    "phenol": {
        "model": "Toth",
        "params": {"n_m": 4.0, "K": 68000.0, "t": 0.44},
        "temp_ref": 298.15,
    },
}


MOLECULAR_WEIGHTS = {
    "toluene": 92.14,
    "benzene": 78.11,
    "xylene": 106.16,
    "ethylbenzene": 106.17,
    "acetone": 58.08,
    "methanol": 32.04,
    "ethanol": 46.07,
    "formaldehyde": 30.03,
    "h2s": 34.08,
    "ammonia": 17.03,
    "chlorine": 70.91,
    "so2": 64.07,
    "styrene": 104.15,
    "phenol": 94.11,
}


def concentration_to_partial_pressure(
    concentration_mg_m3: float,
    molecular_weight: float,
    temperature_c: float = 25.0,
    total_pressure_pa: float = 101325.0,
) -> float:
    """
    Convert concentration (mg/m³) to partial pressure (bar).

    Uses ideal gas law: p = (C × R × T) / (M × 1000)

    Args:
        concentration_mg_m3: Concentration in mg/m³
        molecular_weight: Molecular weight in g/mol
        temperature_c: Temperature in °C
        total_pressure_pa: Total pressure in Pa

    Returns:
        Partial pressure in bar
    """
    R = 8.314  # J/(mol·K)
    T = temperature_c + 273.15  # K

    concentration_mol_m3 = concentration_mg_m3 / (molecular_weight * 1000)
    partial_pressure_pa = concentration_mol_m3 * R * T
    partial_pressure_bar = partial_pressure_pa / 100000

    return partial_pressure_bar


def get_default_isotherm_params(pollutant_name: str) -> Optional[dict]:
    """Get default isotherm parameters for a pollutant."""
    name_lower = pollutant_name.lower().replace(" ", "").replace("-", "")
    return DEFAULT_ISOTHERM_PARAMS.get(name_lower)


def get_molecular_weight(pollutant_name: str) -> Optional[float]:
    """Get molecular weight for a pollutant."""
    name_lower = pollutant_name.lower().replace(" ", "").replace("-", "")
    return MOLECULAR_WEIGHTS.get(name_lower)


def create_isotherm_model(
    model_name: str,
    params: dict,
):
    """
    Create a pyGAPS isotherm model from parameters.

    Args:
        model_name: Isotherm model name (Langmuir, Toth, etc.)
        params: Model parameters dict

    Returns:
        pygaps model object or None if pyGAPS not available
    """
    if not PYGAPS_AVAILABLE:
        return None

    try:
        import pygaps.modelling as pgm
        model = pgm.get_isotherm_model(model_name)
        model.params = params
        return model
    except Exception:
        return None


def predict_loading_at_pressure(
    model,
    pressure_bar: float,
) -> Optional[float]:
    """
    Predict loading at a given pressure using pyGAPS model.

    Args:
        model: pygaps model object
        pressure_bar: Pressure in bar

    Returns:
        Loading in mmol/g or None if calculation fails
    """
    if model is None:
        return None

    try:
        loading = model.loading(pressure_bar)
        return float(loading)
    except Exception:
        return None


def predict_capacity_pygaps(
    pollutant_name: str,
    concentration_mg_m3: float,
    temperature_c: float,
    molecular_weight: Optional[float] = None,
    isotherm_params: Optional[dict] = None,
) -> Optional[float]:
    """
    Predict adsorption capacity using pyGAPS.

    Args:
        pollutant_name: Name of pollutant
        concentration_mg_m3: Inlet concentration in mg/m³
        temperature_c: Temperature in °C
        molecular_weight: Molecular weight in g/mol (optional)
        isotherm_params: Custom isotherm parameters (optional)

    Returns:
        Adsorption capacity in g/g or None if calculation fails
    """
    if not PYGAPS_AVAILABLE:
        return None

    if isotherm_params is None:
        isotherm_params = get_default_isotherm_params(pollutant_name)

    if isotherm_params is None:
        return None

    if molecular_weight is None:
        molecular_weight = get_molecular_weight(pollutant_name)

    if molecular_weight is None:
        molecular_weight = 100.0

    partial_pressure = concentration_to_partial_pressure(
        concentration_mg_m3,
        molecular_weight,
        temperature_c,
    )

    temp_ref = isotherm_params.get("temp_ref", 298.15)
    temperature_k = temperature_c + 273.15

    model = create_isotherm_model(
        model_name=isotherm_params["model"],
        params=isotherm_params["params"],
    )

    if model is None:
        return None

    loading_mmol_g = predict_loading_at_pressure(model, partial_pressure)

    if loading_mmol_g is None:
        return None

    if abs(temperature_k - temp_ref) > 1.0:
        delta_H = -40000  # J/mol typical for VOCs
        R = 8.314
        correction = np.exp((delta_H / R) * (1 / temperature_k - 1 / temp_ref))
        loading_mmol_g *= correction

    capacity_g_g = loading_mmol_g * molecular_weight / 1000

    return capacity_g_g


def fit_isotherm_from_data(
    pressure_data: list[float],
    loading_data: list[float],
    model_name: str = "Toth",
    temperature_k: float = 298.15,
    material: str = "Activated Carbon",
    adsorbate: str = "VOC",
) -> Optional[dict]:
    """
    Fit an isotherm model to experimental data.

    Args:
        pressure_data: List of pressure values (bar)
        loading_data: List of loading values (mmol/g)
        model_name: Model to fit
        temperature_k: Temperature in K
        material: Material name
        adsorbate: Adsorbate name

    Returns:
        Dict with fitted parameters and statistics, or None if fitting fails
    """
    if not PYGAPS_AVAILABLE:
        return None

    try:
        point_iso = pygaps.PointIsotherm(
            pressure=pressure_data,
            loading=loading_data,
            material=material,
            adsorbate=adsorbate,
            temperature=temperature_k,
            pressure_unit="bar",
            loading_unit="mmol/g",
        )

        model_iso = pygaps.ModelIsotherm.from_pointisotherm(
            point_iso,
            model=model_name,
            verbose=False,
        )

        return {
            "model": model_name,
            "params": dict(model_iso.model.params),
            "rmse": model_iso.model.rmse if hasattr(model_iso.model, "rmse") else None,
        }
    except Exception:
        return None


def is_pygaps_available() -> bool:
    """Check if pyGAPS is available."""
    return PYGAPS_AVAILABLE
