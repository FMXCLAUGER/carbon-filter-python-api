import numpy as np
from typing import Tuple, List, Optional

from app.models.schemas import PollutantResult, BreakthroughPoint, IsothermParams
from app.services.corrections import humidity_correction, temperature_correction
from app.services.pygaps_service import (
    predict_capacity_pygaps,
    is_pygaps_available,
    get_default_isotherm_params,
)


# Typical adsorption capacities (g/g) for common pollutants at 25°C, 50% RH (fallback)
TYPICAL_CAPACITIES = {
    "toluene": 0.35,
    "benzene": 0.25,
    "xylene": 0.40,
    "ethylbenzene": 0.38,
    "acetone": 0.15,
    "methanol": 0.10,
    "ethanol": 0.15,
    "formaldehyde": 0.05,
    "h2s": 0.15,
    "ammonia": 0.08,
    "chlorine": 0.20,
    "so2": 0.12,
    "nox": 0.08,
    "styrene": 0.35,
    "phenol": 0.30,
    "default": 0.20,
}

# Typical mass transfer coefficients (1/min)
TYPICAL_KV = {
    "default": 5.0,
    "h2s": 8.0,
    "ammonia": 6.0,
    "toluene": 4.0,
    "benzene": 4.5,
}


def estimate_capacity(
    pollutant_name: str,
    concentration: float,
    temperature: float,
    humidity: float,
    surface_area: float,
    molecular_weight: float | None = None,
    isotherm_params: Optional[IsothermParams] = None,
) -> float:
    """
    Estimate adsorption capacity for a pollutant.

    Uses pyGAPS isotherm models when available, otherwise falls back
    to empirical correlations with T/RH corrections.

    Args:
        pollutant_name: Name of pollutant
        concentration: Inlet concentration (mg/m³)
        temperature: Temperature (°C)
        humidity: Relative humidity (%)
        surface_area: Carbon BET surface area (m²/g)
        molecular_weight: Molecular weight (g/mol)
        isotherm_params: Optional custom isotherm parameters

    Returns:
        Estimated capacity W_e (g/g)
    """
    capacity = None

    if is_pygaps_available():
        params_dict = None
        if isotherm_params is not None:
            params_dict = {
                "model": isotherm_params.model,
                "params": isotherm_params.params,
                "temp_ref": isotherm_params.temp_ref,
            }

        capacity = predict_capacity_pygaps(
            pollutant_name=pollutant_name,
            concentration_mg_m3=concentration,
            temperature_c=temperature,
            molecular_weight=molecular_weight,
            isotherm_params=params_dict,
        )

        if capacity is not None:
            surface_factor = surface_area / 1000.0
            h_factor = humidity_correction(humidity)
            capacity = capacity * surface_factor * h_factor

    if capacity is None:
        name_lower = pollutant_name.lower().replace(" ", "").replace("-", "")
        base_capacity = TYPICAL_CAPACITIES.get(name_lower, TYPICAL_CAPACITIES["default"])

        surface_factor = surface_area / 1000.0
        t_factor = temperature_correction(temperature, t_ref=25.0)
        h_factor = humidity_correction(humidity)

        c_factor = 1.0 + 0.1 * np.log10(max(concentration, 1) / 100)
        c_factor = np.clip(c_factor, 0.8, 1.3)

        capacity = base_capacity * surface_factor * t_factor * h_factor * c_factor

    return max(capacity, 0.01)


def estimate_kv(
    pollutant_name: str,
    velocity: float,
    particle_diameter: float | None = None,
) -> float:
    """
    Estimate mass transfer coefficient k_v.
    
    Args:
        pollutant_name: Name of pollutant
        velocity: Superficial velocity (m/s)
        particle_diameter: Particle diameter (mm)
    
    Returns:
        k_v (1/min)
    """
    name_lower = pollutant_name.lower().replace(" ", "").replace("-", "")
    base_kv = TYPICAL_KV.get(name_lower, TYPICAL_KV["default"])
    
    # Velocity correction (higher velocity = higher k_v)
    velocity_factor = (velocity / 0.3) ** 0.5
    
    # Particle size correction (smaller particles = higher k_v)
    if particle_diameter is not None:
        size_factor = (3.0 / particle_diameter) ** 0.3
    else:
        size_factor = 1.0
    
    return base_kv * velocity_factor * size_factor


def calculate_breakthrough_time(
    W_e: float,
    carbon_mass: float,
    bulk_density: float,
    C_in: float,
    C_out: float,
    flow_rate: float,
    k_v: float,
    bed_height: float,
) -> Tuple[float, float]:
    """
    Calculate breakthrough time using Wheeler-Jonas equation.
    
    t_b = (W_e × M) / (C_in × Q) - (W_e × ρ_b × H) / (k_v × C_in) × ln((C_in - C_out) / C_out)
    
    Args:
        W_e: Equilibrium adsorption capacity (g/g)
        carbon_mass: Mass of carbon bed (kg)
        bulk_density: Bulk density (kg/m³)
        C_in: Inlet concentration (mg/m³)
        C_out: Breakthrough concentration (mg/m³)
        flow_rate: Flow rate (m³/h)
        k_v: Mass transfer coefficient (1/min)
        bed_height: Bed height (m)
    
    Returns:
        Tuple of (breakthrough_time_hours, mass_transfer_zone_m)
    """
    # Convert units
    M_g = carbon_mass * 1000  # kg to g
    C_in_g_m3 = C_in / 1000  # mg/m³ to g/m³
    C_out_g_m3 = C_out / 1000  # mg/m³ to g/m³
    Q_m3_min = flow_rate / 60  # m³/h to m³/min
    
    # Capacity term (minutes)
    term1 = (W_e * M_g) / (C_in_g_m3 * Q_m3_min)
    
    # Kinetic term (minutes)
    if C_in_g_m3 > C_out_g_m3:
        ln_term = np.log((C_in_g_m3 - C_out_g_m3) / C_out_g_m3)
    else:
        ln_term = 0
    
    term2 = (W_e * bulk_density * bed_height) / (k_v * C_in_g_m3) * ln_term
    
    t_b_min = max(term1 - term2, 0)
    t_b_hours = t_b_min / 60
    
    # Mass transfer zone (MTZ)
    # Approximation: MTZ = v × t_mtz where t_mtz ≈ term2 / k_v
    mtz = bed_height * (term2 / term1) if term1 > 0 else 0.1 * bed_height
    mtz = np.clip(mtz, 0.05, bed_height * 0.5)
    
    return t_b_hours, mtz


def generate_breakthrough_curve(
    breakthrough_time: float,
    total_time: float | None = None,
    num_points: int = 50,
) -> List[BreakthroughPoint]:
    """
    Generate S-curve breakthrough data points.
    
    Uses logistic function to model breakthrough curve.
    
    Args:
        breakthrough_time: Time to 5% breakthrough (hours)
        total_time: Total simulation time (hours)
        num_points: Number of data points
    
    Returns:
        List of BreakthroughPoint
    """
    if total_time is None:
        total_time = breakthrough_time * 2.0
    
    times = np.linspace(0, total_time, num_points)
    
    # Logistic function parameters
    # At t = breakthrough_time, C/C0 should be ~0.05 (5%)
    # k controls steepness
    k = 10 / breakthrough_time  # Steepness factor
    t_mid = breakthrough_time * 1.3  # Midpoint (50% breakthrough)
    
    c_c0 = 1 / (1 + np.exp(-k * (times - t_mid)))
    
    return [
        BreakthroughPoint(time=round(t, 2), c_c0=round(c, 4))
        for t, c in zip(times, c_c0)
    ]


def calculate_pollutant_result(
    pollutant_name: str,
    concentration: float,
    target_outlet: float | None,
    temperature: float,
    humidity: float,
    surface_area: float,
    carbon_mass: float,
    bulk_density: float,
    flow_rate: float,
    velocity: float,
    bed_height: float,
    particle_diameter: float | None,
    molecular_weight: float | None,
    isotherm_params: Optional[IsothermParams] = None,
) -> PollutantResult:
    """
    Calculate complete results for a single pollutant.
    """
    W_e = estimate_capacity(
        pollutant_name, concentration, temperature, humidity,
        surface_area, molecular_weight, isotherm_params
    )
    
    # Estimate k_v
    k_v = estimate_kv(pollutant_name, velocity, particle_diameter)
    
    # Determine breakthrough concentration
    if target_outlet is not None and target_outlet > 0:
        C_out = target_outlet
    else:
        C_out = concentration * 0.05  # Default 5% breakthrough
    
    # Calculate breakthrough time
    t_b, mtz = calculate_breakthrough_time(
        W_e, carbon_mass, bulk_density, concentration, C_out,
        flow_rate, k_v, bed_height
    )
    
    # Calculate removal efficiency at breakthrough
    efficiency = (1 - C_out / concentration) * 100
    
    return PollutantResult(
        name=pollutant_name,
        inlet_concentration=concentration,
        outlet_concentration=C_out,
        removal_efficiency=round(efficiency, 1),
        adsorption_capacity=round(W_e, 4),
        breakthrough_time=round(t_b, 1),
        mass_transfer_zone=round(mtz, 3),
    )
