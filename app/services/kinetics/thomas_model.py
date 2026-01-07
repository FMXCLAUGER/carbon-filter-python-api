"""
Thomas Model for Breakthrough Curve Prediction

The Thomas model is one of the most widely used models for predicting
breakthrough curves in fixed-bed adsorption columns. It assumes:
- Langmuir isotherm
- Second-order reversible reaction kinetics
- No axial dispersion

Equation:
C/C0 = 1 / (1 + exp(k_Th/Q × (q0×m - C0×Q×t)))

Where:
- C/C0: Dimensionless outlet concentration
- k_Th: Thomas rate constant (mL/(mg·min) or L/(mg·h))
- Q: Volumetric flow rate
- q0: Maximum adsorption capacity (mg/g)
- m: Mass of adsorbent (g)
- C0: Inlet concentration (mg/L or mg/m³)
- t: Time
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


def thomas_model(
    t: np.ndarray,
    k_Th: float,
    q0: float,
    m: float,
    Q: float,
    C0: float
) -> np.ndarray:
    """
    Calculate breakthrough curve using Thomas model.

    Args:
        t: Time array (minutes)
        k_Th: Thomas rate constant (mL/(mg·min))
        q0: Maximum adsorption capacity (mg/g)
        m: Mass of adsorbent (g)
        Q: Volumetric flow rate (mL/min)
        C0: Inlet concentration (mg/L)

    Returns:
        Array of C/C0 values at each time point
    """
    # Calculate exponent
    exponent = (k_Th / Q) * (q0 * m - C0 * Q * t)

    # Clip to avoid overflow
    exponent = np.clip(exponent, -50, 50)

    # Thomas equation
    C_ratio = 1.0 / (1.0 + np.exp(exponent))

    return C_ratio


def estimate_thomas_params(
    q0: float,
    rho_bulk: float,
    EBCT: float,
    C0: float,
    v: float,
    d_p: float,
    pollutant_type: str = "VOC"
) -> Dict:
    """
    Estimate Thomas model parameters from operating conditions.

    Args:
        q0: Equilibrium adsorption capacity (kg/kg or g/g)
        rho_bulk: Bulk density of carbon (kg/m³)
        EBCT: Empty Bed Contact Time (s)
        C0: Inlet concentration (mg/m³)
        v: Superficial velocity (m/s)
        d_p: Particle diameter (m)
        pollutant_type: Type of pollutant for k_Th estimation

    Returns:
        Dictionary with estimated parameters
    """
    # Empirical k_Th values for different pollutant types (mL/(mg·min))
    # These are typical ranges from literature
    K_TH_TYPICAL = {
        "VOC": {"min": 0.01, "max": 0.1, "default": 0.05},
        "VOC_TOLUENE": {"min": 0.02, "max": 0.08, "default": 0.04},
        "VOC_BENZENE": {"min": 0.02, "max": 0.07, "default": 0.035},
        "VOC_ACETONE": {"min": 0.03, "max": 0.12, "default": 0.06},
        "H2S": {"min": 0.05, "max": 0.2, "default": 0.1},
        "NH3": {"min": 0.01, "max": 0.05, "default": 0.025},
        "default": {"min": 0.02, "max": 0.1, "default": 0.05}
    }

    # Get k_Th range for pollutant type
    k_params = K_TH_TYPICAL.get(pollutant_type, K_TH_TYPICAL["default"])

    # Estimate k_Th based on velocity and particle size
    # Higher velocity -> higher k_Th
    # Smaller particles -> higher k_Th
    velocity_factor = (v / 0.3) ** 0.5  # Normalized to 0.3 m/s
    particle_factor = (0.003 / d_p) ** 0.3  # Normalized to 3mm

    k_Th = k_params["default"] * velocity_factor * particle_factor
    k_Th = max(k_params["min"], min(k_params["max"], k_Th))

    # Calculate breakthrough time at 5% (t_5) and 50% (t_50)
    # For Thomas model: t_50 = q0 * m / (C0 * Q)
    # Approximate mass of carbon in bed
    V_bed = EBCT * v  # Bed volume per unit area
    m_per_area = V_bed * rho_bulk  # kg/m² of bed

    # Convert units for calculation
    Q_m3_per_min = v * 60  # m³/(m²·min) = m/min
    C0_kg_m3 = C0 / 1e6  # mg/m³ to kg/m³

    # Time for 50% breakthrough (minutes)
    if C0_kg_m3 > 0 and Q_m3_per_min > 0:
        t_50 = (q0 * m_per_area) / (C0_kg_m3 * Q_m3_per_min)
    else:
        t_50 = float('inf')

    # Time for 5% breakthrough (approximate)
    # At C/C0 = 0.05: exponent ≈ ln(19) ≈ 3
    # t_5 ≈ t_50 - 3*Q/(k_Th*C0)
    if k_Th > 0 and C0 > 0:
        delta_t = 3 * Q_m3_per_min / (k_Th * C0 / 1000)  # Approximate offset
        t_5 = max(0, t_50 - abs(delta_t) * 0.5)
    else:
        t_5 = t_50 * 0.7

    return {
        "k_Th": k_Th,
        "k_Th_unit": "mL/(mg·min)",
        "t_5_min": t_5,
        "t_50_min": t_50,
        "t_95_min": 2 * t_50 - t_5 if t_50 < float('inf') else float('inf'),
        "velocity_factor": velocity_factor,
        "particle_factor": particle_factor
    }


def calculate_thomas_breakthrough(
    flow_rate: float,
    bed_mass: float,
    C0: float,
    q0: float,
    k_Th: Optional[float] = None,
    v: float = 0.3,
    d_p: float = 0.003,
    pollutant_type: str = "VOC",
    n_points: int = 100,
    time_range_factor: float = 2.0
) -> Dict:
    """
    Calculate complete breakthrough curve using Thomas model.

    Args:
        flow_rate: Volumetric flow rate (m³/h)
        bed_mass: Mass of adsorbent (kg)
        C0: Inlet concentration (mg/m³)
        q0: Maximum adsorption capacity (kg/kg)
        k_Th: Thomas rate constant (if None, will be estimated)
        v: Superficial velocity (m/s)
        d_p: Particle diameter (m)
        pollutant_type: Type of pollutant
        n_points: Number of points in breakthrough curve
        time_range_factor: Factor to extend time range beyond breakthrough

    Returns:
        Dictionary with breakthrough curve data and parameters
    """
    # Convert units
    Q_mL_min = flow_rate * 1e6 / 60  # m³/h to mL/min
    m_g = bed_mass * 1000  # kg to g
    C0_mg_L = C0 / 1000  # mg/m³ to mg/L (assuming 1 m³ ≈ 1000 L at ambient)
    q0_mg_g = q0 * 1000  # kg/kg to mg/g (or g/g to mg/g)

    # Estimate k_Th if not provided
    if k_Th is None:
        params = estimate_thomas_params(
            q0=q0,
            rho_bulk=480,  # Default bulk density
            EBCT=5,  # Default EBCT
            C0=C0,
            v=v,
            d_p=d_p,
            pollutant_type=pollutant_type
        )
        k_Th = params["k_Th"]

    # Calculate theoretical breakthrough time (50%)
    if C0_mg_L > 0 and Q_mL_min > 0:
        t_50 = (q0_mg_g * m_g) / (C0_mg_L * Q_mL_min)
    else:
        t_50 = 1000  # Default 1000 minutes

    # Generate time array
    t_max = t_50 * time_range_factor
    t = np.linspace(0, t_max, n_points)

    # Calculate breakthrough curve
    C_C0 = thomas_model(t, k_Th, q0_mg_g, m_g, Q_mL_min, C0_mg_L)

    # Find breakthrough times
    t_5 = np.interp(0.05, C_C0, t) if C_C0[-1] > 0.05 else t[-1]
    t_10 = np.interp(0.10, C_C0, t) if C_C0[-1] > 0.10 else t[-1]
    t_50_actual = np.interp(0.50, C_C0, t) if C_C0[-1] > 0.50 else t[-1]
    t_90 = np.interp(0.90, C_C0, t) if C_C0[-1] > 0.90 else t[-1]

    # Mass transfer zone (MTZ)
    MTZ_time = t_90 - t_10 if t_90 < t[-1] and t_10 > 0 else 0

    # Create breakthrough curve data
    curve_data = [
        {"time_min": float(t[i]), "time_h": float(t[i] / 60), "C_C0": float(C_C0[i])}
        for i in range(n_points)
    ]

    return {
        "model": "Thomas",
        "parameters": {
            "k_Th": k_Th,
            "k_Th_unit": "mL/(mg·min)",
            "q0_mg_g": q0_mg_g,
            "m_g": m_g,
            "Q_mL_min": Q_mL_min,
            "C0_mg_L": C0_mg_L
        },
        "breakthrough_times": {
            "t_5_min": float(t_5),
            "t_5_h": float(t_5 / 60),
            "t_10_min": float(t_10),
            "t_10_h": float(t_10 / 60),
            "t_50_min": float(t_50_actual),
            "t_50_h": float(t_50_actual / 60),
            "t_90_min": float(t_90),
            "t_90_h": float(t_90 / 60),
            "MTZ_min": float(MTZ_time),
            "MTZ_h": float(MTZ_time / 60)
        },
        "curve": curve_data
    }
