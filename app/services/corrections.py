"""
Corrections for Activated Carbon Adsorption Capacity

This module provides correction factors for:
- Humidity effects (Okazaki simple and Manes advanced)
- Temperature effects (Clausius-Clapeyron / Van't Hoff)
- Pressure effects
- Multi-component competition effects

References:
- Okazaki et al. (1978) - Simple humidity correlation
- Manes (1998) - Polanyi potential theory for water competition
- Clausius-Clapeyron - Temperature dependence of adsorption
"""

import math
import numpy as np
from typing import Dict, Optional, Tuple

R = 8.314  # J/(mol·K)


def humidity_correction(rh: float) -> float:
    """
    Calculate humidity correction factor for adsorption capacity.
    Based on Okazaki correlation.
    
    Args:
        rh: Relative humidity (%)
    
    Returns:
        Correction factor (0.4 to 1.0)
    """
    if rh <= 50:
        return 1.0
    elif rh <= 70:
        return 1.0 - 0.01 * (rh - 50)
    elif rh <= 90:
        return 0.8 - 0.02 * (rh - 70)
    else:
        return 0.4


def temperature_correction(
    t_actual: float,
    t_ref: float = 25.0,
    delta_h: float = -30000.0
) -> float:
    """
    Calculate temperature correction factor using Clausius-Clapeyron.
    
    Args:
        t_actual: Actual temperature (°C)
        t_ref: Reference temperature (°C), default 25°C
        delta_h: Heat of adsorption (J/mol), default -30 kJ/mol
    
    Returns:
        Correction factor
    """
    t_actual_k = t_actual + 273.15
    t_ref_k = t_ref + 273.15
    
    exponent = (delta_h / R) * (1/t_actual_k - 1/t_ref_k)
    return np.exp(exponent)


def pressure_correction(p_actual: float, p_ref: float = 101325.0) -> float:
    """
    Simple linear pressure correction.

    Args:
        p_actual: Actual pressure (Pa)
        p_ref: Reference pressure (Pa), default 101325 Pa (1 atm)

    Returns:
        Correction factor
    """
    return p_actual / p_ref


def manes_humidity_correction(
    rh: float,
    temperature: float,
    pollutant_molar_volume: float,
    water_affinity: float = 1.0,
    pore_volume: float = 0.45
) -> Dict:
    """
    Calculate humidity correction using Manes/Polanyi potential theory.

    The Manes approach accounts for competitive adsorption between water
    and organic compounds based on their adsorption potentials.

    Theory:
    - Water adsorption potential: A_w = R×T×ln(1/RH)
    - VOC adsorption potential: A = R×T×ln(P_sat/P)
    - Competition factor based on relative potentials

    Args:
        rh: Relative humidity (%)
        temperature: Temperature (°C)
        pollutant_molar_volume: Molar volume of pollutant (cm³/mol)
        water_affinity: Carbon affinity for water (0.5-2.0, default 1.0)
                       - < 1.0: hydrophobic carbon
                       - > 1.0: hydrophilic carbon (e.g., coconut shell)
        pore_volume: Total pore volume (cm³/g), default 0.45

    Returns:
        Dictionary with correction factor and detailed analysis
    """
    T_K = temperature + 273.15
    rh_fraction = rh / 100.0

    # Prevent division by zero
    if rh_fraction <= 0:
        rh_fraction = 0.001
    if rh_fraction >= 1:
        rh_fraction = 0.999

    # Water molar volume (cm³/mol at 25°C)
    V_water = 18.07  # cm³/mol

    # Calculate water adsorption potential
    # A_w = R × T × ln(1/RH) / V_water
    A_water = R * T_K * math.log(1 / rh_fraction) / V_water  # J/cm³

    # Characteristic energy for water adsorption (empirical, depends on carbon)
    # Typical values: 10-25 kJ/mol
    E_water = 15000 * water_affinity  # J/mol
    beta_water = E_water / V_water  # J/cm³

    # Fractional pore volume occupied by water (Dubinin-like)
    # W_water/W_0 = exp(-(A_water/E_water)^n)
    # Using n=2 (Dubinin-Radushkevich form)
    if A_water > 0 and beta_water > 0:
        theta_water = math.exp(-((A_water / beta_water) ** 2))
    else:
        theta_water = 0

    # Effective pore volume available for VOC
    # Account for water pre-adsorption
    V_available = pore_volume * (1 - theta_water * water_affinity)

    # Correction factor
    correction_factor = V_available / pore_volume
    correction_factor = max(0.2, min(1.0, correction_factor))

    # Detailed breakdown
    return {
        "correction_factor": correction_factor,
        "method": "Manes-Polanyi",
        "details": {
            "water_adsorption_potential_J_cm3": A_water,
            "water_pore_filling_fraction": theta_water,
            "available_pore_volume_cm3_g": V_available,
            "total_pore_volume_cm3_g": pore_volume,
            "water_affinity": water_affinity
        },
        "recommendations": {
            "critical_rh": 65 if water_affinity > 1.0 else 75,
            "action": "Consider air drying" if rh > 70 else "Humidity within acceptable range"
        }
    }


def advanced_humidity_correction(
    rh: float,
    temperature: float,
    pollutant_type: str = "VOC",
    carbon_type: str = "COAL",
    method: str = "auto"
) -> Dict:
    """
    Unified humidity correction with automatic method selection.

    Args:
        rh: Relative humidity (%)
        temperature: Temperature (°C)
        pollutant_type: Type of pollutant (VOC, H2S, NH3, etc.)
        carbon_type: Type of carbon (COAL, COCONUT, WOOD, etc.)
        method: Correction method ("okazaki", "manes", or "auto")

    Returns:
        Dictionary with correction factor and method used
    """
    # Pollutant molar volumes (cm³/mol)
    MOLAR_VOLUMES = {
        "VOC": 100,
        "VOC_TOLUENE": 106.8,
        "VOC_BENZENE": 89.4,
        "VOC_ACETONE": 74.0,
        "VOC_ETHANOL": 58.5,
        "H2S": 35.9,
        "NH3": 25.0,
        "SO2": 43.8,
        "NO2": 33.8,
        "default": 80
    }

    # Carbon water affinity based on type
    WATER_AFFINITY = {
        "COCONUT": 1.3,  # More hydrophilic
        "COAL": 1.0,
        "WOOD": 1.1,
        "PEAT": 1.2,
        "LIGNITE": 0.9,
        "default": 1.0
    }

    # Select method
    if method == "auto":
        # Use Manes for high humidity or temperature > 40°C
        if rh > 60 or temperature > 40:
            method = "manes"
        else:
            method = "okazaki"

    if method == "okazaki":
        factor = humidity_correction(rh)
        return {
            "correction_factor": factor,
            "method": "Okazaki",
            "simple": True
        }
    else:
        molar_vol = MOLAR_VOLUMES.get(pollutant_type, MOLAR_VOLUMES["default"])
        affinity = WATER_AFFINITY.get(carbon_type, WATER_AFFINITY["default"])

        result = manes_humidity_correction(
            rh=rh,
            temperature=temperature,
            pollutant_molar_volume=molar_vol,
            water_affinity=affinity
        )
        result["simple"] = False
        return result


def temperature_correction_detailed(
    t_actual: float,
    t_ref: float = 25.0,
    delta_h: float = -30000.0,
    pollutant_type: str = "VOC"
) -> Dict:
    """
    Detailed temperature correction with pollutant-specific parameters.

    Uses Clausius-Clapeyron / Van't Hoff equation:
    q(T) = q(T_ref) × exp[(ΔH_ads/R) × (1/T - 1/T_ref)]

    Args:
        t_actual: Actual temperature (°C)
        t_ref: Reference temperature (°C), default 25°C
        delta_h: Heat of adsorption (J/mol), default -30 kJ/mol
                 (if None, estimated from pollutant type)
        pollutant_type: Type of pollutant for ΔH estimation

    Returns:
        Dictionary with correction factor and analysis
    """
    # Typical heats of adsorption (J/mol) - negative values
    DELTA_H_ADS = {
        "VOC": -30000,
        "VOC_TOLUENE": -45000,
        "VOC_BENZENE": -42000,
        "VOC_ACETONE": -35000,
        "VOC_ETHANOL": -40000,
        "H2S": -25000,
        "NH3": -28000,
        "SO2": -35000,
        "NO2": -30000,
        "default": -30000
    }

    # Use pollutant-specific ΔH if not provided
    if delta_h is None or delta_h == -30000.0:
        delta_h = DELTA_H_ADS.get(pollutant_type, DELTA_H_ADS["default"])

    t_actual_k = t_actual + 273.15
    t_ref_k = t_ref + 273.15

    # Clausius-Clapeyron correction
    exponent = (delta_h / R) * (1/t_actual_k - 1/t_ref_k)
    correction_factor = np.exp(exponent)

    # Capacity change percentage
    capacity_change = (correction_factor - 1) * 100

    # Temperature sensitivity
    # d(ln q)/dT = -ΔH/(R×T²)
    sensitivity = -delta_h / (R * t_actual_k ** 2)  # per Kelvin

    return {
        "correction_factor": float(correction_factor),
        "method": "Clausius-Clapeyron",
        "delta_h_J_mol": delta_h,
        "delta_h_kJ_mol": delta_h / 1000,
        "capacity_change_percent": float(capacity_change),
        "sensitivity_per_K": float(sensitivity),
        "analysis": {
            "direction": "decreased" if t_actual > t_ref else "increased",
            "significant": abs(capacity_change) > 10
        },
        "warnings": {
            "high_temp": t_actual > 60,
            "desorption_risk": t_actual > 80,
            "auto_ignition_risk": t_actual > 150
        }
    }


def combined_corrections(
    temperature: float,
    humidity: float,
    pressure: float = 101325.0,
    pollutant_type: str = "VOC",
    carbon_type: str = "COAL",
    t_ref: float = 25.0,
    p_ref: float = 101325.0,
    delta_h: Optional[float] = None
) -> Dict:
    """
    Calculate all corrections and combined factor.

    Args:
        temperature: Operating temperature (°C)
        humidity: Relative humidity (%)
        pressure: Operating pressure (Pa)
        pollutant_type: Type of pollutant
        carbon_type: Type of carbon
        t_ref: Reference temperature (°C)
        p_ref: Reference pressure (Pa)
        delta_h: Heat of adsorption (J/mol)

    Returns:
        Dictionary with all correction factors and combined result
    """
    # Individual corrections
    temp_result = temperature_correction_detailed(
        t_actual=temperature,
        t_ref=t_ref,
        delta_h=delta_h,
        pollutant_type=pollutant_type
    )

    humidity_result = advanced_humidity_correction(
        rh=humidity,
        temperature=temperature,
        pollutant_type=pollutant_type,
        carbon_type=carbon_type
    )

    pressure_factor = pressure_correction(pressure, p_ref)

    # Combined correction factor
    combined_factor = (
        temp_result["correction_factor"] *
        humidity_result["correction_factor"] *
        pressure_factor
    )

    return {
        "combined_factor": float(combined_factor),
        "temperature": {
            "factor": temp_result["correction_factor"],
            "method": temp_result["method"],
            "change_percent": temp_result["capacity_change_percent"]
        },
        "humidity": {
            "factor": humidity_result["correction_factor"],
            "method": humidity_result["method"]
        },
        "pressure": {
            "factor": float(pressure_factor)
        },
        "conditions": {
            "temperature_C": temperature,
            "humidity_percent": humidity,
            "pressure_Pa": pressure
        },
        "warnings": {
            "low_capacity": combined_factor < 0.5,
            "high_temperature": temp_result["warnings"]["high_temp"],
            "desorption_risk": temp_result["warnings"]["desorption_risk"],
            "auto_ignition_risk": temp_result["warnings"]["auto_ignition_risk"]
        },
        "recommendations": []
    }


def estimate_water_loading(
    rh: float,
    temperature: float,
    carbon_type: str = "COAL",
    pore_volume: float = 0.45
) -> Dict:
    """
    Estimate water loading on activated carbon at given conditions.

    Args:
        rh: Relative humidity (%)
        temperature: Temperature (°C)
        carbon_type: Type of carbon
        pore_volume: Total pore volume (cm³/g)

    Returns:
        Dictionary with water loading estimation
    """
    # Water affinity by carbon type
    WATER_AFFINITY = {
        "COCONUT": 1.3,
        "COAL": 1.0,
        "WOOD": 1.1,
        "PEAT": 1.2,
        "LIGNITE": 0.9,
        "default": 1.0
    }

    affinity = WATER_AFFINITY.get(carbon_type, WATER_AFFINITY["default"])

    T_K = temperature + 273.15
    rh_fraction = max(0.001, min(0.999, rh / 100.0))

    # Simplified water isotherm based on RH
    # At high RH, water condensation in micropores
    if rh < 40:
        theta = 0.1 * (rh / 40) * affinity
    elif rh < 70:
        theta = (0.1 + 0.3 * (rh - 40) / 30) * affinity
    else:
        theta = (0.4 + 0.5 * (rh - 70) / 30) * affinity

    theta = min(1.0, theta)

    # Water loading in cm³/g
    water_volume = pore_volume * theta
    # Convert to mass (water density ~1 g/cm³)
    water_mass = water_volume  # g water / g carbon

    return {
        "water_loading_g_g": float(water_mass),
        "water_loading_percent": float(water_mass * 100),
        "pore_filling_fraction": float(theta),
        "available_pore_volume_cm3_g": float(pore_volume * (1 - theta)),
        "conditions": {
            "rh_percent": rh,
            "temperature_C": temperature,
            "carbon_type": carbon_type
        }
    }
