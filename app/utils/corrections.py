"""Thermodynamic corrections for adsorption calculations"""
import math
from typing import Optional
from .constants import R, WATER_VAPOR_PRESSURE, HUMIDITY_THRESHOLD


def clausius_clapeyron_correction(
    q_ref: float,
    T_ref: float,
    T_new: float,
    delta_H: float = -40000,
) -> float:
    """
    Apply Clausius-Clapeyron temperature correction to adsorption capacity.

    The Clausius-Clapeyron equation relates adsorption capacity at different
    temperatures through the isosteric heat of adsorption.

    Args:
        q_ref: Reference adsorption capacity (mg/g) at T_ref
        T_ref: Reference temperature (°C)
        T_new: New temperature (°C)
        delta_H: Isosteric heat of adsorption (J/mol), negative for exothermic

    Returns:
        Corrected adsorption capacity at T_new
    """
    T_ref_K = T_ref + 273.15
    T_new_K = T_new + 273.15

    # Clausius-Clapeyron: ln(q2/q1) = (ΔH/R) * (1/T1 - 1/T2)
    correction_factor = math.exp((delta_H / R) * (1 / T_ref_K - 1 / T_new_K))

    return q_ref * correction_factor


def get_water_vapor_pressure(temperature: float) -> float:
    """
    Get water vapor pressure at a given temperature using Antoine equation.

    Args:
        temperature: Temperature in °C

    Returns:
        Saturation vapor pressure in Pa
    """
    # Antoine equation constants for water (valid 1-100°C)
    A = 8.07131
    B = 1730.63
    C = 233.426

    # Antoine equation gives pressure in mmHg
    log_p_mmhg = A - B / (C + temperature)
    p_mmhg = 10 ** log_p_mmhg

    # Convert to Pa
    return p_mmhg * 133.322


def okazaki_humidity_correction(
    q_dry: float,
    relative_humidity: float,
    temperature: float = 25,
    K_w: float = 0.015,
) -> float:
    """
    Apply Okazaki humidity correction for water vapor competition.

    At high humidity, water competes with VOCs for adsorption sites.
    This correction uses an empirical model based on Okazaki's work.

    Args:
        q_dry: Dry adsorption capacity (mg/g)
        relative_humidity: Relative humidity (%)
        temperature: Temperature (°C)
        K_w: Water competition coefficient (typical 0.01-0.02)

    Returns:
        Corrected adsorption capacity accounting for humidity
    """
    if relative_humidity <= HUMIDITY_THRESHOLD:
        # Below threshold, humidity effect is minimal
        return q_dry

    # Calculate water vapor partial pressure
    p_sat = get_water_vapor_pressure(temperature)
    p_water = (relative_humidity / 100) * p_sat

    # Okazaki correction: q_wet = q_dry / (1 + K_w * p_water)
    correction_factor = 1 / (1 + K_w * p_water / 1000)  # Normalize p_water

    return q_dry * correction_factor


def manes_humidity_correction(
    q_dry: float,
    relative_humidity: float,
    micropore_volume: float,
    temperature: float = 25,
) -> float:
    """
    Apply Manes-Hofer humidity correction based on pore filling.

    This model considers that water preferentially fills micropores,
    reducing the available volume for VOC adsorption.

    Args:
        q_dry: Dry adsorption capacity (mg/g)
        relative_humidity: Relative humidity (%)
        micropore_volume: Micropore volume (cm³/g)
        temperature: Temperature (°C)

    Returns:
        Corrected adsorption capacity
    """
    if relative_humidity <= HUMIDITY_THRESHOLD:
        return q_dry

    # Water molar volume (cm³/mol)
    V_m_water = 18.0

    # Estimate water loading using Dubinin-Serpinsky approach
    RH = relative_humidity / 100

    # Simplified Manes model: fraction of micropores filled by water
    # increases above 40% RH following a sigmoid curve
    if RH > 0.4:
        # Fraction of pores occupied by water
        f_water = 0.2 * ((RH - 0.4) / 0.6) ** 1.5
        f_water = min(f_water, 0.5)  # Cap at 50% pore filling
    else:
        f_water = 0

    return q_dry * (1 - f_water)


def combined_correction(
    q_ref: float,
    T_ref: float,
    T_new: float,
    relative_humidity: float,
    delta_H: float = -40000,
    micropore_volume: Optional[float] = None,
) -> tuple[float, dict]:
    """
    Apply combined temperature and humidity corrections.

    Args:
        q_ref: Reference capacity at T_ref and dry conditions
        T_ref: Reference temperature (°C)
        T_new: New temperature (°C)
        relative_humidity: Relative humidity (%)
        delta_H: Heat of adsorption (J/mol)
        micropore_volume: Micropore volume (cm³/g), optional

    Returns:
        Tuple of (corrected capacity, correction factors dict)
    """
    # Temperature correction
    q_temp = clausius_clapeyron_correction(q_ref, T_ref, T_new, delta_H)
    temp_factor = q_temp / q_ref

    # Humidity correction
    if micropore_volume is not None:
        q_final = manes_humidity_correction(
            q_temp, relative_humidity, micropore_volume, T_new
        )
    else:
        q_final = okazaki_humidity_correction(q_temp, relative_humidity, T_new)

    humidity_factor = q_final / q_temp if q_temp > 0 else 1.0
    total_factor = q_final / q_ref if q_ref > 0 else 1.0

    return q_final, {
        "temperature_factor": temp_factor,
        "humidity_factor": humidity_factor,
        "total_factor": total_factor,
    }
