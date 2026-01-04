import numpy as np

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
