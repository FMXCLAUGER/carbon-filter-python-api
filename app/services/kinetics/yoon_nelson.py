"""
Yoon-Nelson Model for Breakthrough Curve Prediction

The Yoon-Nelson model is a simpler alternative to Thomas model that doesn't
require detailed data about adsorbate characteristics, adsorbent type, or
physical properties of the adsorption bed.

Equation:
C/C0 = exp(k_YN × (t - τ)) / (1 + exp(k_YN × (t - τ)))

Or equivalently:
C/C0 = 1 / (1 + exp(k_YN × (τ - t)))

Where:
- C/C0: Dimensionless outlet concentration
- k_YN: Yoon-Nelson rate constant (min⁻¹)
- t: Time (min)
- τ (tau): Time required for 50% breakthrough (min)

Key advantages:
- Simpler model requiring fewer parameters
- τ directly gives the time for 50% breakthrough
- Works well for symmetric breakthrough curves
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


def yoon_nelson_model(
    t: np.ndarray,
    k_YN: float,
    tau: float
) -> np.ndarray:
    """
    Calculate breakthrough curve using Yoon-Nelson model.

    Args:
        t: Time array (minutes)
        k_YN: Yoon-Nelson rate constant (min⁻¹)
        tau: Time for 50% breakthrough (min)

    Returns:
        Array of C/C0 values at each time point
    """
    # Calculate exponent
    exponent = k_YN * (tau - t)

    # Clip to avoid overflow
    exponent = np.clip(exponent, -50, 50)

    # Yoon-Nelson equation
    C_ratio = 1.0 / (1.0 + np.exp(exponent))

    return C_ratio


def estimate_yoon_nelson_params(
    q0: float,
    rho_bulk: float,
    EBCT: float,
    C0: float,
    v: float,
    d_p: float,
    pollutant_type: str = "VOC"
) -> Dict:
    """
    Estimate Yoon-Nelson model parameters from operating conditions.

    Args:
        q0: Equilibrium adsorption capacity (kg/kg or g/g)
        rho_bulk: Bulk density of carbon (kg/m³)
        EBCT: Empty Bed Contact Time (s)
        C0: Inlet concentration (mg/m³)
        v: Superficial velocity (m/s)
        d_p: Particle diameter (m)
        pollutant_type: Type of pollutant for k_YN estimation

    Returns:
        Dictionary with estimated parameters
    """
    # Empirical k_YN values for different pollutant types (min⁻¹)
    # These are typical ranges from literature
    K_YN_TYPICAL = {
        "VOC": {"min": 0.005, "max": 0.05, "default": 0.02},
        "VOC_TOLUENE": {"min": 0.008, "max": 0.04, "default": 0.018},
        "VOC_BENZENE": {"min": 0.007, "max": 0.035, "default": 0.015},
        "VOC_ACETONE": {"min": 0.01, "max": 0.06, "default": 0.025},
        "H2S": {"min": 0.02, "max": 0.1, "default": 0.05},
        "NH3": {"min": 0.005, "max": 0.03, "default": 0.012},
        "default": {"min": 0.008, "max": 0.05, "default": 0.02}
    }

    # Get k_YN range for pollutant type
    k_params = K_YN_TYPICAL.get(pollutant_type, K_YN_TYPICAL["default"])

    # Estimate k_YN based on velocity and particle size
    # Higher velocity -> higher k_YN (faster mass transfer)
    # Smaller particles -> higher k_YN (larger surface area)
    velocity_factor = (v / 0.3) ** 0.4  # Normalized to 0.3 m/s
    particle_factor = (0.003 / d_p) ** 0.25  # Normalized to 3mm

    k_YN = k_params["default"] * velocity_factor * particle_factor
    k_YN = max(k_params["min"], min(k_params["max"], k_YN))

    # Calculate tau (time for 50% breakthrough)
    # τ = (q0 × m) / (C0 × Q) where m is bed mass and Q is flow rate
    # For a unit bed: τ = (q0 × ρ_bulk × EBCT) / C0

    V_bed = EBCT * v  # Bed volume per unit area (m³/m²)
    m_per_area = V_bed * rho_bulk  # kg/m² of bed

    # Convert units
    Q_m3_per_min = v * 60  # m³/(m²·min) = m/min
    C0_kg_m3 = C0 / 1e6  # mg/m³ to kg/m³

    # Time for 50% breakthrough (minutes)
    if C0_kg_m3 > 0 and Q_m3_per_min > 0:
        tau = (q0 * m_per_area) / (C0_kg_m3 * Q_m3_per_min)
    else:
        tau = float('inf')

    # Time for 5% and 95% breakthrough
    # At C/C0 = 0.05: exp(k_YN × (τ - t)) = 19 → t = τ - ln(19)/k_YN
    # At C/C0 = 0.95: exp(k_YN × (τ - t)) = 1/19 → t = τ + ln(19)/k_YN
    if k_YN > 0 and tau < float('inf'):
        delta_t = math.log(19) / k_YN
        t_5 = max(0, tau - delta_t)
        t_95 = tau + delta_t
    else:
        t_5 = tau * 0.7 if tau < float('inf') else 0
        t_95 = tau * 1.3 if tau < float('inf') else float('inf')

    return {
        "k_YN": k_YN,
        "k_YN_unit": "min⁻¹",
        "tau_min": tau,
        "tau_h": tau / 60 if tau < float('inf') else float('inf'),
        "t_5_min": t_5,
        "t_5_h": t_5 / 60,
        "t_95_min": t_95,
        "t_95_h": t_95 / 60 if t_95 < float('inf') else float('inf'),
        "velocity_factor": velocity_factor,
        "particle_factor": particle_factor
    }


def calculate_yoon_nelson_breakthrough(
    flow_rate: float,
    bed_mass: float,
    C0: float,
    q0: float,
    k_YN: Optional[float] = None,
    tau: Optional[float] = None,
    v: float = 0.3,
    d_p: float = 0.003,
    pollutant_type: str = "VOC",
    n_points: int = 100,
    time_range_factor: float = 2.0
) -> Dict:
    """
    Calculate complete breakthrough curve using Yoon-Nelson model.

    Args:
        flow_rate: Volumetric flow rate (m³/h)
        bed_mass: Mass of adsorbent (kg)
        C0: Inlet concentration (mg/m³)
        q0: Maximum adsorption capacity (kg/kg)
        k_YN: Yoon-Nelson rate constant (if None, will be estimated)
        tau: Time for 50% breakthrough (if None, will be calculated)
        v: Superficial velocity (m/s)
        d_p: Particle diameter (m)
        pollutant_type: Type of pollutant
        n_points: Number of points in breakthrough curve
        time_range_factor: Factor to extend time range beyond breakthrough

    Returns:
        Dictionary with breakthrough curve data and parameters
    """
    # Convert units for calculation
    Q_m3_min = flow_rate / 60  # m³/h to m³/min
    m_g = bed_mass * 1000  # kg to g
    q0_mg_g = q0 * 1000  # kg/kg to mg/g (or g/g to mg/g)
    C0_mg_L = C0 / 1000  # mg/m³ to mg/L

    # Calculate tau if not provided
    if tau is None:
        if C0_mg_L > 0 and Q_m3_min > 0:
            # τ = (q0 × m) / (C0 × Q)
            tau = (q0_mg_g * m_g) / (C0_mg_L * Q_m3_min * 1000)  # in minutes
        else:
            tau = 1000  # Default 1000 minutes

    # Estimate k_YN if not provided
    if k_YN is None:
        params = estimate_yoon_nelson_params(
            q0=q0,
            rho_bulk=480,  # Default bulk density
            EBCT=5,  # Default EBCT
            C0=C0,
            v=v,
            d_p=d_p,
            pollutant_type=pollutant_type
        )
        k_YN = params["k_YN"]

    # Generate time array
    t_max = tau * time_range_factor
    t = np.linspace(0, t_max, n_points)

    # Calculate breakthrough curve
    C_C0 = yoon_nelson_model(t, k_YN, tau)

    # Find breakthrough times by interpolation
    t_5 = np.interp(0.05, C_C0, t) if C_C0[-1] > 0.05 else t[-1]
    t_10 = np.interp(0.10, C_C0, t) if C_C0[-1] > 0.10 else t[-1]
    t_50_actual = np.interp(0.50, C_C0, t) if C_C0[-1] > 0.50 else t[-1]
    t_90 = np.interp(0.90, C_C0, t) if C_C0[-1] > 0.90 else t[-1]

    # Mass transfer zone (MTZ) time
    MTZ_time = t_90 - t_10 if t_90 < t[-1] and t_10 > 0 else 0

    # Sharpness of breakthrough (related to k_YN)
    # Larger k_YN = sharper breakthrough
    sharpness = "sharp" if k_YN > 0.03 else "moderate" if k_YN > 0.015 else "gradual"

    # Create breakthrough curve data
    curve_data = [
        {"time_min": float(t[i]), "time_h": float(t[i] / 60), "C_C0": float(C_C0[i])}
        for i in range(n_points)
    ]

    return {
        "model": "Yoon-Nelson",
        "parameters": {
            "k_YN": k_YN,
            "k_YN_unit": "min⁻¹",
            "tau_min": tau,
            "tau_h": tau / 60,
            "q0_mg_g": q0_mg_g,
            "m_g": m_g,
            "Q_m3_min": Q_m3_min,
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
        "analysis": {
            "sharpness": sharpness,
            "symmetry": "symmetric",  # Yoon-Nelson produces symmetric curves
            "model_validity": "The Yoon-Nelson model is best for symmetric breakthrough curves"
        },
        "curve": curve_data
    }


def compare_with_experimental(
    t_exp: np.ndarray,
    C_C0_exp: np.ndarray,
    k_YN: float,
    tau: float
) -> Dict:
    """
    Compare Yoon-Nelson model predictions with experimental data.

    Args:
        t_exp: Experimental time array (minutes)
        C_C0_exp: Experimental C/C0 values
        k_YN: Yoon-Nelson rate constant (min⁻¹)
        tau: Time for 50% breakthrough (min)

    Returns:
        Dictionary with comparison metrics
    """
    # Calculate model predictions at experimental time points
    C_C0_model = yoon_nelson_model(t_exp, k_YN, tau)

    # Calculate error metrics
    residuals = C_C0_exp - C_C0_model
    SSE = np.sum(residuals ** 2)
    SST = np.sum((C_C0_exp - np.mean(C_C0_exp)) ** 2)
    R_squared = 1 - SSE / SST if SST > 0 else 0

    RMSE = np.sqrt(np.mean(residuals ** 2))
    MAE = np.mean(np.abs(residuals))

    return {
        "R_squared": float(R_squared),
        "RMSE": float(RMSE),
        "MAE": float(MAE),
        "SSE": float(SSE),
        "n_points": len(t_exp),
        "model_quality": "good" if R_squared > 0.95 else "moderate" if R_squared > 0.85 else "poor"
    }


def fit_yoon_nelson(
    t_exp: np.ndarray,
    C_C0_exp: np.ndarray,
    k_YN_initial: float = 0.02,
    tau_initial: Optional[float] = None
) -> Dict:
    """
    Fit Yoon-Nelson model parameters to experimental data.

    Args:
        t_exp: Experimental time array (minutes)
        C_C0_exp: Experimental C/C0 values
        k_YN_initial: Initial guess for k_YN
        tau_initial: Initial guess for tau (if None, estimated from data)

    Returns:
        Dictionary with fitted parameters and statistics
    """
    from scipy.optimize import curve_fit

    # Estimate tau from data if not provided
    if tau_initial is None:
        # Find time closest to C/C0 = 0.5
        idx_50 = np.argmin(np.abs(C_C0_exp - 0.5))
        tau_initial = t_exp[idx_50] if idx_50 < len(t_exp) else t_exp[-1] / 2

    # Define fitting function
    def yoon_nelson_fit(t, k_YN, tau):
        exp_term = k_YN * (tau - t)
        exp_term = np.clip(exp_term, -50, 50)
        return 1.0 / (1.0 + np.exp(exp_term))

    try:
        # Perform curve fitting
        popt, pcov = curve_fit(
            yoon_nelson_fit,
            t_exp,
            C_C0_exp,
            p0=[k_YN_initial, tau_initial],
            bounds=([0.001, 0], [1.0, t_exp[-1] * 2]),
            maxfev=5000
        )

        k_YN_fit, tau_fit = popt
        k_YN_std, tau_std = np.sqrt(np.diag(pcov))

        # Calculate goodness of fit
        comparison = compare_with_experimental(t_exp, C_C0_exp, k_YN_fit, tau_fit)

        return {
            "success": True,
            "k_YN": float(k_YN_fit),
            "k_YN_std": float(k_YN_std),
            "tau_min": float(tau_fit),
            "tau_std": float(tau_std),
            **comparison
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "k_YN": k_YN_initial,
            "tau_min": tau_initial
        }
