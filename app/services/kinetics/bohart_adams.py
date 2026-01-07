"""
Bohart-Adams Model for Breakthrough Curve Prediction

The Bohart-Adams model was originally developed for the adsorption of
chlorine on activated carbon and is particularly useful for describing
the initial part of the breakthrough curve.

Equation:
ln(C/C0) = k_BA × C0 × t - k_BA × N0 × Z / v

Or equivalently:
C/C0 = exp(k_BA × C0 × t) / exp(k_BA × N0 × Z / v)

Rearranged:
C/C0 = exp(k_BA × (C0 × t - N0 × Z / v))

More common form:
C/C0 = exp(k_BA × C0 × t - k_BA × N0 × Z / v)
     = exp(k_BA × C0 × (t - t_stoich))

Where:
- C/C0: Dimensionless outlet concentration
- k_BA: Bohart-Adams rate constant (m³/(kg·s) or L/(mg·min))
- C0: Inlet concentration (kg/m³ or mg/L)
- t: Time (s or min)
- N0: Maximum volumetric adsorption capacity (kg/m³)
- Z: Bed height (m)
- v: Linear velocity (m/s)

Key features:
- Best for describing the initial portion of breakthrough
- Assumes constant separation factor
- Useful for estimating service time and column capacity
"""

import math
import numpy as np
from typing import Dict, List, Tuple, Optional


def bohart_adams_model(
    t: np.ndarray,
    k_BA: float,
    N0: float,
    Z: float,
    v: float,
    C0: float
) -> np.ndarray:
    """
    Calculate breakthrough curve using Bohart-Adams model.

    Args:
        t: Time array (minutes)
        k_BA: Bohart-Adams rate constant (L/(mg·min))
        N0: Maximum volumetric adsorption capacity (mg/L)
        Z: Bed height (m)
        v: Linear velocity (m/min)
        C0: Inlet concentration (mg/L)

    Returns:
        Array of C/C0 values at each time point
    """
    # Stoichiometric time (time for complete saturation at plug flow)
    t_stoich = N0 * Z / (v * C0)

    # Calculate exponent
    exponent = k_BA * C0 * (t - t_stoich)

    # Clip to avoid overflow
    exponent = np.clip(exponent, -50, 50)

    # Bohart-Adams equation
    # C/C0 = exp(exponent) / (1 + exp(exponent))
    # This form ensures C/C0 stays between 0 and 1
    C_ratio = np.exp(exponent) / (1.0 + np.exp(exponent))

    return C_ratio


def bohart_adams_linear(
    t: np.ndarray,
    k_BA: float,
    N0: float,
    Z: float,
    v: float,
    C0: float
) -> np.ndarray:
    """
    Original Bohart-Adams equation (linear form, can exceed bounds).

    This is the original formulation which works well for the initial
    part of the breakthrough curve (C/C0 < 0.15) but can give values
    outside [0, 1] range.

    Args:
        t: Time array (minutes)
        k_BA: Bohart-Adams rate constant (L/(mg·min))
        N0: Maximum volumetric adsorption capacity (mg/L)
        Z: Bed height (m)
        v: Linear velocity (m/min)
        C0: Inlet concentration (mg/L)

    Returns:
        Array of ln(C/C0) values (use exp() to get C/C0)
    """
    ln_C_C0 = k_BA * C0 * t - k_BA * N0 * Z / v
    return ln_C_C0


def estimate_bohart_adams_params(
    q0: float,
    rho_bulk: float,
    EBCT: float,
    C0: float,
    v: float,
    d_p: float,
    Z: float = 1.0,
    pollutant_type: str = "VOC"
) -> Dict:
    """
    Estimate Bohart-Adams model parameters from operating conditions.

    Args:
        q0: Equilibrium adsorption capacity (kg/kg or g/g)
        rho_bulk: Bulk density of carbon (kg/m³)
        EBCT: Empty Bed Contact Time (s)
        C0: Inlet concentration (mg/m³)
        v: Superficial velocity (m/s)
        d_p: Particle diameter (m)
        Z: Bed height (m)
        pollutant_type: Type of pollutant for k_BA estimation

    Returns:
        Dictionary with estimated parameters
    """
    # Empirical k_BA values for different pollutant types (L/(mg·min))
    # These are typical ranges from literature
    K_BA_TYPICAL = {
        "VOC": {"min": 0.0001, "max": 0.005, "default": 0.001},
        "VOC_TOLUENE": {"min": 0.0002, "max": 0.004, "default": 0.0015},
        "VOC_BENZENE": {"min": 0.0002, "max": 0.003, "default": 0.0012},
        "VOC_ACETONE": {"min": 0.0003, "max": 0.006, "default": 0.002},
        "H2S": {"min": 0.0005, "max": 0.01, "default": 0.003},
        "NH3": {"min": 0.0001, "max": 0.002, "default": 0.0005},
        "default": {"min": 0.0002, "max": 0.005, "default": 0.001}
    }

    # Get k_BA range for pollutant type
    k_params = K_BA_TYPICAL.get(pollutant_type, K_BA_TYPICAL["default"])

    # Estimate k_BA based on velocity and particle size
    # Higher velocity -> higher k_BA
    # Smaller particles -> higher k_BA
    velocity_factor = (v / 0.3) ** 0.5  # Normalized to 0.3 m/s
    particle_factor = (0.003 / d_p) ** 0.35  # Normalized to 3mm

    k_BA = k_params["default"] * velocity_factor * particle_factor
    k_BA = max(k_params["min"], min(k_params["max"], k_BA))

    # Calculate N0 (volumetric adsorption capacity)
    # N0 = q0 × ρ_bulk (in consistent units)
    # q0 is in kg/kg, ρ_bulk is in kg/m³
    # N0 in kg/m³ = mg/L equivalent
    N0 = q0 * rho_bulk * 1000  # Convert to mg/L (approximation for gases)

    # Linear velocity in m/min
    v_min = v * 60

    # Convert C0 from mg/m³ to mg/L (approximate for ambient conditions)
    C0_mg_L = C0 / 1000

    # Stoichiometric breakthrough time
    if C0_mg_L > 0 and v_min > 0:
        t_stoich = N0 * Z / (v_min * C0_mg_L)
    else:
        t_stoich = float('inf')

    # Time for 5% and 50% breakthrough
    # At C/C0 = 0.05: ln(0.05/0.95) = k_BA × C0 × (t_5 - t_stoich)
    # t_5 ≈ t_stoich + ln(0.05/0.95) / (k_BA × C0)
    if k_BA > 0 and C0_mg_L > 0 and t_stoich < float('inf'):
        delta_5 = math.log(0.05 / 0.95) / (k_BA * C0_mg_L)
        delta_50 = 0  # At t = t_stoich, C/C0 = 0.5
        t_5 = max(0, t_stoich + delta_5)
        t_50 = t_stoich
        t_95 = t_stoich - delta_5  # Symmetric around t_stoich
    else:
        t_5 = t_stoich * 0.7 if t_stoich < float('inf') else 0
        t_50 = t_stoich
        t_95 = t_stoich * 1.3 if t_stoich < float('inf') else float('inf')

    return {
        "k_BA": k_BA,
        "k_BA_unit": "L/(mg·min)",
        "N0": N0,
        "N0_unit": "mg/L",
        "Z": Z,
        "Z_unit": "m",
        "v_min": v_min,
        "v_unit": "m/min",
        "t_stoich_min": t_stoich,
        "t_stoich_h": t_stoich / 60 if t_stoich < float('inf') else float('inf'),
        "t_5_min": t_5,
        "t_50_min": t_50,
        "t_95_min": t_95 if t_95 < float('inf') else float('inf'),
        "velocity_factor": velocity_factor,
        "particle_factor": particle_factor
    }


def calculate_bohart_adams_breakthrough(
    flow_rate: float,
    bed_mass: float,
    bed_height: float,
    bed_area: float,
    C0: float,
    q0: float,
    rho_bulk: float = 480.0,
    k_BA: Optional[float] = None,
    v: float = 0.3,
    d_p: float = 0.003,
    pollutant_type: str = "VOC",
    n_points: int = 100,
    time_range_factor: float = 2.0
) -> Dict:
    """
    Calculate complete breakthrough curve using Bohart-Adams model.

    Args:
        flow_rate: Volumetric flow rate (m³/h)
        bed_mass: Mass of adsorbent (kg)
        bed_height: Height of the bed (m)
        bed_area: Cross-sectional area of the bed (m²)
        C0: Inlet concentration (mg/m³)
        q0: Maximum adsorption capacity (kg/kg)
        rho_bulk: Bulk density (kg/m³)
        k_BA: Bohart-Adams rate constant (if None, will be estimated)
        v: Superficial velocity (m/s)
        d_p: Particle diameter (m)
        pollutant_type: Type of pollutant
        n_points: Number of points in breakthrough curve
        time_range_factor: Factor to extend time range beyond breakthrough

    Returns:
        Dictionary with breakthrough curve data and parameters
    """
    # Calculate linear velocity
    v_actual = flow_rate / (bed_area * 3600)  # m/s
    v_min = v_actual * 60  # m/min

    # Calculate N0 (volumetric capacity)
    N0 = q0 * rho_bulk * 1000  # mg/L equivalent

    # Convert C0 from mg/m³ to mg/L
    C0_mg_L = C0 / 1000

    # Estimate k_BA if not provided
    if k_BA is None:
        params = estimate_bohart_adams_params(
            q0=q0,
            rho_bulk=rho_bulk,
            EBCT=bed_height / v_actual if v_actual > 0 else 5,
            C0=C0,
            v=v_actual,
            d_p=d_p,
            Z=bed_height,
            pollutant_type=pollutant_type
        )
        k_BA = params["k_BA"]

    # Calculate stoichiometric time
    if C0_mg_L > 0 and v_min > 0:
        t_stoich = N0 * bed_height / (v_min * C0_mg_L)
    else:
        t_stoich = 1000  # Default 1000 minutes

    # Generate time array
    t_max = t_stoich * time_range_factor
    t = np.linspace(0, t_max, n_points)

    # Calculate breakthrough curve
    C_C0 = bohart_adams_model(t, k_BA, N0, bed_height, v_min, C0_mg_L)

    # Find breakthrough times by interpolation
    t_5 = np.interp(0.05, C_C0, t) if C_C0[-1] > 0.05 else t[-1]
    t_10 = np.interp(0.10, C_C0, t) if C_C0[-1] > 0.10 else t[-1]
    t_50_actual = np.interp(0.50, C_C0, t) if C_C0[-1] > 0.50 else t[-1]
    t_90 = np.interp(0.90, C_C0, t) if C_C0[-1] > 0.90 else t[-1]

    # Mass transfer zone (MTZ)
    MTZ_time = t_90 - t_10 if t_90 < t[-1] and t_10 > 0 else 0

    # Calculate bed utilization at breakthrough
    # Fraction of bed utilized when C/C0 = 0.05
    bed_utilization = t_5 / t_stoich if t_stoich > 0 and t_stoich < float('inf') else 0

    # Create breakthrough curve data
    curve_data = [
        {"time_min": float(t[i]), "time_h": float(t[i] / 60), "C_C0": float(C_C0[i])}
        for i in range(n_points)
    ]

    return {
        "model": "Bohart-Adams",
        "parameters": {
            "k_BA": k_BA,
            "k_BA_unit": "L/(mg·min)",
            "N0": N0,
            "N0_unit": "mg/L",
            "Z": bed_height,
            "Z_unit": "m",
            "v_min": v_min,
            "v_unit": "m/min",
            "C0_mg_L": C0_mg_L,
            "t_stoich_min": t_stoich,
            "t_stoich_h": t_stoich / 60
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
            "t_stoich_min": float(t_stoich),
            "t_stoich_h": float(t_stoich / 60),
            "MTZ_min": float(MTZ_time),
            "MTZ_h": float(MTZ_time / 60)
        },
        "analysis": {
            "bed_utilization_at_5pct": float(bed_utilization),
            "model_validity": "Best for initial portion of breakthrough (C/C0 < 0.15)",
            "recommended_use": "Design of adsorption columns, estimation of service time"
        },
        "curve": curve_data
    }


def calculate_bed_depth_service_time(
    C0: float,
    v: float,
    k_BA: float,
    N0: float,
    C_b: float = 0.05,
    bed_heights: Optional[List[float]] = None
) -> Dict:
    """
    Calculate service time for different bed depths (BDST analysis).

    The Bed Depth Service Time (BDST) model is a direct application of
    Bohart-Adams for column design.

    Equation:
    t = (N0 × Z) / (C0 × v) - (1 / (k_BA × C0)) × ln(C0/Cb - 1)

    Args:
        C0: Inlet concentration (mg/L)
        v: Linear velocity (m/min)
        k_BA: Bohart-Adams rate constant (L/(mg·min))
        N0: Volumetric adsorption capacity (mg/L)
        C_b: Breakthrough concentration ratio (default 0.05 = 5%)
        bed_heights: List of bed heights to analyze (m)

    Returns:
        Dictionary with BDST analysis results
    """
    if bed_heights is None:
        bed_heights = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

    # BDST slope (service time per unit bed depth)
    slope = N0 / (C0 * v)  # min/m

    # BDST intercept (time lag due to mass transfer resistance)
    if k_BA > 0 and C0 > 0 and C_b > 0:
        intercept = -(1 / (k_BA * C0)) * math.log(C0 / (C_b * C0) - 1)
        intercept = -(1 / (k_BA * C0)) * math.log((1 - C_b) / C_b)
    else:
        intercept = 0

    # Critical bed depth (minimum bed depth for any adsorption)
    Z_critical = intercept * C0 * v / N0 if N0 > 0 else 0

    # Calculate service times for each bed height
    bdst_data = []
    for Z in bed_heights:
        if Z > Z_critical:
            t_service = slope * Z + intercept
        else:
            t_service = 0

        bdst_data.append({
            "bed_height_m": Z,
            "service_time_min": max(0, t_service),
            "service_time_h": max(0, t_service / 60),
            "service_time_days": max(0, t_service / 1440)
        })

    return {
        "model": "BDST (Bed Depth Service Time)",
        "parameters": {
            "slope_min_per_m": slope,
            "intercept_min": intercept,
            "Z_critical_m": Z_critical,
            "breakthrough_ratio": C_b
        },
        "equation": f"t = {slope:.4f} × Z + ({intercept:.4f})",
        "interpretation": {
            "slope": "Service time gained per meter of bed depth",
            "intercept": "Time lag due to mass transfer (negative = time required to establish concentration front)",
            "Z_critical": "Minimum bed depth required for any adsorption"
        },
        "data": bdst_data
    }


def fit_bohart_adams(
    t_exp: np.ndarray,
    C_C0_exp: np.ndarray,
    Z: float,
    v: float,
    C0: float,
    k_BA_initial: float = 0.001,
    N0_initial: Optional[float] = None
) -> Dict:
    """
    Fit Bohart-Adams model parameters to experimental data.

    Args:
        t_exp: Experimental time array (minutes)
        C_C0_exp: Experimental C/C0 values
        Z: Bed height (m)
        v: Linear velocity (m/min)
        C0: Inlet concentration (mg/L)
        k_BA_initial: Initial guess for k_BA
        N0_initial: Initial guess for N0 (if None, estimated from data)

    Returns:
        Dictionary with fitted parameters and statistics
    """
    from scipy.optimize import curve_fit

    # Estimate N0 from data if not provided
    if N0_initial is None:
        # Find time at C/C0 = 0.5 (stoichiometric time)
        idx_50 = np.argmin(np.abs(C_C0_exp - 0.5))
        t_stoich = t_exp[idx_50] if idx_50 < len(t_exp) else t_exp[-1] / 2
        # N0 = t_stoich × v × C0 / Z
        N0_initial = t_stoich * v * C0 / Z if Z > 0 else 1000

    # Define fitting function
    def ba_fit(t, k_BA, N0):
        t_stoich = N0 * Z / (v * C0) if (v > 0 and C0 > 0) else t[-1]
        exponent = k_BA * C0 * (t - t_stoich)
        exponent = np.clip(exponent, -50, 50)
        return np.exp(exponent) / (1.0 + np.exp(exponent))

    try:
        # Perform curve fitting
        popt, pcov = curve_fit(
            ba_fit,
            t_exp,
            C_C0_exp,
            p0=[k_BA_initial, N0_initial],
            bounds=([1e-6, 1], [1.0, N0_initial * 10]),
            maxfev=5000
        )

        k_BA_fit, N0_fit = popt
        k_BA_std, N0_std = np.sqrt(np.diag(pcov))

        # Calculate goodness of fit
        C_C0_model = ba_fit(t_exp, k_BA_fit, N0_fit)
        residuals = C_C0_exp - C_C0_model
        SSE = np.sum(residuals ** 2)
        SST = np.sum((C_C0_exp - np.mean(C_C0_exp)) ** 2)
        R_squared = 1 - SSE / SST if SST > 0 else 0
        RMSE = np.sqrt(np.mean(residuals ** 2))

        # Calculate derived parameters
        t_stoich_fit = N0_fit * Z / (v * C0) if (v > 0 and C0 > 0) else 0

        return {
            "success": True,
            "k_BA": float(k_BA_fit),
            "k_BA_std": float(k_BA_std),
            "k_BA_unit": "L/(mg·min)",
            "N0": float(N0_fit),
            "N0_std": float(N0_std),
            "N0_unit": "mg/L",
            "t_stoich_min": float(t_stoich_fit),
            "R_squared": float(R_squared),
            "RMSE": float(RMSE),
            "SSE": float(SSE),
            "n_points": len(t_exp),
            "model_quality": "good" if R_squared > 0.95 else "moderate" if R_squared > 0.85 else "poor"
        }

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "k_BA": k_BA_initial,
            "N0": N0_initial
        }
