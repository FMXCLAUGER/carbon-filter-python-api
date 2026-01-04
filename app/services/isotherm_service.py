"""Isotherm models and fitting service"""
import math
from typing import List, Tuple, Dict, Optional
from dataclasses import dataclass
import numpy as np
from scipy.optimize import curve_fit


@dataclass
class IsothermParams:
    """Base class for isotherm parameters"""

    model: str
    params: Dict[str, float]
    r2: float = 0.0
    rmse: float = 0.0


# ==================== ISOTHERM MODELS ====================


def langmuir(p: np.ndarray, q_max: float, K: float) -> np.ndarray:
    """
    Langmuir isotherm model.

    q = q_max * K * p / (1 + K * p)

    Args:
        p: Pressure or concentration
        q_max: Maximum adsorption capacity
        K: Langmuir constant

    Returns:
        Loading q
    """
    return q_max * K * p / (1 + K * p)


def freundlich(p: np.ndarray, K: float, n: float) -> np.ndarray:
    """
    Freundlich isotherm model.

    q = K * p^(1/n)

    Args:
        p: Pressure or concentration
        K: Freundlich constant
        n: Freundlich exponent

    Returns:
        Loading q
    """
    return K * np.power(p, 1 / n)


def sips(p: np.ndarray, q_max: float, K: float, n: float) -> np.ndarray:
    """
    Sips (Langmuir-Freundlich) isotherm model.

    q = q_max * (K*p)^(1/n) / (1 + (K*p)^(1/n))

    Args:
        p: Pressure or concentration
        q_max: Maximum capacity
        K: Sips constant
        n: Heterogeneity parameter

    Returns:
        Loading q
    """
    Kp_n = np.power(K * p, 1 / n)
    return q_max * Kp_n / (1 + Kp_n)


def toth(p: np.ndarray, q_max: float, K: float, t: float) -> np.ndarray:
    """
    Toth isotherm model.

    q = q_max * K * p / (1 + (K*p)^t)^(1/t)

    Args:
        p: Pressure or concentration
        q_max: Maximum capacity
        K: Toth constant
        t: Toth exponent

    Returns:
        Loading q
    """
    return q_max * K * p / np.power(1 + np.power(K * p, t), 1 / t)


def dubinin_radushkevich(
    p: np.ndarray, W_0: float, E: float, p_sat: float
) -> np.ndarray:
    """
    Dubinin-Radushkevich (DR) isotherm for microporous adsorbents.

    W = W_0 * exp(-(RT/E * ln(p_sat/p))^2)

    Args:
        p: Pressure
        W_0: Micropore volume
        E: Characteristic energy
        p_sat: Saturation pressure

    Returns:
        Loading (volume filling)
    """
    R = 8.314  # J/(mol·K)
    T = 298.15  # K (25°C, can be parameterized)

    # Avoid log of zero
    p = np.maximum(p, 1e-10)
    ratio = p_sat / p
    ratio = np.maximum(ratio, 1.0)  # Avoid negative log

    exponent = -np.power(R * T / E * np.log(ratio), 2)
    return W_0 * np.exp(exponent)


def dubinin_astakhov(
    p: np.ndarray, W_0: float, E: float, n: float, p_sat: float
) -> np.ndarray:
    """
    Dubinin-Astakhov (DA) isotherm, generalization of DR.

    W = W_0 * exp(-(RT/E * ln(p_sat/p))^n)

    Args:
        p: Pressure
        W_0: Micropore volume
        E: Characteristic energy
        n: DA exponent (n=2 gives DR)
        p_sat: Saturation pressure

    Returns:
        Loading (volume filling)
    """
    R = 8.314
    T = 298.15

    p = np.maximum(p, 1e-10)
    ratio = p_sat / p
    ratio = np.maximum(ratio, 1.0)

    exponent = -np.power(R * T / E * np.log(ratio), n)
    return W_0 * np.exp(exponent)


# ==================== FITTING FUNCTIONS ====================


def fit_isotherm(
    pressures: np.ndarray,
    loadings: np.ndarray,
    model: str = "langmuir",
    p_sat: float = 101325,
) -> IsothermParams:
    """
    Fit isotherm model to experimental data.

    Args:
        pressures: Array of pressures
        loadings: Array of loadings
        model: Model name
        p_sat: Saturation pressure (for DR/DA models)

    Returns:
        IsothermParams with fitted parameters
    """
    p = np.array(pressures)
    q = np.array(loadings)

    try:
        if model == "langmuir":
            # Initial guess: q_max from max loading, K from initial slope
            q_max_init = max(q) * 1.5
            K_init = q[0] / (p[0] * (q_max_init - q[0])) if p[0] > 0 else 1.0

            popt, _ = curve_fit(
                langmuir,
                p,
                q,
                p0=[q_max_init, K_init],
                bounds=([0, 0], [np.inf, np.inf]),
                maxfev=5000,
            )
            params = {"q_max": popt[0], "K": popt[1]}
            q_pred = langmuir(p, *popt)

        elif model == "freundlich":
            popt, _ = curve_fit(
                freundlich,
                p,
                q,
                p0=[1.0, 2.0],
                bounds=([0, 1], [np.inf, 10]),
                maxfev=5000,
            )
            params = {"K": popt[0], "n": popt[1]}
            q_pred = freundlich(p, *popt)

        elif model == "sips":
            q_max_init = max(q) * 1.5
            popt, _ = curve_fit(
                sips,
                p,
                q,
                p0=[q_max_init, 1e-5, 2.0],
                bounds=([0, 0, 0.5], [np.inf, np.inf, 10]),
                maxfev=5000,
            )
            params = {"q_max": popt[0], "K": popt[1], "n": popt[2]}
            q_pred = sips(p, *popt)

        elif model == "toth":
            q_max_init = max(q) * 1.5
            popt, _ = curve_fit(
                toth,
                p,
                q,
                p0=[q_max_init, 1e-5, 1.0],
                bounds=([0, 0, 0.1], [np.inf, np.inf, 5]),
                maxfev=5000,
            )
            params = {"q_max": popt[0], "K": popt[1], "t": popt[2]}
            q_pred = toth(p, *popt)

        elif model == "dr":
            W_0_init = max(q)

            def dr_fit(p, W_0, E):
                return dubinin_radushkevich(p, W_0, E, p_sat)

            popt, _ = curve_fit(
                dr_fit,
                p,
                q,
                p0=[W_0_init, 15000],
                bounds=([0, 1000], [np.inf, 50000]),
                maxfev=5000,
            )
            params = {"W_0": popt[0], "E": popt[1], "p_sat": p_sat}
            q_pred = dr_fit(p, *popt)

        elif model == "da":
            W_0_init = max(q)

            def da_fit(p, W_0, E, n):
                return dubinin_astakhov(p, W_0, E, n, p_sat)

            popt, _ = curve_fit(
                da_fit,
                p,
                q,
                p0=[W_0_init, 15000, 2.0],
                bounds=([0, 1000, 1], [np.inf, 50000, 5]),
                maxfev=5000,
            )
            params = {"W_0": popt[0], "E": popt[1], "n": popt[2], "p_sat": p_sat}
            q_pred = da_fit(p, *popt)

        else:
            raise ValueError(f"Unknown model: {model}")

        # Calculate fit statistics
        ss_res = np.sum((q - q_pred) ** 2)
        ss_tot = np.sum((q - np.mean(q)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        rmse = np.sqrt(np.mean((q - q_pred) ** 2))

        return IsothermParams(model=model, params=params, r2=r2, rmse=rmse)

    except Exception as e:
        # Return default with zero R²
        return IsothermParams(
            model=model, params={}, r2=0.0, rmse=float("inf")
        )


def predict_loading(
    pressure: float,
    params: IsothermParams,
) -> float:
    """
    Predict loading at given pressure using fitted isotherm.

    Args:
        pressure: Pressure or concentration
        params: Fitted isotherm parameters

    Returns:
        Predicted loading
    """
    p = np.array([pressure])

    if params.model == "langmuir":
        return float(langmuir(p, params.params["q_max"], params.params["K"])[0])
    elif params.model == "freundlich":
        return float(freundlich(p, params.params["K"], params.params["n"])[0])
    elif params.model == "sips":
        return float(
            sips(p, params.params["q_max"], params.params["K"], params.params["n"])[0]
        )
    elif params.model == "toth":
        return float(
            toth(p, params.params["q_max"], params.params["K"], params.params["t"])[0]
        )
    elif params.model == "dr":
        return float(
            dubinin_radushkevich(
                p, params.params["W_0"], params.params["E"], params.params["p_sat"]
            )[0]
        )
    elif params.model == "da":
        return float(
            dubinin_astakhov(
                p,
                params.params["W_0"],
                params.params["E"],
                params.params["n"],
                params.params["p_sat"],
            )[0]
        )
    else:
        raise ValueError(f"Unknown model: {params.model}")


def fit_best_isotherm(
    pressures: np.ndarray,
    loadings: np.ndarray,
    models: List[str] = None,
) -> Tuple[IsothermParams, Dict[str, IsothermParams]]:
    """
    Fit multiple isotherm models and return the best one.

    Args:
        pressures: Array of pressures
        loadings: Array of loadings
        models: List of models to try (default: all)

    Returns:
        Tuple of (best model, all results)
    """
    if models is None:
        models = ["langmuir", "freundlich", "sips", "toth"]

    results = {}
    for model in models:
        results[model] = fit_isotherm(pressures, loadings, model)

    # Find best by R²
    best_model = max(results.keys(), key=lambda m: results[m].r2)

    return results[best_model], results


def estimate_capacity_from_surface_area(
    surface_area: float,
    molecular_weight: float,
    molecular_cross_section: float = 0.35,
) -> float:
    """
    Estimate maximum adsorption capacity from BET surface area.

    Rough estimate using monolayer coverage.

    Args:
        surface_area: BET surface area (m²/g)
        molecular_weight: Molecular weight (g/mol)
        molecular_cross_section: Cross-section area (nm²)

    Returns:
        Estimated maximum capacity (mg/g)
    """
    # Avogadro's number
    N_A = 6.022e23

    # Convert cross-section to m²
    sigma = molecular_cross_section * 1e-18

    # Maximum molecules per gram of adsorbent
    n_max = surface_area / sigma

    # Maximum moles per gram
    mol_max = n_max / N_A

    # Maximum mass in mg/g
    q_max = mol_max * molecular_weight * 1000

    # Typical packing efficiency factor
    efficiency = 0.7

    return q_max * efficiency
