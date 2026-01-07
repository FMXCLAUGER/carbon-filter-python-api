"""
IAST (Ideal Adsorbed Solution Theory) service for multi-component adsorption.

Calculates competitive adsorption between multiple pollutants using pyGAPS IAST.
"""

from typing import Optional
import numpy as np

from app.services.pygaps_service import (
    PYGAPS_AVAILABLE,
    create_isotherm_model,
    get_default_isotherm_params,
    get_molecular_weight,
    concentration_to_partial_pressure,
)

IAST_AVAILABLE = False

if PYGAPS_AVAILABLE:
    try:
        import pygaps.iast as pgiast
        IAST_AVAILABLE = True
    except ImportError:
        pass


def is_iast_available() -> bool:
    """Check if IAST functionality is available."""
    return IAST_AVAILABLE


def calculate_iast_loadings(
    pollutant_names: list[str],
    partial_pressures: list[float],
    isotherm_params_list: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Calculate competitive loadings using IAST.

    Args:
        pollutant_names: List of pollutant names
        partial_pressures: List of partial pressures (bar)
        isotherm_params_list: Optional list of custom isotherm parameters

    Returns:
        Dict with loadings and selectivities, or None if calculation fails
    """
    if not IAST_AVAILABLE:
        return None

    if len(pollutant_names) < 2:
        return None

    try:
        isotherms = []

        for i, name in enumerate(pollutant_names):
            if isotherm_params_list and i < len(isotherm_params_list):
                params = isotherm_params_list[i]
            else:
                params = get_default_isotherm_params(name)

            if params is None:
                return None

            model = create_isotherm_model(
                model_name=params["model"],
                params=params["params"],
            )

            if model is None:
                return None

            isotherms.append(model)

        loadings = pgiast.iast(
            isotherms=isotherms,
            partial_pressures=partial_pressures,
        )

        pure_loadings = []
        for i, model in enumerate(isotherms):
            pure_q = model.loading(partial_pressures[i])
            pure_loadings.append(float(pure_q))

        reduction_factors = []
        for i in range(len(loadings)):
            if pure_loadings[i] > 0:
                factor = loadings[i] / pure_loadings[i]
            else:
                factor = 1.0
            reduction_factors.append(factor)

        selectivities = {}
        for i in range(len(pollutant_names)):
            for j in range(i + 1, len(pollutant_names)):
                if loadings[j] > 0 and partial_pressures[i] > 0 and partial_pressures[j] > 0:
                    selectivity = (loadings[i] / loadings[j]) * (partial_pressures[j] / partial_pressures[i])
                else:
                    selectivity = None
                key = f"{pollutant_names[i]}/{pollutant_names[j]}"
                selectivities[key] = selectivity

        return {
            "pollutants": pollutant_names,
            "partial_pressures": partial_pressures,
            "iast_loadings": [float(q) for q in loadings],
            "pure_loadings": pure_loadings,
            "reduction_factors": reduction_factors,
            "selectivities": selectivities,
        }

    except Exception as e:
        print(f"IAST calculation error: {e}")
        return None


def calculate_competitive_capacity(
    pollutant_names: list[str],
    concentrations: list[float],
    molecular_weights: list[float],
    temperature_c: float = 25.0,
    isotherm_params_list: Optional[list[dict]] = None,
) -> Optional[dict]:
    """
    Calculate competitive adsorption capacities for multiple pollutants.

    Args:
        pollutant_names: List of pollutant names
        concentrations: List of concentrations (mg/m³)
        molecular_weights: List of molecular weights (g/mol)
        temperature_c: Temperature in °C
        isotherm_params_list: Optional list of custom isotherm parameters

    Returns:
        Dict with competitive capacities in g/g
    """
    if not IAST_AVAILABLE:
        return None

    if len(pollutant_names) != len(concentrations):
        return None

    mw_list = []
    for i, name in enumerate(pollutant_names):
        if i < len(molecular_weights) and molecular_weights[i]:
            mw = molecular_weights[i]
        else:
            mw = get_molecular_weight(name) or 100.0
        mw_list.append(mw)

    partial_pressures = []
    for conc, mw in zip(concentrations, mw_list):
        pp = concentration_to_partial_pressure(conc, mw, temperature_c)
        partial_pressures.append(pp)

    result = calculate_iast_loadings(
        pollutant_names,
        partial_pressures,
        isotherm_params_list,
    )

    if result is None:
        return None

    capacities_g_g = []
    for loading_mmol_g, mw in zip(result["iast_loadings"], mw_list):
        capacity = loading_mmol_g * mw / 1000
        capacities_g_g.append(capacity)

    pure_capacities_g_g = []
    for loading_mmol_g, mw in zip(result["pure_loadings"], mw_list):
        capacity = loading_mmol_g * mw / 1000
        pure_capacities_g_g.append(capacity)

    return {
        "pollutants": pollutant_names,
        "concentrations": concentrations,
        "molecular_weights": mw_list,
        "iast_capacities_g_g": capacities_g_g,
        "pure_capacities_g_g": pure_capacities_g_g,
        "reduction_factors": result["reduction_factors"],
        "selectivities": result["selectivities"],
        "competitive_effect": True,
    }


def estimate_iast_breakthrough_reduction(
    num_pollutants: int,
    avg_reduction_factor: float,
) -> float:
    """
    Estimate breakthrough time reduction due to competitive adsorption.

    Args:
        num_pollutants: Number of pollutants
        avg_reduction_factor: Average capacity reduction factor

    Returns:
        Estimated breakthrough time reduction factor
    """
    return avg_reduction_factor * (1.0 - 0.05 * (num_pollutants - 1))
