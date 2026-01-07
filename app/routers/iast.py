"""
IAST (Ideal Adsorbed Solution Theory) router for multi-component calculations.
"""

from fastapi import APIRouter, HTTPException

from app.models.schemas import IASTRequest, IASTResponse, PollutantInput
from app.services.iast_service import (
    is_iast_available,
    calculate_competitive_capacity,
)
from app.services.pygaps_service import get_molecular_weight

router = APIRouter(prefix="/api/iast", tags=["iast"])


@router.get("/status")
async def iast_status():
    """Check if IAST functionality is available."""
    return {
        "iast_available": is_iast_available(),
        "description": "IAST calculates competitive adsorption between multiple pollutants",
        "min_pollutants": 2,
    }


@router.post("/calculate", response_model=IASTResponse)
async def calculate_iast(request: IASTRequest):
    """
    Calculate competitive adsorption using IAST.

    Requires at least 2 pollutants. Returns competitive capacities
    and selectivities compared to pure component adsorption.
    """
    if not is_iast_available():
        raise HTTPException(
            status_code=503,
            detail="IAST functionality not available. pyGAPS may not be installed.",
        )

    if len(request.pollutants) < 2:
        raise HTTPException(
            status_code=400,
            detail="IAST requires at least 2 pollutants for competitive calculation.",
        )

    pollutant_names = [p.name for p in request.pollutants]
    concentrations = [p.concentration for p in request.pollutants]

    molecular_weights = []
    for p in request.pollutants:
        if p.molecular_weight:
            molecular_weights.append(p.molecular_weight)
        else:
            mw = get_molecular_weight(p.name)
            molecular_weights.append(mw if mw else 100.0)

    isotherm_params_list = None
    if any(p.isotherm_params for p in request.pollutants):
        isotherm_params_list = []
        for p in request.pollutants:
            if p.isotherm_params:
                isotherm_params_list.append({
                    "model": p.isotherm_params.model,
                    "params": p.isotherm_params.params,
                    "temp_ref": p.isotherm_params.temp_ref,
                })
            else:
                isotherm_params_list.append(None)

    result = calculate_competitive_capacity(
        pollutant_names=pollutant_names,
        concentrations=concentrations,
        molecular_weights=molecular_weights,
        temperature_c=request.temperature,
        isotherm_params_list=isotherm_params_list,
    )

    if result is None:
        raise HTTPException(
            status_code=400,
            detail="IAST calculation failed. Check pollutant names and parameters.",
        )

    return IASTResponse(
        pollutants=result["pollutants"],
        concentrations=result["concentrations"],
        molecular_weights=result["molecular_weights"],
        iast_capacities_g_g=result["iast_capacities_g_g"],
        pure_capacities_g_g=result["pure_capacities_g_g"],
        reduction_factors=result["reduction_factors"],
        selectivities=result["selectivities"],
        competitive_effect=result["competitive_effect"],
    )


def get_iast_for_pollutants(
    pollutants: list[PollutantInput],
    temperature: float = 25.0,
) -> IASTResponse | None:
    """
    Helper function to get IAST results for a list of pollutants.
    Returns None if IAST is not available or calculation fails.
    """
    if not is_iast_available() or len(pollutants) < 2:
        return None

    pollutant_names = [p.name for p in pollutants]
    concentrations = [p.concentration for p in pollutants]

    molecular_weights = []
    for p in pollutants:
        if p.molecular_weight:
            molecular_weights.append(p.molecular_weight)
        else:
            mw = get_molecular_weight(p.name)
            molecular_weights.append(mw if mw else 100.0)

    result = calculate_competitive_capacity(
        pollutant_names=pollutant_names,
        concentrations=concentrations,
        molecular_weights=molecular_weights,
        temperature_c=temperature,
    )

    if result is None:
        return None

    return IASTResponse(
        pollutants=result["pollutants"],
        concentrations=result["concentrations"],
        molecular_weights=result["molecular_weights"],
        iast_capacities_g_g=result["iast_capacities_g_g"],
        pure_capacities_g_g=result["pure_capacities_g_g"],
        reduction_factors=result["reduction_factors"],
        selectivities=result["selectivities"],
        competitive_effect=result["competitive_effect"],
    )
