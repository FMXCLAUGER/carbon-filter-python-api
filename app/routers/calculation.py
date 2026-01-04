"""Calculation API endpoints"""
from fastapi import APIRouter, HTTPException
from app.models.schemas import CalculationRequest, CalculationResult
from app.services.calculation_service import perform_calculation

router = APIRouter()


@router.post("/calculate", response_model=CalculationResult)
async def calculate_filter(request: CalculationRequest) -> CalculationResult:
    """
    Calculate activated carbon filter dimensions and performance.

    This endpoint performs a complete dimensioning calculation including:
    - Filter dimensions (volume, mass, diameter, height)
    - Operating conditions (velocity, EBCT, pressure drop)
    - Performance for each pollutant (capacity, breakthrough time)
    - Breakthrough curve generation
    - Thermal analysis
    - Warnings and recommendations

    Args:
        request: CalculationRequest with effluent conditions, pollutants, and carbon

    Returns:
        CalculationResult with complete analysis
    """
    try:
        result = perform_calculation(request)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Calculation error: {str(e)}")


@router.post("/calculate/quick")
async def quick_estimate(
    flow_rate: float,
    concentration: float,
    pollutant_name: str = "COV",
    molecular_weight: float = 100,
    target_life_months: int = 12,
):
    """
    Quick estimate for simple single-pollutant cases.

    Args:
        flow_rate: Flow rate in m³/h
        concentration: Pollutant concentration in mg/m³
        pollutant_name: Name of pollutant
        molecular_weight: Molecular weight in g/mol
        target_life_months: Target service life in months

    Returns:
        Simplified estimate result
    """
    from app.models.schemas import PollutantInput, CarbonProperties

    # Use default carbon properties
    default_carbon = CarbonProperties(
        name="Charbon actif générique",
        surface_area=1000,
        bulk_density=450,
        bed_voidage=0.4,
        particle_diameter=3.0,
    )

    request = CalculationRequest(
        flow_rate=flow_rate,
        temperature=25,
        humidity=50,
        pressure=101325,
        pollutants=[
            PollutantInput(
                name=pollutant_name,
                concentration=concentration,
                molecular_weight=molecular_weight,
            )
        ],
        carbon=default_carbon,
        design_life_months=target_life_months,
    )

    try:
        result = perform_calculation(request)

        # Return simplified response
        return {
            "bed_volume_m3": round(result.dimensions.bed_volume, 3),
            "bed_mass_kg": round(result.dimensions.bed_mass, 1),
            "bed_diameter_m": round(result.dimensions.bed_diameter, 2),
            "bed_height_m": round(result.dimensions.bed_height, 2),
            "velocity_m_s": round(result.operating.superficial_velocity, 3),
            "ebct_s": round(result.operating.ebct, 1),
            "pressure_drop_pa": round(result.operating.pressure_drop, 0),
            "service_life_months": round(result.service_life_months, 1),
            "warnings": result.warnings,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
