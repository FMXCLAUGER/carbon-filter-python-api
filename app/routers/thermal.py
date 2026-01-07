"""
Thermal Router - API endpoints for thermal analysis of adsorption beds

Provides endpoints for:
- Temperature profile calculation
- Temperature rise estimation
- Safety assessment
- Cooling requirements
- Auto-ignition temperature reference
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    TemperatureProfileRequest,
    TemperatureProfileResult,
    TemperatureRiseRequest,
    TemperatureRiseResult,
    SafetyCheckRequest,
    SafetyCheckResult,
    SafetyAlert,
    AutoIgnitionTemp,
)
from app.services.thermal import (
    calculate_bed_temperature_profile,
    calculate_temperature_rise,
    check_thermal_safety,
)

router = APIRouter(prefix="/api/thermal", tags=["thermal"])


# Auto-ignition temperatures for reference
AUTO_IGNITION_TEMPS = {
    "COAL": 350,
    "COCONUT": 380,
    "WOOD": 320,
    "PEAT": 300,
    "LIGNITE": 280,
}


@router.post("/temperature-profile", response_model=TemperatureProfileResult)
async def calculate_profile(request: TemperatureProfileRequest) -> TemperatureProfileResult:
    """
    Calculate temperature profile along the adsorption bed.

    Returns the temperature at different positions in the bed,
    along with safety assessment and recommendations.
    """
    try:
        result = calculate_bed_temperature_profile(
            inlet_temp=request.inlet_temp,
            C0=request.C0,
            Q_gas=request.Q_gas,
            bed_height=request.bed_height,
            bed_area=request.bed_area,
            delta_H_ads=request.delta_H_ads,
            rho_bulk=request.rho_bulk,
            porosity=request.porosity,
            C_p_carbon=request.C_p_carbon,
            molecular_weight=request.molecular_weight,
            n_segments=request.n_segments,
            carbon_type=request.carbon_type,
        )

        return TemperatureProfileResult(
            inlet_temp_C=result.inlet_temp_C,
            max_temp_C=result.max_temp_C,
            temp_rise_C=result.temp_rise_C,
            safety_status=result.safety_status,
            auto_ignition_risk=result.auto_ignition_risk,
            desorption_risk=result.desorption_risk,
            warnings=result.warnings,
            recommendations=result.recommendations,
            profile=[
                {
                    "position_m": p["position_m"],
                    "position_normalized": p["position_normalized"],
                    "temperature_C": p["temperature_C"],
                    "temp_rise_C": p["temp_rise_C"],
                    "concentration_mg_m3": p["concentration_mg_m3"],
                    "heat_generation_W": p["heat_generation_W"],
                }
                for p in result.temp_profile
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temperature profile error: {str(e)}")


@router.post("/temperature-rise", response_model=TemperatureRiseResult)
async def calculate_rise(request: TemperatureRiseRequest) -> TemperatureRiseResult:
    """
    Calculate temperature rise during adsorption operation.

    Estimates both adiabatic and actual temperature rise accounting for
    heat removal by gas flow.
    """
    try:
        result = calculate_temperature_rise(
            C0=request.C0,
            Q_gas=request.Q_gas,
            delta_H_ads=request.delta_H_ads,
            rho_bulk=request.rho_bulk,
            bed_mass=request.bed_mass,
            C_p_carbon=request.C_p_carbon,
            molecular_weight=request.molecular_weight,
            operation_time_h=request.operation_time_h,
        )

        # Safety status based on temperature rise
        delta_T = result["delta_T_actual_C"]
        if delta_T > 50:
            safety_status = "DANGER"
        elif delta_T > 40:
            safety_status = "WARNING"
        elif delta_T > 20:
            safety_status = "CAUTION"
        else:
            safety_status = "NORMAL"

        return TemperatureRiseResult(
            delta_T_adiabatic_C=round(result["delta_T_adiabatic_C"], 2),
            delta_T_actual_C=round(result["delta_T_actual_C"], 2),
            heat_generated_kJ=round(result["heat_generated_J"] / 1000, 2),
            heat_generation_rate_W=round(result["heat_generation_rate_W"], 2),
            moles_adsorbed=round(result["moles_adsorbed"], 4),
            safety_status=safety_status,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Temperature rise error: {str(e)}")


@router.post("/safety-check", response_model=SafetyCheckResult)
async def safety_check(request: SafetyCheckRequest) -> SafetyCheckResult:
    """
    Perform thermal safety assessment for operating conditions.

    Checks operating temperature against safety thresholds and
    auto-ignition temperature for the carbon type.
    """
    try:
        result = check_thermal_safety(
            operating_temp=request.operating_temp,
            temp_rise=request.temp_rise,
            carbon_type=request.carbon_type,
            include_recommendations=True,
        )

        return SafetyCheckResult(
            status=result["status"],
            operating_temp_C=result["operating_temp_C"],
            temp_rise_C=result["temp_rise_C"],
            auto_ignition_temp_C=result["auto_ignition_temp_C"],
            safety_margin_C=result["safety_margin_C"],
            alerts=[
                SafetyAlert(
                    level=a["level"],
                    message=a["message"],
                    action=a["action"],
                )
                for a in result["alerts"]
            ],
            recommendations=result["recommendations"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Safety check error: {str(e)}")


@router.get("/auto-ignition-temps", response_model=list[AutoIgnitionTemp])
async def list_auto_ignition_temps() -> list[AutoIgnitionTemp]:
    """
    List auto-ignition temperatures for different carbon types.

    These are approximate values - always verify with manufacturer data.
    """
    return [
        AutoIgnitionTemp(
            carbon_type="COAL",
            temp_C=350,
            safe_operating_max_C=200,
            notes="Coal-based activated carbon, typical industrial grade",
        ),
        AutoIgnitionTemp(
            carbon_type="COCONUT",
            temp_C=380,
            safe_operating_max_C=230,
            notes="Coconut shell carbon, higher ignition temperature",
        ),
        AutoIgnitionTemp(
            carbon_type="WOOD",
            temp_C=320,
            safe_operating_max_C=170,
            notes="Wood-based carbon, lower ignition point",
        ),
        AutoIgnitionTemp(
            carbon_type="PEAT",
            temp_C=300,
            safe_operating_max_C=150,
            notes="Peat-based carbon, requires careful temperature monitoring",
        ),
        AutoIgnitionTemp(
            carbon_type="LIGNITE",
            temp_C=280,
            safe_operating_max_C=130,
            notes="Lignite-based carbon, lowest ignition temperature",
        ),
    ]


@router.get("/safety-thresholds")
async def get_safety_thresholds() -> dict:
    """
    Get safety threshold values used for thermal assessment.
    """
    return {
        "temperature_rise_thresholds_C": {
            "normal": {"max": 20, "description": "Normal operation"},
            "moderate": {"max": 40, "description": "Increased monitoring recommended"},
            "caution": {"max": 50, "description": "Review operating conditions"},
            "warning": {"max": 60, "description": "Risk of desorption"},
            "danger": {"above": 60, "description": "Immediate action required"},
        },
        "absolute_temperature_thresholds_C": {
            "normal": {"max": 60, "description": "Safe operation"},
            "warning": {"max": 80, "description": "Potential desorption"},
            "danger": {"max": 150, "description": "Approaching auto-ignition zone"},
            "critical": {"above": 150, "description": "Near auto-ignition, stop operation"},
        },
        "safety_margin_C": 150,
        "notes": [
            "Always maintain 150°C safety margin below auto-ignition",
            "Temperature sensors should be installed at multiple bed heights",
            "High-temperature shutdown interlocks recommended for > 100°C",
        ],
    }
