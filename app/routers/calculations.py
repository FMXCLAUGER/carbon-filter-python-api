import time
from fastapi import APIRouter, HTTPException

from app.models.schemas import (
    CalculationRequest,
    CalculationResponse,
    PollutantResult,
)
from app.services.filter_sizing import calculate_dimensions
from app.services.wheeler_jonas import (
    calculate_pollutant_result,
    generate_breakthrough_curve,
)

router = APIRouter(tags=["calculations"])


def analyze_results(
    operating: dict,
    pollutant_results: list[PollutantResult],
    design_life_months: int,
    safety_factor: float,
) -> tuple[list[str], list[str]]:
    """Generate warnings and recommendations based on results."""
    warnings = []
    recommendations = []
    
    # Check velocity
    if operating.superficial_velocity < 0.15:
        warnings.append("Vitesse superficielle faible - risque de channeling")
        recommendations.append("Augmenter le débit ou réduire la section")
    elif operating.superficial_velocity > 0.45:
        warnings.append("Vitesse superficielle élevée - efficacité réduite")
        recommendations.append("Réduire le débit ou augmenter la section")
    
    # Check EBCT
    if operating.ebct < 3.0:
        warnings.append("EBCT trop court - temps de contact insuffisant")
        recommendations.append("Augmenter la hauteur de lit ou réduire le débit")
    
    # Check breakthrough times
    min_bt = min(p.breakthrough_time for p in pollutant_results)
    design_life_hours = design_life_months * 30 * 24  # Approximate
    
    if min_bt < design_life_hours / safety_factor:
        limiting_pollutant = min(pollutant_results, key=lambda p: p.breakthrough_time)
        warnings.append(
            f"Durée de vie insuffisante: {limiting_pollutant.name} perce en {min_bt:.0f}h"
        )
        recommendations.append("Augmenter la masse de charbon ou changer de type")
    
    # Check removal efficiency
    for p in pollutant_results:
        if p.removal_efficiency < 90:
            warnings.append(f"Efficacité faible pour {p.name}: {p.removal_efficiency:.1f}%")
    
    # Check pressure drop
    if operating.pressure_drop > 2000:
        warnings.append("Perte de charge élevée (>2000 Pa)")
        recommendations.append("Utiliser des granulés plus gros ou réduire la hauteur")
    
    return warnings, recommendations


@router.post("/api/calculate", response_model=CalculationResponse)
async def calculate(request: CalculationRequest):
    """
    Main calculation endpoint for filter sizing.
    
    Calculates filter dimensions, operating conditions,
    and breakthrough times for all pollutants.
    """
    start_time = time.time()
    
    try:
        # 1. Calculate filter dimensions
        dimensions, operating = calculate_dimensions(
            flow_rate=request.flow_rate,
            bulk_density=request.carbon.bulk_density,
            bed_voidage=request.carbon.bed_voidage,
            target_velocity=request.target_velocity,
            target_ebct=request.target_ebct,
        )
        
        # 2. Calculate results for each pollutant
        pollutant_results = []
        for pollutant in request.pollutants:
            result = calculate_pollutant_result(
                pollutant_name=pollutant.name,
                concentration=pollutant.concentration,
                target_outlet=pollutant.target_outlet,
                temperature=request.temperature,
                humidity=request.humidity,
                surface_area=request.carbon.surface_area,
                carbon_mass=dimensions.bed_mass,
                bulk_density=request.carbon.bulk_density,
                flow_rate=request.flow_rate,
                velocity=operating.superficial_velocity,
                bed_height=dimensions.bed_height,
                particle_diameter=request.carbon.particle_diameter,
                molecular_weight=pollutant.molecular_weight,
            )
            pollutant_results.append(result)
        
        # 3. Overall efficiency (weighted by concentration)
        total_inlet = sum(p.concentration for p in request.pollutants)
        if total_inlet > 0:
            overall_efficiency = sum(
                (p.inlet_concentration - p.outlet_concentration) / total_inlet * 100
                for p in pollutant_results
            )
        else:
            overall_efficiency = 0
        
        # 4. Service life based on limiting pollutant
        min_breakthrough = min(p.breakthrough_time for p in pollutant_results)
        service_life_months = (min_breakthrough / (30 * 24)) / request.safety_factor
        
        # 5. Generate breakthrough curve for limiting pollutant
        breakthrough_curve = generate_breakthrough_curve(
            breakthrough_time=min_breakthrough,
            total_time=min_breakthrough * 2,
            num_points=50,
        )
        
        # 6. Thermal check (exothermic adsorption)
        total_loading = sum(p.adsorption_capacity * dimensions.bed_mass for p in pollutant_results)
        delta_h_ads = 40000  # J/mol typical
        cp_carbon = 1000  # J/(kg·K)
        max_temp_rise = (total_loading * delta_h_ads) / (dimensions.bed_mass * cp_carbon)
        thermal_warning = max_temp_rise > 50  # Warning if >50°C rise possible
        
        # 7. Generate warnings and recommendations
        warnings, recommendations = analyze_results(
            operating, pollutant_results, request.design_life_months, request.safety_factor
        )
        
        calculation_time_ms = (time.time() - start_time) * 1000
        
        return CalculationResponse(
            dimensions=dimensions,
            operating=operating,
            pollutant_results=pollutant_results,
            overall_efficiency=round(overall_efficiency, 1),
            service_life_months=round(service_life_months, 1),
            breakthrough_curve=breakthrough_curve,
            breakthrough_model=request.breakthrough_model,
            max_temperature_rise=round(max_temp_rise, 1) if max_temp_rise > 0 else None,
            thermal_warning=thermal_warning,
            warnings=warnings,
            recommendations=recommendations,
            calculation_time_ms=round(calculation_time_ms, 2),
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/api/calculate/quick")
async def quick_estimate(
    flow_rate: float,
    concentration: float,
    pollutant_name: str = "VOC",
    molecular_weight: float = 100,
    target_life_months: int = 12,
):
    """
    Quick estimation endpoint for simple cases.
    Returns approximate filter dimensions.
    """
    # Default carbon properties (typical GAC)
    default_carbon = {
        "name": "Standard GAC",
        "surface_area": 1000,
        "bulk_density": 450,
        "bed_voidage": 0.4,
        "particle_diameter": 3.0,
    }
    
    request = CalculationRequest(
        flow_rate=flow_rate,
        temperature=25.0,
        humidity=50.0,
        pressure=101325,
        pollutants=[{
            "name": pollutant_name,
            "concentration": concentration,
            "molecular_weight": molecular_weight,
        }],
        carbon=default_carbon,
        safety_factor=1.2,
        design_life_months=target_life_months,
    )
    
    return await calculate(request)
