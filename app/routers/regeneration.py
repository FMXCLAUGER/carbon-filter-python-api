"""
Regeneration Router - API endpoints for regeneration analysis

Provides endpoints for:
- TSA energy calculation
- PSA energy calculation
- Steam regeneration
- Economics comparison (regeneration vs replacement)
- CO2 savings calculation
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    TSAEnergyRequest,
    PSAEnergyRequest,
    VSAEnergyRequest,
    SteamRegenRequest,
    RegenCompareRequest,
    RegenCompareResult,
    RegenMethodResult,
    RegenerationEnergyResult,
    EconomicsCompareRequest,
    EconomicsCompareResult,
    ReplacementCosts,
    RegenerationCosts,
    EconomicsComparison,
    CO2SavingsRequest,
    CO2SavingsResult,
    RegenerationMethod,
)
from app.services.regeneration import (
    calculate_tsa_energy,
    calculate_psa_energy,
    calculate_vsa_energy,
    calculate_steam_regeneration,
    compare_regeneration_methods,
    RegenerationType,
    compare_regeneration_vs_replacement,
    calculate_co2_savings,
)

router = APIRouter(prefix="/api/regeneration", tags=["regeneration"])


def _energy_to_result(energy, regen_type: str) -> RegenerationEnergyResult:
    """Convert RegenerationEnergy dataclass to Pydantic model."""
    mass_adsorbate = energy.details.get("adsorbate_mass_kg", 0)
    energy_per_kg = energy.total_energy_kWh / mass_adsorbate if mass_adsorbate > 0 else 0

    return RegenerationEnergyResult(
        regen_type=regen_type,
        Q_sensible_kWh=round(energy.sensible_heat_kWh, 2),
        Q_desorption_kWh=round(energy.desorption_heat_kWh, 2),
        Q_total_kWh=round(energy.total_energy_kWh, 2),
        power_kW=round(energy.power_required_kW, 2),
        energy_per_kg_adsorbate=round(energy_per_kg, 2),
        efficiency=energy.efficiency,
        regen_time_h=energy.regeneration_time_h,
        notes=[
            f"Temperature rise: {energy.details.get('delta_T_C', 'N/A')}°C",
            f"Adsorbate mass: {energy.details.get('adsorbate_mass_kg', 0):.2f} kg",
        ],
    )


@router.post("/tsa/energy", response_model=RegenerationEnergyResult)
async def calculate_tsa(request: TSAEnergyRequest) -> RegenerationEnergyResult:
    """
    Calculate energy requirements for TSA (Temperature Swing Adsorption) regeneration.

    TSA involves heating the carbon bed to desorb contaminants, then cooling back
    to adsorption temperature. Most common method for activated carbon regeneration.
    """
    try:
        energy = calculate_tsa_energy(
            bed_mass=request.bed_mass,
            q_loading=request.q_loading,
            delta_H_ads=request.delta_H_ads,
            T_ads=request.T_ads,
            T_regen=request.T_regen,
            molecular_weight=request.molecular_weight,
            regen_time_h=request.regen_time_h,
            heating_efficiency=request.heating_efficiency,
            C_p_carbon=request.Cp_carbon,
        )
        return _energy_to_result(energy, "TSA")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"TSA calculation error: {str(e)}")


@router.post("/psa/energy", response_model=RegenerationEnergyResult)
async def calculate_psa(request: PSAEnergyRequest) -> RegenerationEnergyResult:
    """
    Calculate energy requirements for PSA (Pressure Swing Adsorption) regeneration.

    PSA uses pressure reduction to desorb contaminants. Less energy-intensive than TSA
    but requires pressure-rated vessels.
    """
    try:
        energy = calculate_psa_energy(
            bed_mass=request.bed_mass,
            q_loading=request.q_loading,
            P_ads=request.P_ads / 1e5,  # Pa to bar
            P_regen=request.P_regen / 1e5,  # Pa to bar
            compressor_efficiency=request.vacuum_efficiency,
            cycle_time_min=request.cycle_time_min,
        )

        return RegenerationEnergyResult(
            regen_type="PSA",
            Q_sensible_kWh=0,
            Q_desorption_kWh=0,
            Q_total_kWh=round(energy.total_energy_kWh, 2),
            power_kW=round(energy.power_required_kW, 2),
            energy_per_kg_adsorbate=round(energy.specific_energy_kWh_kg * request.bed_mass / (request.bed_mass * request.q_loading) if request.q_loading > 0 else 0, 2),
            efficiency=energy.efficiency,
            regen_time_h=energy.regeneration_time_h,
            notes=[
                f"Pressure ratio: {energy.details.get('pressure_ratio', 'N/A'):.1f}",
                f"Cycles per hour: {energy.details.get('cycles_per_hour', 'N/A'):.1f}",
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"PSA calculation error: {str(e)}")


@router.post("/steam/energy", response_model=RegenerationEnergyResult)
async def calculate_steam(request: SteamRegenRequest) -> RegenerationEnergyResult:
    """
    Calculate energy requirements for steam regeneration.

    Steam regeneration is effective for VOCs and enables solvent recovery.
    """
    try:
        energy = calculate_steam_regeneration(
            bed_mass=request.bed_mass,
            q_loading=request.q_loading,
            delta_H_ads=40000,  # Default
            P_steam_bar=request.steam_pressure_bar,
            steam_ratio=request.steam_ratio,
            regen_time_h=request.regen_time_h,
        )

        return RegenerationEnergyResult(
            regen_type="STEAM",
            Q_sensible_kWh=round(energy.sensible_heat_kWh, 2),
            Q_desorption_kWh=round(energy.desorption_heat_kWh, 2),
            Q_total_kWh=round(energy.total_energy_kWh, 2),
            power_kW=round(energy.power_required_kW, 2),
            energy_per_kg_adsorbate=round(energy.specific_energy_kWh_kg * request.bed_mass / (request.bed_mass * request.q_loading) if request.q_loading > 0 else 0, 2),
            efficiency=energy.efficiency,
            regen_time_h=energy.regeneration_time_h,
            notes=[
                f"Steam required: {energy.details.get('steam_required_kg', 0):.1f} kg",
                f"Steam ratio: {request.steam_ratio} kg steam/kg adsorbate",
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Steam calculation error: {str(e)}")


@router.post("/vsa/energy", response_model=RegenerationEnergyResult)
async def calculate_vsa(request: VSAEnergyRequest) -> RegenerationEnergyResult:
    """
    Calculate energy requirements for VSA (Vacuum Swing Adsorption) regeneration.

    VSA uses vacuum to reduce pressure for desorption. Lower energy than PSA
    and operates at ambient temperature.
    """
    try:
        energy = calculate_vsa_energy(
            bed_mass=request.bed_mass,
            q_loading=request.q_loading,
            P_ads=request.P_ads / 1e5,  # Pa to bar
            P_vacuum=request.P_vacuum / 1e5,  # Pa to bar
            bed_void_fraction=request.bed_void_fraction,
            vacuum_pump_efficiency=request.vacuum_pump_efficiency,
            cycle_time_min=request.cycle_time_min,
        )

        return RegenerationEnergyResult(
            regen_type="VSA",
            Q_sensible_kWh=0,
            Q_desorption_kWh=0,
            Q_total_kWh=round(energy.total_energy_kWh, 2),
            power_kW=round(energy.power_required_kW, 2),
            energy_per_kg_adsorbate=round(energy.specific_energy_kWh_kg * request.bed_mass / (request.bed_mass * request.q_loading) if request.q_loading > 0 else 0, 2),
            efficiency=energy.efficiency,
            regen_time_h=energy.regeneration_time_h,
            notes=[
                f"Pressure ratio: {energy.details.get('pressure_ratio', 'N/A'):.1f}",
                f"Cycles per hour: {energy.details.get('cycles_per_hour', 'N/A'):.1f}",
                f"Volume evacuated: {energy.details.get('volume_evacuated_m3', 0):.3f} m³",
            ],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"VSA calculation error: {str(e)}")


@router.post("/compare-all", response_model=RegenCompareResult)
async def compare_all_methods(request: RegenCompareRequest) -> RegenCompareResult:
    """
    Compare all four regeneration methods (TSA, PSA, VSA, Steam) for a given application.

    Returns energy requirements, advantages/disadvantages, and a recommendation
    based on adsorbate loading and properties.
    """
    try:
        result = compare_regeneration_methods(
            bed_mass=request.bed_mass,
            q_loading=request.q_loading,
            delta_H_ads=request.delta_H_ads,
            molecular_weight=request.molecular_weight,
        )

        return RegenCompareResult(
            TSA=RegenMethodResult(
                energy_kWh=result["TSA"]["energy_kWh"],
                specific_energy_kWh_kg=result["TSA"]["specific_energy_kWh_kg"],
                power_kW=result["TSA"]["power_kW"],
                time_h=result["TSA"]["time_h"],
                advantages=result["TSA"]["advantages"],
                disadvantages=result["TSA"]["disadvantages"],
            ),
            PSA=RegenMethodResult(
                energy_kWh=result["PSA"]["energy_kWh"],
                specific_energy_kWh_kg=result["PSA"]["specific_energy_kWh_kg"],
                power_kW=result["PSA"]["power_kW"],
                time_h=result["PSA"]["time_h"],
                advantages=result["PSA"]["advantages"],
                disadvantages=result["PSA"]["disadvantages"],
            ),
            VSA=RegenMethodResult(
                energy_kWh=result["VSA"]["energy_kWh"],
                specific_energy_kWh_kg=result["VSA"]["specific_energy_kWh_kg"],
                power_kW=result["VSA"]["power_kW"],
                time_h=result["VSA"]["time_h"],
                advantages=result["VSA"]["advantages"],
                disadvantages=result["VSA"]["disadvantages"],
            ),
            STEAM=RegenMethodResult(
                energy_kWh=result["STEAM"]["energy_kWh"],
                specific_energy_kWh_kg=result["STEAM"]["specific_energy_kWh_kg"],
                power_kW=result["STEAM"]["power_kW"],
                time_h=result["STEAM"]["time_h"],
                advantages=result["STEAM"]["advantages"],
                disadvantages=result["STEAM"]["disadvantages"],
            ),
            recommendation=result["recommendation"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Comparison error: {str(e)}")


@router.post("/economics/compare", response_model=EconomicsCompareResult)
async def compare_economics(request: EconomicsCompareRequest) -> EconomicsCompareResult:
    """
    Compare economics of regeneration vs. carbon replacement.

    Returns complete economic analysis including annual costs, NPV,
    payback period, ROI, and CO2 savings.
    """
    try:
        # Map string to enum
        regen_type_map = {
            "TSA": RegenerationType.TSA,
            "PSA": RegenerationType.PSA,
            "VSA": RegenerationType.VSA,
            "STEAM": RegenerationType.STEAM,
            "EXTERNAL": RegenerationType.EXTERNAL,
        }
        regen_type = regen_type_map.get(request.regen_type, RegenerationType.TSA)

        result = compare_regeneration_vs_replacement(
            bed_mass=request.bed_mass,
            q_loading=request.q_loading,
            carbon_price_eur_kg=request.carbon_price_eur_kg,
            energy_price_eur_kwh=request.energy_price_eur_kwh,
            disposal_cost_eur_kg=request.disposal_cost_eur_kg,
            cycles_per_year=request.cycles_per_year,
            delta_H_ads=request.delta_H_ads,
            molecular_weight=request.molecular_weight,
            regen_type=regen_type,
            regen_equipment_cost=request.equipment_cost_eur,
            labor_cost_eur_h=request.labor_cost_eur_h,
            regen_labor_h=request.labor_hours_per_regen,
            carbon_lifetime_cycles=request.carbon_lifetime_cycles,
            discount_rate=request.discount_rate,
            analysis_years=request.analysis_years,
        )

        # Convert to Pydantic models
        replacement = ReplacementCosts(
            annual_cost_eur=result["replacement"]["annual_cost_eur"],
            npv_eur=result["replacement"]["npv_10y_eur"],
            co2_per_year_kg=result["replacement"]["co2_kg_per_year"],
        )

        regeneration = RegenerationCosts(
            annual_cost_eur=result["regeneration"]["annual_cost_eur"],
            npv_eur=result["regeneration"]["npv_10y_eur"],
            co2_per_year_kg=result["regeneration"]["co2_kg_per_year"],
            energy_per_cycle_kwh=result["regeneration"]["energy_cost_per_cycle_eur"] / request.energy_price_eur_kwh if request.energy_price_eur_kwh > 0 else 0,
        )

        comparison = EconomicsComparison(
            annual_savings_eur=result["comparison"]["annual_savings_eur"],
            total_savings_eur=result["comparison"]["total_savings_10y_eur"],
            payback_years=result["comparison"]["payback_years"] if isinstance(result["comparison"]["payback_years"], (int, float)) else None,
            roi_percent=result["comparison"]["roi_percent"] if isinstance(result["comparison"]["roi_percent"], (int, float)) else None,
            co2_savings_kg_year=result["comparison"]["co2_savings_kg_year"],
            recommendation="regeneration" if result["comparison"]["recommendation"] == "Regeneration" else "replacement",
        )

        return EconomicsCompareResult(
            replacement=replacement,
            regeneration=regeneration,
            comparison=comparison,
            analysis_years=request.analysis_years,
            discount_rate=request.discount_rate,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Economics comparison error: {str(e)}")


@router.post("/co2-savings", response_model=CO2SavingsResult)
async def calculate_co2(request: CO2SavingsRequest) -> CO2SavingsResult:
    """
    Calculate CO2 savings from regeneration vs. replacement.

    Includes equivalents (cars, trees) for context.
    """
    try:
        result = calculate_co2_savings(
            bed_mass=request.bed_mass,
            cycles_per_year=request.cycles_per_year,
            regen_energy_kwh=request.energy_per_regen_kwh,
            carbon_lifetime_cycles=request.carbon_lifetime_cycles,
            co2_per_kg_carbon=request.co2_per_kg_carbon,
            co2_per_kwh=request.co2_per_kwh,
        )

        return CO2SavingsResult(
            co2_replacement_kg_year=result["replacement"]["co2_kg_year"],
            co2_regeneration_kg_year=result["regeneration"]["co2_kg_year"],
            co2_savings_kg_year=result["savings"]["co2_kg_year"],
            equivalent_cars=result["savings"]["equivalent_cars"],
            equivalent_trees=result["savings"]["equivalent_trees"],
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CO2 savings calculation error: {str(e)}")


@router.get("/methods", response_model=list[RegenerationMethod])
async def list_methods() -> list[RegenerationMethod]:
    """
    List all available regeneration methods with descriptions.
    """
    return [
        RegenerationMethod(
            type="TSA",
            name="Temperature Swing Adsorption",
            description="Most common method. Heats the bed to 150-300°C to desorb contaminants, then cools back to adsorption temperature.",
            typical_temperature="150-300°C",
            typical_pressure="Atmospheric",
            energy_range="50-200 kWh/tonne carbon",
            best_for="VOCs, solvents, most organic compounds",
            advantages=[
                "High regeneration efficiency (90-95%)",
                "Well-established technology",
                "Works for strongly adsorbed species",
            ],
            disadvantages=[
                "High energy consumption",
                "Thermal stress on carbon",
                "Slower cycle times (2-8 hours)",
            ],
        ),
        RegenerationMethod(
            type="PSA",
            name="Pressure Swing Adsorption",
            description="Uses pressure reduction to desorb contaminants. Operates at ambient temperature.",
            typical_temperature="Ambient",
            typical_pressure="1-10 bar (ads) / 0.1-1 bar (regen)",
            energy_range="10-50 kWh/tonne carbon",
            best_for="Gases, weakly adsorbed compounds",
            advantages=[
                "Fast cycles (minutes)",
                "No thermal degradation",
                "Continuous operation possible",
            ],
            disadvantages=[
                "Requires pressure-rated vessels",
                "Lower working capacity",
                "Not suitable for strongly adsorbed species",
            ],
        ),
        RegenerationMethod(
            type="VSA",
            name="Vacuum Swing Adsorption",
            description="Uses vacuum to reduce pressure for desorption. Lower energy than PSA.",
            typical_temperature="Ambient",
            typical_pressure="1 bar (ads) / 0.1-0.3 bar (regen)",
            energy_range="5-30 kWh/tonne carbon",
            best_for="Weakly to moderately adsorbed compounds",
            advantages=[
                "Lower energy than PSA",
                "No compression needed",
                "Good for moderately adsorbed species",
            ],
            disadvantages=[
                "Vacuum pump maintenance",
                "Air ingress risk",
                "Limited pressure ratio",
            ],
        ),
        RegenerationMethod(
            type="STEAM",
            name="Steam Regeneration",
            description="Uses steam to provide both heat and displacement for desorption. Enables solvent recovery.",
            typical_temperature="100-150°C",
            typical_pressure="2-5 bar",
            energy_range="100-300 kWh/tonne carbon",
            best_for="VOC recovery, solvents, easily desorbed compounds",
            advantages=[
                "Fast regeneration",
                "Enables solvent recovery",
                "Effective for VOCs",
            ],
            disadvantages=[
                "Requires steam supply",
                "Condensate handling needed",
                "May wet the carbon",
            ],
        ),
        RegenerationMethod(
            type="EXTERNAL",
            name="External/Off-site Regeneration",
            description="Carbon sent to specialized facility for thermal reactivation at 700-900°C.",
            typical_temperature="700-900°C",
            typical_pressure="N/A",
            energy_range="N/A (outsourced)",
            best_for="Spent carbon, high-value recovery, specialized applications",
            advantages=[
                "High capacity recovery (90-95%)",
                "No on-site equipment needed",
                "Professional handling of spent carbon",
            ],
            disadvantages=[
                "Transport logistics",
                "Turnaround time (1-2 weeks)",
                "Requires spare carbon inventory",
            ],
        ),
    ]
