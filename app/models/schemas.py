from pydantic import BaseModel, Field
from typing import Optional, List
from enum import Enum


# =============================================================================
# Isotherm Fitting Schemas
# =============================================================================

class IsothermModelType(str, Enum):
    """Available isotherm models for fitting."""
    langmuir = "langmuir"
    freundlich = "freundlich"
    sips = "sips"
    toth = "toth"
    dubinin_radushkevich = "dubinin_radushkevich"
    dubinin_astakhov = "dubinin_astakhov"
    bet = "bet"


class IsothermDataPoint(BaseModel):
    """A single data point for isotherm fitting."""
    pressure: float = Field(..., description="Pressure or concentration")
    loading: float = Field(..., gt=0, description="Adsorbed amount")


class IsothermFitRequest(BaseModel):
    """Request for isotherm model fitting."""
    data_points: List[IsothermDataPoint] = Field(..., min_length=3)
    models_to_fit: List[IsothermModelType] = Field(
        default=[IsothermModelType.langmuir, IsothermModelType.freundlich, IsothermModelType.sips],
        description="Models to fit to the data"
    )
    temperature: float = Field(298.15, description="Temperature in K")


class IsothermFitResult(BaseModel):
    """Result of isotherm model fitting."""
    model: IsothermModelType
    parameters: dict = Field(..., description="Fitted parameters")
    r2: float = Field(..., ge=0, le=1, description="R-squared value")
    rmse: float = Field(..., ge=0, description="Root mean square error")


# =============================================================================
# Original Schemas
# =============================================================================

class IsothermParams(BaseModel):
    """Isotherm model parameters for pyGAPS calculations."""
    model: str = Field(..., description="Isotherm model name (Langmuir, Toth, Freundlich, DR, DA)")
    params: dict = Field(..., description="Model parameters (e.g., {'n_m': 4.8, 'K': 0.75, 't': 0.48})")
    temp_ref: float = Field(298.15, description="Reference temperature in K")


class PollutantInput(BaseModel):
    name: str
    cas_number: Optional[str] = None
    concentration: float = Field(..., gt=0, description="Concentration in mg/m³")
    molecular_weight: Optional[float] = Field(None, gt=0, description="Molecular weight in g/mol")
    target_outlet: Optional[float] = Field(None, ge=0, description="Target outlet concentration (VLE) in mg/m³")
    isotherm_params: Optional[IsothermParams] = Field(None, description="Custom isotherm parameters for pyGAPS")


class CarbonInput(BaseModel):
    name: str
    surface_area: float = Field(..., gt=0, description="BET surface area in m²/g")
    bulk_density: float = Field(..., gt=0, description="Bulk density in kg/m³")
    bed_voidage: float = Field(..., gt=0, lt=1, description="Bed voidage (0-1)")
    particle_diameter: Optional[float] = Field(None, gt=0, description="Particle diameter in mm")
    micropore_volume: Optional[float] = Field(None, gt=0, description="Micropore volume in cm³/g")


class CalculationRequest(BaseModel):
    flow_rate: float = Field(..., gt=0, description="Flow rate in m³/h")
    temperature: float = Field(..., description="Temperature in °C")
    humidity: float = Field(..., ge=0, le=100, description="Relative humidity in %")
    pressure: float = Field(..., gt=0, description="Pressure in Pa")
    pollutants: list[PollutantInput] = Field(..., min_length=1)
    carbon: CarbonInput
    target_velocity: Optional[float] = Field(None, gt=0, description="Target superficial velocity in m/s")
    target_ebct: Optional[float] = Field(None, gt=0, description="Target EBCT in seconds")
    safety_factor: float = Field(1.2, ge=1.0, le=3.0)
    design_life_months: int = Field(12, ge=1, le=60)
    breakthrough_model: str = Field("WHEELER_JONAS", pattern="^(WHEELER_JONAS|THOMAS|YOON_NELSON)$")


class FilterDimensions(BaseModel):
    bed_volume: float = Field(..., description="Bed volume in m³")
    bed_mass: float = Field(..., description="Carbon mass in kg")
    bed_diameter: float = Field(..., description="Bed diameter in m")
    bed_height: float = Field(..., description="Bed height in m")
    cross_section: float = Field(..., description="Cross-sectional area in m²")


class OperatingConditions(BaseModel):
    superficial_velocity: float = Field(..., description="Superficial velocity in m/s")
    ebct: float = Field(..., description="Empty Bed Contact Time in seconds")
    residence_time: float = Field(..., description="Actual residence time in seconds")
    pressure_drop: float = Field(..., description="Pressure drop in Pa")


class PollutantResult(BaseModel):
    name: str
    inlet_concentration: float
    outlet_concentration: float
    removal_efficiency: float = Field(..., ge=0, le=100, description="Removal efficiency in %")
    adsorption_capacity: float = Field(..., description="Adsorption capacity in g/g")
    breakthrough_time: float = Field(..., description="Breakthrough time in hours")
    mass_transfer_zone: float = Field(..., description="Mass transfer zone in m")


class BreakthroughPoint(BaseModel):
    time: float = Field(..., description="Time in hours")
    c_c0: float = Field(..., ge=0, le=1, description="C/C0 ratio")


class CalculationResponse(BaseModel):
    dimensions: FilterDimensions
    operating: OperatingConditions
    pollutant_results: list[PollutantResult]
    overall_efficiency: float = Field(..., ge=0, le=100)
    service_life_months: float
    breakthrough_curve: list[BreakthroughPoint]
    breakthrough_model: str
    max_temperature_rise: Optional[float] = None
    thermal_warning: bool = False
    warnings: list[str] = []
    recommendations: list[str] = []
    calculation_time_ms: float
    iast_results: Optional["IASTResponse"] = None


class IASTRequest(BaseModel):
    """Request for IAST multi-component calculation."""
    pollutants: list[PollutantInput] = Field(..., min_length=2, description="At least 2 pollutants required")
    temperature: float = Field(25.0, description="Temperature in °C")


class IASTResponse(BaseModel):
    """IAST calculation results."""
    pollutants: list[str]
    concentrations: list[float] = Field(..., description="Inlet concentrations in mg/m³")
    molecular_weights: list[float]
    iast_capacities_g_g: list[float] = Field(..., description="Competitive capacities in g/g")
    pure_capacities_g_g: list[float] = Field(..., description="Pure component capacities in g/g")
    reduction_factors: list[float] = Field(..., description="Capacity reduction due to competition")
    selectivities: dict[str, Optional[float]] = Field(..., description="Selectivity ratios between pairs")
    competitive_effect: bool = Field(True, description="Whether competitive adsorption was calculated")


# ============================================================================
# KINETICS SCHEMAS
# ============================================================================

class ThomasRequest(BaseModel):
    """Request for Thomas breakthrough model calculation."""
    flow_rate: float = Field(..., gt=0, description="Volumetric flow rate in m³/h")
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    q0: float = Field(..., gt=0, description="Maximum adsorption capacity in kg/kg")
    k_Th: Optional[float] = Field(None, gt=0, description="Thomas rate constant in m³/(kg·h)")
    v: float = Field(0.3, gt=0, description="Superficial velocity in m/s")
    d_p: float = Field(0.003, gt=0, description="Particle diameter in m")
    pollutant_type: str = Field("VOC", description="Pollutant type for k_Th estimation")
    n_points: int = Field(100, ge=10, le=500, description="Number of points in breakthrough curve")


class YoonNelsonRequest(BaseModel):
    """Request for Yoon-Nelson breakthrough model calculation."""
    flow_rate: float = Field(..., gt=0, description="Volumetric flow rate in m³/h")
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    q0: float = Field(..., gt=0, description="Maximum adsorption capacity in kg/kg")
    k_YN: Optional[float] = Field(None, gt=0, description="Yoon-Nelson rate constant in 1/h")
    tau: Optional[float] = Field(None, gt=0, description="Time for 50% breakthrough in hours")
    pollutant_type: str = Field("VOC", description="Pollutant type for parameter estimation")
    n_points: int = Field(100, ge=10, le=500, description="Number of points in breakthrough curve")


class BohartAdamsRequest(BaseModel):
    """Request for Bohart-Adams breakthrough model calculation."""
    flow_rate: float = Field(..., gt=0, description="Volumetric flow rate in m³/h")
    bed_height: float = Field(..., gt=0, description="Bed height in m")
    bed_area: float = Field(..., gt=0, description="Cross-sectional area in m²")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    N0: Optional[float] = Field(None, gt=0, description="Saturation concentration in kg/m³")
    k_BA: Optional[float] = Field(None, gt=0, description="Bohart-Adams rate constant in m³/(kg·h)")
    rho_bulk: float = Field(480, gt=0, description="Bulk density in kg/m³")
    pollutant_type: str = Field("VOC", description="Pollutant type for parameter estimation")
    n_points: int = Field(100, ge=10, le=500, description="Number of points in breakthrough curve")


class KineticsCompareRequest(BaseModel):
    """Request for comparing all kinetic models."""
    flow_rate: float = Field(..., gt=0, description="Volumetric flow rate in m³/h")
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    bed_height: float = Field(..., gt=0, description="Bed height in m")
    bed_area: float = Field(..., gt=0, description="Cross-sectional area in m²")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    q0: float = Field(..., gt=0, description="Maximum adsorption capacity in kg/kg")
    W_e: Optional[float] = Field(None, gt=0, description="Equilibrium capacity for Wheeler-Jonas in g/g (defaults to q0)")
    rho_bulk: float = Field(480, gt=0, description="Bulk density in kg/m³")
    velocity: float = Field(0.3, gt=0, description="Superficial velocity in m/s")
    particle_diameter: Optional[float] = Field(None, description="Particle diameter in mm")
    pollutant_type: str = Field("VOC", description="Pollutant type")
    n_points: int = Field(100, ge=10, le=500, description="Number of points")


class BreakthroughTimes(BaseModel):
    """Breakthrough times at different C/C0 ratios."""
    t_5: float = Field(..., description="Time for 5% breakthrough in hours")
    t_10: float = Field(..., description="Time for 10% breakthrough in hours")
    t_50: float = Field(..., description="Time for 50% breakthrough in hours")
    t_90: float = Field(..., description="Time for 90% breakthrough in hours")
    MTZ: float = Field(..., description="Mass Transfer Zone length in m")


class KineticsResult(BaseModel):
    """Result from a kinetic model calculation."""
    model: str = Field(..., description="Model name (Thomas, Yoon-Nelson, Bohart-Adams)")
    parameters: dict = Field(..., description="Model parameters used")
    breakthrough_times: BreakthroughTimes
    curve: list[dict] = Field(..., description="Breakthrough curve [{time_h, C_C0}]")
    service_time_h: float = Field(..., description="Service time at 10% breakthrough in hours")


class WheelerJonasKineticsRequest(BaseModel):
    """Request for Wheeler-Jonas breakthrough model calculation."""
    flow_rate: float = Field(..., gt=0, description="Volumetric flow rate in m³/h")
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    bed_height: float = Field(..., gt=0, description="Bed height in m")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    C_out: float = Field(None, description="Breakthrough concentration in mg/m³ (default 5% of C0)")
    W_e: float = Field(..., gt=0, description="Equilibrium capacity in g/g")
    k_v: Optional[float] = Field(None, description="Mass transfer coefficient in 1/min")
    rho_bulk: float = Field(480, gt=0, description="Bulk density in kg/m³")
    velocity: float = Field(0.3, gt=0, description="Superficial velocity in m/s")
    particle_diameter: Optional[float] = Field(None, description="Particle diameter in mm")
    n_points: int = Field(100, ge=10, le=500, description="Number of points in breakthrough curve")


class KineticsCompareResult(BaseModel):
    """Comparison of all kinetic models."""
    wheeler_jonas: Optional[KineticsResult] = Field(None, description="Wheeler-Jonas model result")
    thomas: KineticsResult
    yoon_nelson: KineticsResult
    bohart_adams: KineticsResult
    recommended_model: str = Field(..., description="Recommended model for this case")
    recommendation_reason: str


class KineticModel(BaseModel):
    """Description of a kinetic model."""
    name: str
    description: str
    equation: str
    parameters: list[str]
    best_for: str


# ============================================================================
# REGENERATION SCHEMAS
# ============================================================================

class TSAEnergyRequest(BaseModel):
    """Request for TSA regeneration energy calculation."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    q_loading: float = Field(..., gt=0, description="Adsorbate loading in kg/kg")
    delta_H_ads: float = Field(40000, description="Heat of adsorption in J/mol")
    T_ads: float = Field(25, description="Adsorption temperature in °C")
    T_regen: float = Field(200, ge=100, le=400, description="Regeneration temperature in °C")
    molecular_weight: float = Field(92, gt=0, description="Molecular weight in g/mol")
    regen_time_h: float = Field(4.0, gt=0, description="Regeneration time in hours")
    heating_efficiency: float = Field(0.85, gt=0, le=1, description="Heating system efficiency")
    Cp_carbon: float = Field(800, gt=0, description="Carbon specific heat in J/(kg·K)")


class PSAEnergyRequest(BaseModel):
    """Request for PSA regeneration energy calculation."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    q_loading: float = Field(..., gt=0, description="Adsorbate loading in kg/kg")
    P_ads: float = Field(101325, gt=0, description="Adsorption pressure in Pa")
    P_regen: float = Field(10000, gt=0, description="Regeneration pressure in Pa")
    vacuum_efficiency: float = Field(0.70, gt=0, le=1, description="Vacuum pump efficiency")
    cycle_time_min: float = Field(10, gt=0, description="Cycle time in minutes")


class SteamRegenRequest(BaseModel):
    """Request for steam regeneration calculation."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    q_loading: float = Field(..., gt=0, description="Adsorbate loading in kg/kg")
    steam_pressure_bar: float = Field(3.0, gt=0, description="Steam pressure in bar")
    steam_ratio: float = Field(3.0, gt=0, description="kg steam / kg adsorbate")
    regen_time_h: float = Field(2.0, gt=0, description="Regeneration time in hours")
    condensate_recovery: float = Field(0.8, gt=0, le=1, description="Condensate recovery efficiency")


class VSAEnergyRequest(BaseModel):
    """Request for VSA (Vacuum Swing Adsorption) regeneration energy calculation."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    q_loading: float = Field(..., gt=0, description="Adsorbate loading in kg/kg")
    P_ads: float = Field(101325, gt=0, description="Adsorption pressure in Pa")
    P_vacuum: float = Field(20000, gt=0, lt=101325, description="Vacuum pressure in Pa")
    bed_void_fraction: float = Field(0.4, gt=0, lt=1, description="Bed void fraction")
    vacuum_pump_efficiency: float = Field(0.65, gt=0, le=1, description="Vacuum pump efficiency")
    cycle_time_min: float = Field(15, gt=0, description="Cycle time in minutes")


class RegenCompareRequest(BaseModel):
    """Request for comparing all regeneration methods."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    q_loading: float = Field(..., gt=0, description="Adsorbate loading in kg/kg")
    delta_H_ads: float = Field(40000, description="Heat of adsorption in J/mol")
    molecular_weight: float = Field(92, gt=0, description="Molecular weight in g/mol")


class RegenMethodResult(BaseModel):
    """Result for a single regeneration method in comparison."""
    energy_kWh: float
    specific_energy_kWh_kg: float
    power_kW: float
    time_h: float
    advantages: list[str]
    disadvantages: list[str]


class RegenCompareResult(BaseModel):
    """Comparison of all regeneration methods."""
    TSA: RegenMethodResult
    PSA: RegenMethodResult
    VSA: RegenMethodResult
    STEAM: RegenMethodResult
    recommendation: dict


class RegenerationEnergyResult(BaseModel):
    """Energy calculation result for regeneration."""
    regen_type: str = Field(..., description="Regeneration type (TSA, PSA, VSA, STEAM)")
    Q_sensible_kWh: float = Field(..., description="Sensible heat in kWh")
    Q_desorption_kWh: float = Field(..., description="Desorption heat in kWh")
    Q_total_kWh: float = Field(..., description="Total energy per cycle in kWh")
    power_kW: float = Field(..., description="Average power requirement in kW")
    energy_per_kg_adsorbate: float = Field(..., description="Energy per kg adsorbate in kWh/kg")
    efficiency: float = Field(..., description="System efficiency")
    regen_time_h: float = Field(..., description="Regeneration time in hours")
    notes: list[str] = Field(default_factory=list)


class EconomicsCompareRequest(BaseModel):
    """Request for regeneration vs replacement economics comparison."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    q_loading: float = Field(..., gt=0, description="Adsorbate loading in kg/kg")
    carbon_price_eur_kg: float = Field(3.5, gt=0, description="Carbon price in €/kg")
    energy_price_eur_kwh: float = Field(0.15, gt=0, description="Energy price in €/kWh")
    disposal_cost_eur_kg: float = Field(0.5, ge=0, description="Disposal cost in €/kg")
    cycles_per_year: int = Field(12, gt=0, description="Number of cycles per year")
    regen_type: str = Field("TSA", pattern="^(TSA|PSA|VSA|STEAM|EXTERNAL)$")
    equipment_cost_eur: float = Field(25000, ge=0, description="Regeneration equipment cost in €")
    labor_cost_eur_h: float = Field(50, gt=0, description="Labor cost in €/h")
    labor_hours_per_regen: float = Field(4, gt=0, description="Labor hours per regeneration")
    carbon_lifetime_cycles: int = Field(100, gt=0, description="Carbon lifetime in cycles")
    discount_rate: float = Field(0.08, ge=0, le=0.2, description="Discount rate for NPV")
    analysis_years: int = Field(10, ge=1, le=30, description="Analysis period in years")
    delta_H_ads: float = Field(40000, description="Heat of adsorption in J/mol")
    molecular_weight: float = Field(92, gt=0, description="Molecular weight in g/mol")


class CO2SavingsRequest(BaseModel):
    """Request for CO2 savings calculation."""
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    cycles_per_year: int = Field(12, gt=0, description="Cycles per year")
    energy_per_regen_kwh: float = Field(..., gt=0, description="Energy per regeneration in kWh")
    carbon_lifetime_cycles: int = Field(100, gt=0, description="Carbon lifetime in cycles")
    co2_per_kg_carbon: float = Field(4.0, gt=0, description="CO2 for carbon production in kg CO2/kg")
    co2_per_kwh: float = Field(0.4, gt=0, description="CO2 for electricity in kg CO2/kWh")


class ReplacementCosts(BaseModel):
    """Costs for carbon replacement strategy."""
    annual_cost_eur: float
    npv_eur: float
    co2_per_year_kg: float


class RegenerationCosts(BaseModel):
    """Costs for regeneration strategy."""
    annual_cost_eur: float
    npv_eur: float
    co2_per_year_kg: float
    energy_per_cycle_kwh: float


class EconomicsComparison(BaseModel):
    """Comparison metrics."""
    annual_savings_eur: float
    total_savings_eur: float
    payback_years: Optional[float] = None
    roi_percent: Optional[float] = None
    co2_savings_kg_year: float
    recommendation: str = Field(..., pattern="^(regeneration|replacement)$")


class EconomicsCompareResult(BaseModel):
    """Full economics comparison result."""
    replacement: ReplacementCosts
    regeneration: RegenerationCosts
    comparison: EconomicsComparison
    analysis_years: int
    discount_rate: float


class CO2SavingsResult(BaseModel):
    """CO2 savings calculation result."""
    co2_replacement_kg_year: float
    co2_regeneration_kg_year: float
    co2_savings_kg_year: float
    equivalent_cars: float = Field(..., description="Equivalent to X cars per year")
    equivalent_trees: float = Field(..., description="Equivalent to X trees absorbing CO2")


class RegenerationMethod(BaseModel):
    """Description of a regeneration method."""
    type: str
    name: str
    description: str
    typical_temperature: Optional[str] = None
    typical_pressure: Optional[str] = None
    energy_range: str
    best_for: str
    advantages: list[str]
    disadvantages: list[str]


# ============================================================================
# THERMAL SCHEMAS
# ============================================================================

class TemperatureProfileRequest(BaseModel):
    """Request for bed temperature profile calculation."""
    inlet_temp_C: float = Field(..., description="Inlet gas temperature in °C")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    Q_gas: float = Field(..., gt=0, description="Gas flow rate in m³/h")
    bed_height: float = Field(..., gt=0, description="Bed height in m")
    bed_area: float = Field(..., gt=0, description="Cross-sectional area in m²")
    delta_H_ads: float = Field(40000, description="Heat of adsorption in J/mol")
    rho_bulk: float = Field(480, gt=0, description="Bulk density in kg/m³")
    molecular_weight: float = Field(92, gt=0, description="Molecular weight in g/mol")
    carbon_type: str = Field("COAL", description="Carbon type for auto-ignition temp")
    n_segments: int = Field(10, ge=5, le=50, description="Number of bed segments")


class TemperatureRiseRequest(BaseModel):
    """Request for quick temperature rise calculation."""
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    Q_gas: float = Field(..., gt=0, description="Gas flow rate in m³/h")
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    delta_H_ads: float = Field(40000, description="Heat of adsorption in J/mol")
    molecular_weight: float = Field(92, gt=0, description="Molecular weight in g/mol")
    q0: float = Field(0.15, gt=0, description="Adsorption capacity in kg/kg")


class SafetyCheckRequest(BaseModel):
    """Request for thermal safety check."""
    inlet_temp_C: float = Field(..., description="Inlet gas temperature in °C")
    C0: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    Q_gas: float = Field(..., gt=0, description="Gas flow rate in m³/h")
    bed_mass: float = Field(..., gt=0, description="Carbon bed mass in kg")
    delta_H_ads: float = Field(40000, description="Heat of adsorption in J/mol")
    molecular_weight: float = Field(92, gt=0, description="Molecular weight in g/mol")
    carbon_type: str = Field("COAL", description="Carbon type")


class TemperatureProfileResult(BaseModel):
    """Temperature profile calculation result."""
    inlet_temp_C: float
    max_temp_C: float
    temp_rise_C: float
    temp_profile: list[dict] = Field(..., description="[{position_m, temperature_C}]")
    safety_status: str = Field(..., pattern="^(SAFE|WARNING|CRITICAL)$")
    warnings: list[str] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)
    auto_ignition_risk: bool
    desorption_risk: bool


class TemperatureRiseResult(BaseModel):
    """Temperature rise estimation result."""
    delta_T_adiabatic_C: float
    delta_T_actual_C: float
    heat_generated_kJ: float
    heat_generation_rate_W: float
    moles_adsorbed: float
    safety_status: str


class SafetyAlert(BaseModel):
    """A single safety alert."""
    level: str = Field(..., description="CRITICAL, DANGER, WARNING, CAUTION")
    message: str
    action: str


class SafetyCheckResult(BaseModel):
    """Safety check result."""
    status: str = Field(..., description="SAFE, CAUTION, WARNING, DANGER, CRITICAL")
    operating_temp_C: float
    temp_rise_C: float
    auto_ignition_temp_C: float
    safety_margin_C: float
    alerts: list[SafetyAlert] = Field(default_factory=list)
    recommendations: list[str] = Field(default_factory=list)


class AutoIgnitionTemp(BaseModel):
    """Auto-ignition temperature for a carbon type."""
    carbon_type: str
    temp_C: float
    safe_operating_max_C: float
    notes: str
