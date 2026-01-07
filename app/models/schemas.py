from pydantic import BaseModel, Field
from typing import Optional


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
