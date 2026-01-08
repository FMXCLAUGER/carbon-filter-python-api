"""CADET-Process API endpoints for high-precision breakthrough simulation."""
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from app.services.cadet.cadet_service import (
    CADETBreakthroughService,
    CADETConfig,
    CADETResult,
    get_available_models,
    get_cadet_status,
    check_cadet_available,
)

router = APIRouter(prefix="/cadet", tags=["cadet"])


class CADETSimulationRequest(BaseModel):
    """Request for CADET breakthrough simulation."""

    bed_length: float = Field(..., gt=0, description="Bed length in m")
    bed_diameter: float = Field(..., gt=0, description="Bed diameter in m")
    particle_radius: float = Field(1.5e-3, gt=0, description="Particle radius in m")
    bed_porosity: float = Field(0.4, gt=0, lt=1, description="Bed porosity")
    particle_porosity: float = Field(0.7, gt=0, lt=1, description="Particle porosity")

    flow_rate_m3_h: float = Field(..., gt=0, description="Flow rate in m³/h")
    inlet_concentration_mg_m3: float = Field(..., gt=0, description="Inlet concentration in mg/m³")
    molecular_weight: float = Field(92.0, gt=0, description="Molecular weight in g/mol")

    isotherm_model: str = Field("langmuir", description="Isotherm model")
    isotherm_K: float = Field(..., gt=0, description="Equilibrium constant")
    isotherm_qmax: float = Field(..., gt=0, description="Max capacity in mol/m³")
    isotherm_n: float = Field(1.0, gt=0, description="Freundlich exponent")

    simulation_time_h: float = Field(24.0, gt=0, description="Simulation time in hours")
    n_points: int = Field(100, ge=10, le=500, description="Output points")


class CADETSimulationResponse(BaseModel):
    """Response from CADET simulation."""

    success: bool
    model_type: str = "GRM"
    breakthrough_time_h: float
    saturation_time_h: float
    bed_utilization: float
    curve: List[Dict[str, float]]
    computation_time_s: float
    error: Optional[str] = None
    warnings: List[str] = []
    parameters: Dict[str, Any] = {}


class CADETComparisonRequest(BaseModel):
    """Request for comparing CADET with Wheeler-Jonas."""

    bed_length: float = Field(..., gt=0)
    bed_diameter: float = Field(..., gt=0)
    particle_radius: float = Field(1.5e-3, gt=0)
    bed_porosity: float = Field(0.4, gt=0, lt=1)
    particle_porosity: float = Field(0.7, gt=0, lt=1)

    flow_rate_m3_h: float = Field(..., gt=0)
    inlet_concentration_mg_m3: float = Field(..., gt=0)
    molecular_weight: float = Field(92.0, gt=0)

    isotherm_K: float = Field(..., gt=0)
    isotherm_qmax: float = Field(..., gt=0)

    simulation_time_h: float = Field(24.0, gt=0)
    n_points: int = Field(100, ge=10, le=500)

    wj_breakthrough_h: float = Field(..., gt=0, description="Wheeler-Jonas breakthrough time")
    wj_saturation_h: float = Field(..., gt=0, description="Wheeler-Jonas saturation time")


class CADETComparisonResponse(BaseModel):
    """Response from model comparison."""

    comparison_available: bool
    error: Optional[str] = None
    wheeler_jonas: Dict[str, float]
    cadet: Optional[Dict[str, Any]] = None
    deviations: Optional[Dict[str, float]] = None
    recommendations: List[str] = []
    computation_time_s: float = 0.0


class CADETStatusResponse(BaseModel):
    """CADET installation status."""

    available: bool
    error: Optional[str] = None
    version: Optional[str] = None
    simulator: Optional[str] = None


class CADETModelInfo(BaseModel):
    """Information about a CADET binding model."""

    id: str
    name: str
    description: str
    parameters: List[str]
    recommended_for: str
    available: bool


@router.get("/status", response_model=CADETStatusResponse)
async def get_status() -> CADETStatusResponse:
    """
    Check CADET-Process installation status.

    Returns whether CADET is available, version info, and any errors.
    """
    status = get_cadet_status()
    return CADETStatusResponse(**status)


@router.get("/models", response_model=List[CADETModelInfo])
async def list_models() -> List[CADETModelInfo]:
    """
    List available CADET binding models.

    Returns information about supported isotherm models for simulation.
    """
    models = get_available_models()
    return [CADETModelInfo(**m) for m in models]


@router.post("/simulate", response_model=CADETSimulationResponse)
async def simulate_breakthrough(request: CADETSimulationRequest) -> CADETSimulationResponse:
    """
    Run high-precision breakthrough simulation using CADET GRM.

    Uses the General Rate Model with full mass transfer considerations:
    - Axial dispersion
    - Film mass transfer
    - Pore diffusion
    - Surface adsorption kinetics

    Returns breakthrough curve with timing analysis.
    """
    available, error = check_cadet_available()
    if not available:
        return CADETSimulationResponse(
            success=False,
            breakthrough_time_h=0,
            saturation_time_h=0,
            bed_utilization=0,
            curve=[],
            computation_time_s=0,
            error=error or "CADET not available. Install with: pip install cadet-process",
        )

    config = CADETConfig(
        bed_length=request.bed_length,
        bed_diameter=request.bed_diameter,
        particle_radius=request.particle_radius,
        bed_porosity=request.bed_porosity,
        particle_porosity=request.particle_porosity,
        flow_rate_m3_h=request.flow_rate_m3_h,
        inlet_concentration_mg_m3=request.inlet_concentration_mg_m3,
        molecular_weight=request.molecular_weight,
        isotherm_model=request.isotherm_model,
        isotherm_K=request.isotherm_K,
        isotherm_qmax=request.isotherm_qmax,
        isotherm_n=request.isotherm_n,
        simulation_time_h=request.simulation_time_h,
        n_points=request.n_points,
    )

    service = CADETBreakthroughService(timeout=60)
    result = service.simulate(config)

    return CADETSimulationResponse(
        success=result.success,
        model_type=result.model_type,
        breakthrough_time_h=result.breakthrough_time_h,
        saturation_time_h=result.saturation_time_h,
        bed_utilization=result.bed_utilization,
        curve=result.curve,
        computation_time_s=result.computation_time_s,
        error=result.error,
        warnings=result.warnings,
        parameters=result.parameters,
    )


@router.post("/compare", response_model=CADETComparisonResponse)
async def compare_models(request: CADETComparisonRequest) -> CADETComparisonResponse:
    """
    Compare CADET GRM results with Wheeler-Jonas model.

    Runs CADET simulation and compares breakthrough/saturation times
    with provided Wheeler-Jonas results. Returns deviation analysis
    and recommendations.
    """
    available, error = check_cadet_available()
    if not available:
        return CADETComparisonResponse(
            comparison_available=False,
            error=error or "CADET not available",
            wheeler_jonas={
                "breakthrough_time_h": request.wj_breakthrough_h,
                "saturation_time_h": request.wj_saturation_h,
            },
        )

    config = CADETConfig(
        bed_length=request.bed_length,
        bed_diameter=request.bed_diameter,
        particle_radius=request.particle_radius,
        bed_porosity=request.bed_porosity,
        particle_porosity=request.particle_porosity,
        flow_rate_m3_h=request.flow_rate_m3_h,
        inlet_concentration_mg_m3=request.inlet_concentration_mg_m3,
        molecular_weight=request.molecular_weight,
        isotherm_K=request.isotherm_K,
        isotherm_qmax=request.isotherm_qmax,
        simulation_time_h=request.simulation_time_h,
        n_points=request.n_points,
    )

    service = CADETBreakthroughService(timeout=60)
    comparison = service.compare_with_wheeler_jonas(
        config=config,
        wj_breakthrough_h=request.wj_breakthrough_h,
        wj_saturation_h=request.wj_saturation_h,
    )

    return CADETComparisonResponse(**comparison)
