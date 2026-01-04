"""Breakthrough curve API endpoints"""
from fastapi import APIRouter, HTTPException
from typing import List, Optional
from pydantic import BaseModel, Field

from app.services.wheeler_jonas import (
    WheelerJonasParams,
    wheeler_jonas_breakthrough_time,
    generate_breakthrough_curve,
    estimate_kv_from_velocity,
    calculate_mass_transfer_zone,
    calculate_bed_utilization,
)

router = APIRouter()


class BreakthroughRequest(BaseModel):
    """Request for breakthrough calculation"""

    # Capacity and kinetics
    adsorption_capacity: float = Field(..., gt=0, description="mg/g")
    kinetic_constant: Optional[float] = Field(None, description="1/min, auto-estimated if not provided")

    # Bed properties
    bulk_density: float = Field(..., gt=0, description="kg/m³")
    bed_mass: float = Field(..., gt=0, description="kg")
    bed_height: float = Field(..., gt=0, description="m")
    bed_voidage: float = Field(0.4, gt=0, lt=1)
    particle_diameter: float = Field(3.0, gt=0, description="mm")

    # Operating conditions
    flow_rate: float = Field(..., gt=0, description="m³/h")
    inlet_concentration: float = Field(..., gt=0, description="mg/m³")
    breakthrough_concentration: float = Field(..., gt=0, description="mg/m³")

    # Curve generation
    n_points: int = Field(50, ge=10, le=200)


class BreakthroughResult(BaseModel):
    """Breakthrough calculation result"""

    breakthrough_time_hours: float
    saturation_time_hours: float
    mass_transfer_zone_m: float
    bed_utilization: float
    curve: List[dict]  # List of {time_hours, c_c0}


@router.post("/breakthrough/calculate", response_model=BreakthroughResult)
async def calculate_breakthrough(request: BreakthroughRequest) -> BreakthroughResult:
    """
    Calculate breakthrough curve using Wheeler-Jonas model.

    Args:
        request: BreakthroughRequest with bed and operating parameters

    Returns:
        BreakthroughResult with breakthrough time and curve
    """
    try:
        # Convert units
        Q_min = request.flow_rate / 60  # m³/min
        dp_m = request.particle_diameter / 1000  # m

        # Calculate velocity
        import math

        cross_section = request.bed_mass / (request.bulk_density * request.bed_height)
        velocity = (request.flow_rate / 3600) / cross_section

        # Estimate k_v if not provided
        if request.kinetic_constant is not None:
            k_v = request.kinetic_constant
        else:
            k_v = estimate_kv_from_velocity(velocity, dp_m, request.bed_voidage)

        # Wheeler-Jonas parameters
        params = WheelerJonasParams(
            W_e=request.adsorption_capacity / 1000,  # g/g
            k_v=k_v,
            rho_b=request.bulk_density,
            Q=Q_min,
            C_in=request.inlet_concentration,
            C_out=request.breakthrough_concentration,
            W_bed=request.bed_mass,
        )

        # Calculate breakthrough time
        t_b = wheeler_jonas_breakthrough_time(params)  # minutes
        t_b_hours = t_b / 60

        # Calculate saturation time (at 95% breakthrough)
        params_sat = WheelerJonasParams(
            W_e=params.W_e,
            k_v=params.k_v,
            rho_b=params.rho_b,
            Q=params.Q,
            C_in=params.C_in,
            C_out=0.95 * params.C_in,
            W_bed=params.W_bed,
        )
        t_sat = wheeler_jonas_breakthrough_time(params_sat) / 60

        # MTZ
        mtz = calculate_mass_transfer_zone(params, request.bed_height)

        # Bed utilization
        utilization = calculate_bed_utilization(params, request.bed_height)

        # Generate curve
        curve_data = generate_breakthrough_curve(params, n_points=request.n_points)
        curve = [{"time_hours": t / 60, "c_c0": c} for t, c in curve_data]

        return BreakthroughResult(
            breakthrough_time_hours=t_b_hours,
            saturation_time_hours=t_sat,
            mass_transfer_zone_m=mtz,
            bed_utilization=utilization,
            curve=curve,
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/breakthrough/models")
async def list_breakthrough_models():
    """
    List available breakthrough models.
    """
    return {
        "models": {
            "wheeler_jonas": {
                "name": "Wheeler-Jonas",
                "description": "Most common model for activated carbon, based on kinetic approach",
                "parameters": ["W_e", "k_v", "rho_b"],
                "equation": "t_b = (W_e·W)/(Q·C_in) - (W_e·ρ_b)/(k_v·C_in)·ln((C_in-C_out)/C_out)",
            },
            "thomas": {
                "name": "Thomas",
                "description": "Based on Langmuir kinetics and second-order reaction",
                "parameters": ["k_T", "q_0"],
                "equation": "C/C_0 = 1/(1 + exp(k_T·q_0·m/Q - k_T·C_0·t))",
            },
            "yoon_nelson": {
                "name": "Yoon-Nelson",
                "description": "Simplified model, fewer parameters",
                "parameters": ["k_YN", "τ"],
                "equation": "ln(C/(C_0-C)) = k_YN·t - k_YN·τ",
            },
            "bohart_adams": {
                "name": "Bohart-Adams",
                "description": "For deep bed adsorption, surface reaction limited",
                "parameters": ["k_BA", "N_0"],
                "equation": "ln(C_0/C - 1) = k_BA·N_0·Z/v - k_BA·C_0·t",
            },
        }
    }
