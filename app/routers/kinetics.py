"""
Kinetics Router - API endpoints for breakthrough curve kinetic models

Provides endpoints for:
- Wheeler-Jonas model calculation
- Thomas model calculation
- Yoon-Nelson model calculation
- Bohart-Adams model calculation
- Model comparison
"""

from fastapi import APIRouter, HTTPException
from app.models.schemas import (
    ThomasRequest,
    YoonNelsonRequest,
    BohartAdamsRequest,
    WheelerJonasKineticsRequest,
    KineticsCompareRequest,
    KineticsResult,
    KineticsCompareResult,
    KineticModel,
    BreakthroughTimes,
)
from app.services.kinetics import (
    calculate_thomas_breakthrough,
    calculate_yoon_nelson_breakthrough,
    calculate_bohart_adams_breakthrough,
)
from app.services.wheeler_jonas import (
    WheelerJonasParams,
    wheeler_jonas_breakthrough_time,
    generate_breakthrough_curve_wj,
    calculate_mass_transfer_zone,
    estimate_kv_from_velocity,
)

router = APIRouter(prefix="/api/kinetics", tags=["kinetics"])


def _calculate_wheeler_jonas(
    flow_rate: float,
    bed_mass: float,
    bed_height: float,
    C0: float,
    C_out: float | None,
    W_e: float,
    k_v: float | None,
    rho_bulk: float,
    velocity: float,
    particle_diameter: float | None,
    n_points: int,
) -> dict:
    """Calculate Wheeler-Jonas breakthrough and return standardized result."""
    # Default breakthrough concentration (5% of inlet)
    if C_out is None or C_out <= 0:
        C_out = C0 * 0.05

    # Estimate k_v if not provided
    if k_v is None:
        dp = particle_diameter / 1000 if particle_diameter else 0.003  # mm to m
        k_v = estimate_kv_from_velocity(velocity, dp, voidage=0.4)

    Q_m3_min = flow_rate / 60  # m³/h to m³/min

    params = WheelerJonasParams(
        W_e=W_e,
        k_v=k_v,
        rho_b=rho_bulk,
        Q=Q_m3_min,
        C_in=C0,
        C_out=C_out,
        W_bed=bed_mass,
    )

    # Calculate breakthrough time
    t_b_min = wheeler_jonas_breakthrough_time(params)
    t_b_h = t_b_min / 60

    # Calculate MTZ
    mtz = calculate_mass_transfer_zone(params, bed_height)

    # Generate breakthrough curve
    curve_data = generate_breakthrough_curve_wj(params, n_points)

    # Calculate breakthrough times at different C/C0 levels
    def find_time_for_cc0(target_cc0: float) -> float:
        params_t = WheelerJonasParams(
            W_e=W_e, k_v=k_v, rho_b=rho_bulk, Q=Q_m3_min,
            C_in=C0, C_out=C0 * target_cc0, W_bed=bed_mass,
        )
        return wheeler_jonas_breakthrough_time(params_t) / 60  # hours

    t_5 = find_time_for_cc0(0.05)
    t_10 = find_time_for_cc0(0.10)
    t_50 = find_time_for_cc0(0.50)
    t_90 = find_time_for_cc0(0.90)

    return {
        "parameters": {
            "W_e": W_e,
            "k_v": k_v,
            "rho_bulk": rho_bulk,
            "C0": C0,
            "C_out": C_out,
            "bed_mass": bed_mass,
        },
        "breakthrough_times": {
            "t_5_h": t_5,
            "t_10_h": t_10,
            "t_50_h": t_50,
            "t_90_h": t_90,
            "MTZ_h": mtz,
        },
        "curve": [{"time_h": t / 60, "C_C0": c} for t, c in curve_data],
    }


def _convert_to_kinetics_result(result: dict, model_name: str) -> KineticsResult:
    """Convert service result to KineticsResult schema."""
    bt = result["breakthrough_times"]

    # Calculate MTZ in meters (approximate)
    v = 0.3  # m/s default velocity
    mtz_h = bt.get("MTZ_h", 0)
    mtz_m = mtz_h * 3600 * v * 0.1  # Approximate MTZ length

    return KineticsResult(
        model=model_name,
        parameters=result["parameters"],
        breakthrough_times=BreakthroughTimes(
            t_5=bt["t_5_h"],
            t_10=bt["t_10_h"],
            t_50=bt["t_50_h"],
            t_90=bt["t_90_h"],
            MTZ=mtz_m,
        ),
        curve=[{"time_h": p["time_h"], "C_C0": p["C_C0"]} for p in result["curve"]],
        service_time_h=bt["t_10_h"],
    )


@router.post("/wheeler-jonas/calculate", response_model=KineticsResult)
async def calculate_wheeler_jonas(request: WheelerJonasKineticsRequest) -> KineticsResult:
    """
    Calculate breakthrough curve using Wheeler-Jonas model.

    The Wheeler-Jonas model is the industry standard for activated carbon filter design.
    Best for: VOC adsorption, industrial air treatment, regulatory compliance.
    """
    try:
        result = _calculate_wheeler_jonas(
            flow_rate=request.flow_rate,
            bed_mass=request.bed_mass,
            bed_height=request.bed_height,
            C0=request.C0,
            C_out=request.C_out,
            W_e=request.W_e,
            k_v=request.k_v,
            rho_bulk=request.rho_bulk,
            velocity=request.velocity,
            particle_diameter=request.particle_diameter,
            n_points=request.n_points,
        )
        return _convert_to_kinetics_result(result, "Wheeler-Jonas")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Wheeler-Jonas calculation error: {str(e)}")


@router.post("/thomas/calculate", response_model=KineticsResult)
async def calculate_thomas(request: ThomasRequest) -> KineticsResult:
    """
    Calculate breakthrough curve using Thomas model.

    The Thomas model assumes Langmuir isotherm and second-order kinetics.
    Best for: Low to medium concentrations, symmetrical breakthrough curves.
    """
    try:
        result = calculate_thomas_breakthrough(
            flow_rate=request.flow_rate,
            bed_mass=request.bed_mass,
            C0=request.C0,
            q0=request.q0,
            k_Th=request.k_Th,
            v=request.v,
            d_p=request.d_p,
            pollutant_type=request.pollutant_type,
            n_points=request.n_points,
        )
        return _convert_to_kinetics_result(result, "Thomas")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Thomas calculation error: {str(e)}")


@router.post("/yoon-nelson/calculate", response_model=KineticsResult)
async def calculate_yoon_nelson(request: YoonNelsonRequest) -> KineticsResult:
    """
    Calculate breakthrough curve using Yoon-Nelson model.

    The Yoon-Nelson model is simpler than Thomas and requires fewer parameters.
    Best for: Single-component adsorption, quick estimations.
    """
    try:
        result = calculate_yoon_nelson_breakthrough(
            flow_rate=request.flow_rate,
            bed_mass=request.bed_mass,
            C0=request.C0,
            q0=request.q0,
            k_YN=request.k_YN,
            tau=request.tau,
            pollutant_type=request.pollutant_type,
            n_points=request.n_points,
        )
        return _convert_to_kinetics_result(result, "Yoon-Nelson")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Yoon-Nelson calculation error: {str(e)}")


@router.post("/bohart-adams/calculate", response_model=KineticsResult)
async def calculate_bohart_adams(request: BohartAdamsRequest) -> KineticsResult:
    """
    Calculate breakthrough curve using Bohart-Adams model.

    The Bohart-Adams model considers bed depth and is good for initial breakthrough.
    Best for: Low concentration effluents, early breakthrough prediction.
    """
    try:
        result = calculate_bohart_adams_breakthrough(
            flow_rate=request.flow_rate,
            bed_height=request.bed_height,
            bed_area=request.bed_area,
            C0=request.C0,
            N0=request.N0,
            k_BA=request.k_BA,
            rho_bulk=request.rho_bulk,
            pollutant_type=request.pollutant_type,
            n_points=request.n_points,
        )
        return _convert_to_kinetics_result(result, "Bohart-Adams")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Bohart-Adams calculation error: {str(e)}")


@router.post("/compare", response_model=KineticsCompareResult)
async def compare_models(request: KineticsCompareRequest) -> KineticsCompareResult:
    """
    Compare all four kinetic models with the same input parameters.

    Returns breakthrough curves from Wheeler-Jonas, Thomas, Yoon-Nelson, and Bohart-Adams models
    along with a recommendation for which model to use.
    """
    try:
        # Calculate Wheeler-Jonas (industry standard)
        W_e = request.W_e if request.W_e is not None else request.q0
        wheeler_jonas_result = _calculate_wheeler_jonas(
            flow_rate=request.flow_rate,
            bed_mass=request.bed_mass,
            bed_height=request.bed_height,
            C0=request.C0,
            C_out=None,  # Default 5% breakthrough
            W_e=W_e,
            k_v=None,  # Auto-estimate
            rho_bulk=request.rho_bulk,
            velocity=request.velocity,
            particle_diameter=request.particle_diameter,
            n_points=request.n_points,
        )

        # Calculate Thomas
        thomas_result = calculate_thomas_breakthrough(
            flow_rate=request.flow_rate,
            bed_mass=request.bed_mass,
            C0=request.C0,
            q0=request.q0,
            pollutant_type=request.pollutant_type,
            n_points=request.n_points,
        )

        # Calculate Yoon-Nelson
        yoon_nelson_result = calculate_yoon_nelson_breakthrough(
            flow_rate=request.flow_rate,
            bed_mass=request.bed_mass,
            C0=request.C0,
            q0=request.q0,
            pollutant_type=request.pollutant_type,
            n_points=request.n_points,
        )

        # Calculate Bohart-Adams
        bohart_adams_result = calculate_bohart_adams_breakthrough(
            flow_rate=request.flow_rate,
            bed_height=request.bed_height,
            bed_area=request.bed_area,
            C0=request.C0,
            rho_bulk=request.rho_bulk,
            pollutant_type=request.pollutant_type,
            n_points=request.n_points,
        )

        # Determine recommended model based on conditions
        recommended_model = "Wheeler-Jonas"
        recommendation_reason = "Wheeler-Jonas is the industry standard for activated carbon filter design and regulatory compliance."

        # Heuristics for recommendation
        if request.C0 < 50:  # Low concentration
            recommended_model = "Bohart-Adams"
            recommendation_reason = "Bohart-Adams is recommended for low concentration applications (< 50 mg/m³), especially for initial breakthrough prediction."
        elif request.C0 > 500:  # High concentration
            recommended_model = "Wheeler-Jonas"
            recommendation_reason = "Wheeler-Jonas is most reliable for high concentration VOC applications (> 500 mg/m³) in industrial air treatment."
        elif request.bed_height > 0.5 and request.q0 < 0.1:  # Deep bed, low capacity
            recommended_model = "Yoon-Nelson"
            recommendation_reason = "Yoon-Nelson provides good approximation for deep beds with moderate adsorption capacity."

        return KineticsCompareResult(
            wheeler_jonas=_convert_to_kinetics_result(wheeler_jonas_result, "Wheeler-Jonas"),
            thomas=_convert_to_kinetics_result(thomas_result, "Thomas"),
            yoon_nelson=_convert_to_kinetics_result(yoon_nelson_result, "Yoon-Nelson"),
            bohart_adams=_convert_to_kinetics_result(bohart_adams_result, "Bohart-Adams"),
            recommended_model=recommended_model,
            recommendation_reason=recommendation_reason,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison error: {str(e)}")


@router.get("/models", response_model=list[KineticModel])
async def list_models() -> list[KineticModel]:
    """
    List all available kinetic models with descriptions.
    """
    return [
        KineticModel(
            name="Wheeler-Jonas",
            description="Industry standard for activated carbon filter design. Based on mass transfer zone theory with equilibrium capacity.",
            equation="t_b = (W_e × W) / (C_in × Q) - (W_e × ρ_b) / (k_v × C_in) × ln((C_in - C_b) / C_b)",
            parameters=["W_e (equilibrium capacity)", "k_v (mass transfer coeff.)", "ρ_b (bulk density)", "W (bed mass)", "Q (flow rate)"],
            best_for="VOC adsorption, industrial air treatment, regulatory compliance (ICPE, TA-Luft)",
        ),
        KineticModel(
            name="Thomas",
            description="Most widely used model for fixed-bed adsorption. Assumes Langmuir isotherm and second-order reversible kinetics.",
            equation="C/C0 = 1 / (1 + exp(k_Th/Q × (q0×m - C0×Q×t)))",
            parameters=["k_Th (rate constant)", "q0 (max capacity)", "m (bed mass)", "Q (flow rate)", "C0 (inlet conc.)"],
            best_for="Low to medium concentrations, symmetrical breakthrough curves, VOC adsorption",
        ),
        KineticModel(
            name="Yoon-Nelson",
            description="Simpler model requiring fewer parameters. Based on assumption that decrease in adsorption probability is proportional to adsorbate breakthrough.",
            equation="C/C0 = exp(k_YN × (t - τ)) / (1 + exp(k_YN × (t - τ)))",
            parameters=["k_YN (rate constant)", "τ (time for 50% breakthrough)"],
            best_for="Single-component adsorption, quick estimations, symmetric curves",
        ),
        KineticModel(
            name="Bohart-Adams",
            description="Based on surface reaction rate theory. Good for predicting initial part of breakthrough curve.",
            equation="ln(C/C0) = k_BA × C0 × t - k_BA × N0 × Z / v",
            parameters=["k_BA (rate constant)", "N0 (saturation conc.)", "Z (bed depth)", "v (velocity)"],
            best_for="Low concentration effluents, initial breakthrough prediction, BDST analysis",
        ),
    ]
